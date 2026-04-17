import io
import sys
import contextlib
import argparse
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, TrainerCallback
from dataset import CocoDetectionDataset, DetrCollator
from models import DetrResnet50
from transformers import Trainer, TrainingArguments

# --- Globals ---
NUM_CLASSES = 10
EVAL_BATCH_SIZE = 16       # batch size used during val inference (no gradients, can be larger)
NUM_WORKERS = 8            # dataloader workers; rule of thumb: 4 per GPU
WARMUP_STEPS = 100         # fraction of total steps used for LR warm-up
MAP_EVAL_FREQ = 5          # run full COCO mAP eval every N epochs (expensive; skip the rest)
LOG_STEPS = 50             # how often to log training loss
CHECKPOINT_LIMIT = 3       # max HF checkpoints kept on disk


def plot_losses(log_history, output_dir):
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    for log in log_history:
        if "loss" in log and "step" in log:
            train_steps.append(log["step"])
            train_losses.append(log["loss"])
        if "eval_loss" in log and "step" in log:
            eval_steps.append(log["step"])
            eval_losses.append(log["eval_loss"])

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label="Train Loss", color='blue')
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Eval Loss", color='orange', marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curves.png")
    plt.close()


def plot_confusion_matrix(cm, classes, output_dir):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (IoU > 0.5)')
    plt.colorbar()
    target_names = [str(c) for c in classes] + ['Background']
    plt.xticks(np.arange(len(target_names)), target_names, rotation=45)
    plt.yticks(np.arange(len(target_names)), target_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(int(cm[i, j]), 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def _run_coco_inference(model, dataset, device, processor):
    """Batched inference; returns raw COCO prediction dicts."""
    model.eval()
    results = []
    for start in tqdm(range(0, len(dataset), EVAL_BATCH_SIZE), desc="Running Inference"):
        end = min(start + EVAL_BATCH_SIZE, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        batch_images, batch_targets = zip(*batch)
        inputs = processor(images=list(batch_images), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        res_batch = processor.post_process_object_detection(
            out, target_sizes=[t["orig_size"] for t in batch_targets], threshold=0.01
        )
        for target, res in zip(batch_targets, res_batch):
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            labels = res["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if label >= NUM_CLASSES:
                    continue
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": target["image_id"],
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
    return results


def compute_coco_map(model, dataset, device, processor):
    """Returns AP@IoU=0.50 without printing COCO eval tables."""
    results = _run_coco_inference(model, dataset, device, processor)
    if not results:
        return 0.0
    coco_gt = COCO(dataset.annotation_file)
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    return float(coco_eval.stats[1])  # AP@IoU=0.50


def evaluate_metrics(model, dataset, output_dir, device):
    processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")
    model.eval()

    print("Running inference for COCO evaluation...")
    results = _run_coco_inference(model, dataset, device, processor)

    if not results:
        print("No predictions generated.")
        return

    coco_gt = COCO(dataset.annotation_file)
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("Generating Confusion Matrix...")
    cm = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)
    for img_info in coco_gt.imgs.values():
        img_id = img_info['id']
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])
        dt_anns = coco_dt.loadAnns(dt_ann_ids)

        matched_gt = set()
        for dt_ann in sorted(dt_anns, key=lambda x: x['score'], reverse=True):
            dt_bbox = dt_ann['bbox']
            dt_cat = dt_ann['category_id']
            best_iou = 0.5
            best_gt_idx = -1

            for idx, gt_ann in enumerate(gt_anns):
                if idx in matched_gt:
                    continue
                gx, gy, gw, gh = gt_ann['bbox']
                dx, dy, dw, dh = dt_bbox
                xA = max(gx, dx)
                yA = max(gy, dy)
                xB = min(gx + gw, dx + dw)
                yB = min(gy + gh, dy + dh)
                interArea = max(0, xB - xA) * max(0, yB - yA)
                iou = interArea / float(gw * gh + dw * dh - interArea)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_gt_idx >= 0:
                gt_cat = gt_anns[best_gt_idx]['category_id']
                cm[gt_cat, dt_cat] += 1
                matched_gt.add(best_gt_idx)
            else:
                cm[NUM_CLASSES, dt_cat] += 1

        for idx, gt_ann in enumerate(gt_anns):
            if idx not in matched_gt:
                cm[gt_ann['category_id'], NUM_CLASSES] += 1

    plot_confusion_matrix(cm, range(NUM_CLASSES), output_dir)

    # precision shape: (iouThrs, recThrs, catIds, areaRng, maxDets)
    precisions = coco_eval.eval['precision'][0, :, :, 0, 2]
    recalls = coco_eval.params.recThrs

    plt.figure(figsize=(8, 6))
    for c in range(NUM_CLASSES):
        cat_id = coco_gt.getCatIds()[c] if c < len(coco_gt.getCatIds()) else c
        cat_idx = list(coco_gt.getCatIds()).index(cat_id)
        p = precisions[:, cat_idx]
        if p[p > -1].size > 0:
            plt.plot(recalls, p, label=f"Class {cat_id}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Native COCO PR Curve (IoU=0.50)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "pr_curves.png")
    plt.close()
    print(f"Metrics saved into {output_dir}")


class BestMapCheckpointCallback(TrainerCallback):
    """Saves model weights whenever mAP@0.50 improves.

    Only runs the expensive COCO eval every `eval_freq` epochs to avoid
    spending half the training budget on validation inference.
    """

    def __init__(self, eval_dataset, processor, output_dir, eval_freq=MAP_EVAL_FREQ):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.eval_freq = eval_freq
        self.best_map = 0.0

    def on_evaluate(self, args, state, control, model, **kwargs):
        current_epoch = round(state.epoch)
        is_last_epoch = current_epoch >= int(args.num_train_epochs)
        if current_epoch % self.eval_freq != 0 and not is_last_epoch:
            return

        map50 = compute_coco_map(model, self.eval_dataset, args.device, self.processor)
        print(f"  eval_map50 = {map50:.4f}  (best so far: {self.best_map:.4f})")
        if map50 > self.best_map:
            self.best_map = map50
            torch.save(model.state_dict(), self.output_dir / "best_map_model.pt")
            print("  --> New best mAP@0.50, checkpoint saved.")


def main():
    parser = argparse.ArgumentParser("DETR training via HF Trainer")
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)    # Deformable DETR standard
    parser.add_argument("--lr-backbone", type=float, default=2e-5)    # Deformable DETR standard
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="DataLoader worker processes (recommend 4 per GPU)")
    parser.add_argument("--map-eval-freq", type=int, default=MAP_EVAL_FREQ,
                        help="Run full COCO mAP eval every N epochs (1 = every epoch, expensive)")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    train_dataset = CocoDetectionDataset(
        root_dir=str(data_root / "train"),
        annotation_file=str(data_root / "train.json"),
        is_train=True,
    )
    eval_dataset = CocoDetectionDataset(
        root_dir=str(data_root / "valid"),
        annotation_file=str(data_root / "valid.json"),
        is_train=False,
    )

    model = DetrResnet50(num_classes=NUM_CLASSES)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    eval_processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        fp16=True,
        logging_steps=LOG_STEPS,
        save_strategy="epoch",
        eval_strategy="epoch",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=False,
        save_total_limit=CHECKPOINT_LIMIT,
        report_to="none",          # Disable wandb/tensorboard logging overhead
    )

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    custom_optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)

    if sys.platform == "win32":
        torch.backends.cudnn.enabled = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(custom_optimizer, None),
        data_collator=DetrCollator(),
        callbacks=[
            BestMapCheckpointCallback(
                eval_dataset, eval_processor, run_output_dir, eval_freq=args.map_eval_freq
            )
        ],
    )

    print(f"Starting training, saving models to {run_output_dir}")
    print(f"Using device: {trainer.args.device}")
    if trainer.args.device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(trainer.args.device)}")

    trainer.train()

    # Load the checkpoint with the highest mAP@0.50 for final evaluation
    best_weights = run_output_dir / "best_map_model.pt"
    if best_weights.exists():
        model.load_state_dict(torch.load(best_weights, map_location=trainer.args.device))
        print(f"Loaded best mAP@0.50 model from {best_weights}")

    plot_losses(trainer.state.log_history, run_output_dir)
    evaluate_metrics(model, eval_dataset, run_output_dir, trainer.args.device)


if __name__ == '__main__':
    main()
