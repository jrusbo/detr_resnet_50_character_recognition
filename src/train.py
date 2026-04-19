import io
import contextlib
import argparse
import math
import re
import time
from typing import cast
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DeformableDetrImageProcessor, TrainerCallback, EarlyStoppingCallback
from dataset import CocoDetectionDataset, DetrCollator
from models import DetrResnet50
from transformers import Trainer, TrainingArguments

# --- Globals ---
NUM_CLASSES = 10
MAP_EVAL_FREQ = 5          # run full COCO mAP eval every N epochs (expensive; skip the rest)
LOG_STEPS = 50             # how often to log training loss
CHECKPOINT_LIMIT = 3       # max HF checkpoints kept on disk
MAP_SCORE_THRESHOLD = 0.2  # Score threshold for mAP evaluation (keep low to maximize metric sensitivity, separate from final PR/confusion matrix plots)

SAMPLER_ALPHA = 0.5
SAMPLER_CAP = 0.6
SAMPLER_MAX_IMAGE_WEIGHT = 1.8


def _extract_checkpoint_step(path):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(run_dir):
    checkpoints = [p for p in Path(run_dir).iterdir() if p.is_dir() and _extract_checkpoint_step(p) >= 0]
    if not checkpoints:
        return None
    return max(checkpoints, key=_extract_checkpoint_step)


def validate_resume_checkpoint(checkpoint_dir, fp16_enabled):
    """Ensure checkpoint has model + trainer/optimizer/scheduler/rng state for exact resume."""
    checkpoint_dir = Path(checkpoint_dir)
    has_torch_model = (checkpoint_dir / "pytorch_model.bin").exists()
    has_safe_model = (checkpoint_dir / "model.safetensors").exists()
    if not (has_torch_model or has_safe_model):
        raise ValueError(
            "Checkpoint missing model weights file (expected pytorch_model.bin or model.safetensors): "
            f"{checkpoint_dir}"
        )

    required_files = [
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    ]
    if fp16_enabled:
        required_files.append("scaler.pt")

    missing = [name for name in required_files if not (checkpoint_dir / name).exists()]
    if missing:
        raise ValueError(
            "Checkpoint missing required resume files: "
            f"{', '.join(missing)} at {checkpoint_dir}"
        )

    has_single_rng = (checkpoint_dir / "rng_state.pth").exists()
    has_sharded_rng = any(checkpoint_dir.glob("rng_state_*.pth"))
    if not (has_single_rng or has_sharded_rng):
        raise ValueError(
            "Checkpoint missing RNG state file(s): expected rng_state.pth "
            f"or rng_state_*.pth at {checkpoint_dir}"
        )


def build_class_balanced_image_weights(dataset, num_classes):
    """Build per-image sampling weights from class frequencies (multi-class aware)."""
    class_counts = np.zeros(num_classes, dtype=np.float64)
    for anns in dataset.annotations.values():
        for ann in anns:
            cat = int(ann["category_id"])
            if 0 <= cat < num_classes:
                class_counts[cat] += 1.0

    nonzero = class_counts[class_counts > 0]
    if nonzero.size == 0:
        return [1.0] * len(dataset), class_counts.tolist(), [1.0] * num_classes

    max_count = float(nonzero.max())
    class_weights = np.ones(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if class_counts[c] > 0:
            class_weights[c] = (max_count / class_counts[c]) ** SAMPLER_ALPHA

    image_weights = []
    for image_id in dataset.image_ids:
        anns = dataset.annotations.get(image_id, [])
        classes = sorted({int(ann["category_id"]) for ann in anns if 0 <= int(ann["category_id"]) < num_classes})
        if not classes:
            image_weights.append(1.0)
            continue

        mean_class_weight = float(np.mean([class_weights[c] for c in classes]))
        w_img = 1.0 + SAMPLER_CAP * mean_class_weight
        w_img = min(SAMPLER_MAX_IMAGE_WEIGHT, max(1.0, w_img))
        image_weights.append(w_img)

    return image_weights, class_counts.tolist(), class_weights.tolist()


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

    # Dynamically scale Y-axis to ignore the initial random-initialization loss spike
    if len(train_losses) > 5:
        # Peak of valid convergence (ignore first few log steps completely)
        tail_max = max(train_losses[5:])
        if eval_losses:
            tail_max = max(tail_max, max(eval_losses))
        plt.ylim(0.0, float(tail_max) * 1.2)

    plt.savefig(output_dir / "loss_curves.png")
    plt.close()


def plot_confusion_matrix(cm, classes, output_dir):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap("Blues"))
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


def _run_coco_inference(model, dataset, device, processor, batch_size, score_threshold=MAP_SCORE_THRESHOLD):
    """Batched inference; returns raw COCO prediction dicts."""
    model.eval()
    results = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Running Inference"):
        end = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        batch_images, batch_targets = zip(*batch)
        inputs = processor(images=list(batch_images), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        res_batch = processor.post_process_object_detection(
            out, target_sizes=[t["orig_size"] for t in batch_targets], threshold=score_threshold
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
                    "category_id": int(label) + 1,  # Map 0-9 back to 1-10 for COCO format
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
    return results


def compute_coco_map(model, dataset, device, processor, batch_size, score_threshold=MAP_SCORE_THRESHOLD):
    """Returns AP@IoU=0.50 without printing COLO eval tables."""
    results = _run_coco_inference(
        model, dataset, device, processor, batch_size, score_threshold=score_threshold
    )
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


def evaluate_metrics(model, dataset, output_dir, device, batch_size, processor, score_threshold):
    model.eval()

    print("Running inference for COCO evaluation...")
    results = _run_coco_inference(
        model, dataset, device, processor, batch_size, score_threshold=score_threshold
    )

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
            # Re-subtract the 1-10 mapped ID to plot correctly onto the 0-9 metric plots
            dt_cat = int(dt_ann['category_id']) - 1
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
                gt_cat = int(gt_anns[best_gt_idx]['category_id']) - 1
                cm[gt_cat, dt_cat] += 1
                matched_gt.add(best_gt_idx)
            else:
                cm[NUM_CLASSES, dt_cat] += 1

        for idx, gt_ann in enumerate(gt_anns):
            if idx not in matched_gt:
                cm[int(gt_ann['category_id']) - 1, NUM_CLASSES] += 1

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

    def __init__(self, eval_dataset, processor, output_dir, eval_batch_size, eval_freq=MAP_EVAL_FREQ):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.eval_batch_size = eval_batch_size
        self.eval_freq = eval_freq
        self.best_map = 0.0

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        current_epoch = round(state.epoch)
        is_last_epoch = current_epoch >= int(args.num_train_epochs)
        if current_epoch % self.eval_freq != 0 and not is_last_epoch:
            return control

        map50 = compute_coco_map(model, self.eval_dataset, args.device, self.processor, self.eval_batch_size)
        print(f"  eval_map50 = {map50:.4f}  (best so far: {self.best_map:.4f})")
        if map50 > self.best_map:
            self.best_map = map50
            torch.save(model.state_dict(), self.output_dir / "best_map_model.pt")
            print("  --> New best mAP@0.50, checkpoint saved.")
        return control


class TimeLimitStopCallback(TrainerCallback):
    """Gracefully stop training close to wall-clock limit and force a checkpoint save."""

    def __init__(self, max_train_hours):
        self.max_train_seconds = max_train_hours * 3600.0
        self.start_time = None
        self.triggered = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.triggered = False
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.triggered or self.start_time is None:
            return control

        elapsed = time.time() - self.start_time
        if elapsed >= self.max_train_seconds:
            self.triggered = True
            elapsed_h = elapsed / 3600.0
            print(
                f"Reached wall-clock training limit ({elapsed_h:.2f}h >= {self.max_train_seconds / 3600.0:.2f}h). "
                "Stopping and saving checkpoint for resume."
            )
            control.should_save = True
            control.should_training_stop = True
        return control


class TorchCheckpointTrainer(Trainer):
    """Trainer variant that always writes torch checkpoints (.bin) to avoid safetensors aliasing issues."""

    def __init__(self, train_sampler=None, **kwargs):
        self.train_sampler = train_sampler
        super().__init__(**kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = cast(Dataset, self.train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )


def main():
    parser = argparse.ArgumentParser("DETR training via HF Trainer")
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="Batch size for generating predictions during evaluation")
    parser.add_argument("--min-size", type=int, default=200, help="Shortest image edge for preprocessing")
    parser.add_argument("--max-size", type=int, default=400, help="Longest image edge for preprocessing")
    parser.add_argument("--lr", type=float, default=2e-4)    # Deformable DETR standard
    parser.add_argument("--lr-backbone", type=float, default=2e-5)    # Deformable DETR standard
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of total training steps to use for linear warmup")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (recommend 4 per GPU)")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint directory or run directory to resume from. "
            "If a run directory is provided, the latest checkpoint-* is used."
        ),
    )
    parser.add_argument(
        "--max-train-hours",
        type=float,
        default=11.8,
        help="Wall-clock limit for a single session; trainer stops and saves to allow resume (Kaggle-safe).",
    )
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=2,
        help="Run HF eval_loss every N epochs (reduces validation overhead vs every epoch)",
    )
    parser.add_argument(
        "--map-eval-freq",
        type=int,
        default=MAP_EVAL_FREQ,
        help="Run expensive COCO mAP callback every N epochs; set <=0 to disable during training",
    )
    parser.add_argument(
        "--use-class-balanced-sampler",
        action="store_true",
        help="Enable time-neutral weighted sampling to upweight images containing rarer classes",
    )
    parser.add_argument(
        "--metrics-threshold",
        type=float,
        default=0.3,
        help="Score threshold for final confusion-matrix/PR plots (kept separate from mAP threshold)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop training after this many eval epochs without improvement in eval_loss",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum eval_loss improvement required to reset early stopping patience",
    )
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

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    calculated_warmup_steps = int(total_steps * args.warmup_ratio)
    eval_interval_epochs = max(1, int(args.eval_every_epochs))
    eval_steps = steps_per_epoch * eval_interval_epochs
    print(f"Calculated warmup steps: {calculated_warmup_steps} over {total_steps} total steps.")
    print(f"HF eval interval: every {eval_interval_epochs} epoch(s) ({eval_steps} steps).")

    model = DetrResnet50(num_classes=NUM_CLASSES)


    resume_checkpoint = None
    if args.resume_from_checkpoint:
        resume_path = Path(args.resume_from_checkpoint).resolve()
        if not resume_path.exists() or not resume_path.is_dir():
            raise ValueError(f"Resume path does not exist or is not a directory: {resume_path}")

        if _extract_checkpoint_step(resume_path) >= 0:
            resume_checkpoint = resume_path
            run_output_dir = resume_path.parent
        else:
            run_output_dir = resume_path
            resume_checkpoint = find_latest_checkpoint(run_output_dir)
            if resume_checkpoint is None:
                raise ValueError(f"No checkpoint-* directories found under run directory: {run_output_dir}")

        print(f"Resuming from checkpoint: {resume_checkpoint}")
        print(f"Run output directory: {run_output_dir}")
        validate_resume_checkpoint(resume_checkpoint, fp16_enabled=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = Path(args.output_dir) / f"run_{timestamp}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

    image_processor = DeformableDetrImageProcessor.from_pretrained(
        "SenseTime/deformable-detr",
        size={"shortest_edge": args.min_size, "longest_edge": args.max_size}
    )

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=calculated_warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        fp16=True,
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=eval_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        save_total_limit=CHECKPOINT_LIMIT,
        report_to="none",          # Disable wandb/tensorboard logging overhead
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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

    train_sampler = None
    if args.use_class_balanced_sampler:
        print(
            "Balanced sampler config (fixed): "
            f"alpha={SAMPLER_ALPHA}, cap={SAMPLER_CAP}, max_image_weight={SAMPLER_MAX_IMAGE_WEIGHT}"
        )
        image_weights, class_counts, class_weights = build_class_balanced_image_weights(
            train_dataset,
            num_classes=NUM_CLASSES,
        )
        print(f"Class counts (mapped 0-9): {class_counts}")
        print(f"Class weights: {[round(w, 4) for w in class_weights]}")
        print(
            "Image weight stats: "
            f"min={min(image_weights):.3f}, mean={float(np.mean(image_weights)):.3f}, max={max(image_weights):.3f}"
        )
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(image_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        print("Enabled class-balanced sampler (time-neutral: same samples per epoch).")

    callbacks = []
    if args.map_eval_freq > 0:
        callbacks.append(
            BestMapCheckpointCallback(
                eval_dataset,
                image_processor,
                run_output_dir,
                args.eval_batch_size,
                eval_freq=args.map_eval_freq,
            )
        )
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    if args.max_train_hours > 0:
        callbacks.append(TimeLimitStopCallback(max_train_hours=args.max_train_hours))

    trainer = TorchCheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_sampler=train_sampler,
        optimizers=(custom_optimizer, None),
        data_collator=DetrCollator(processor=image_processor),
        callbacks=callbacks,
    )

    print(f"Starting training, saving models to {run_output_dir}")
    print(f"Using device: {trainer.args.device}")
    if trainer.args.device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(trainer.args.device)}")

    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)

    # Load the checkpoint with the highest mAP@0.50 for final evaluation
    best_weights = run_output_dir / "best_map_model.pt"
    if best_weights.exists():
        model.load_state_dict(torch.load(best_weights, map_location=trainer.args.device))
        print(f"Loaded best mAP@0.50 model from {best_weights}")

    plot_losses(trainer.state.log_history, run_output_dir)
    evaluate_metrics(
        model,
        eval_dataset,
        run_output_dir,
        trainer.args.device,
        args.eval_batch_size,
        image_processor,
        score_threshold=args.metrics_threshold,
    )


if __name__ == '__main__':
    main()
