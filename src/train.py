import sys
import argparse
from datetime import datetime
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor
from dataset import CocoDetectionDataset, DetrCollator
from models import DetrResnet50
from transformers import Trainer, TrainingArguments

NUM_CLASSES = 10

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

def evaluate_metrics(model, dataset, output_dir, device):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model.eval()

    print("Running inference for COCO evaluation...")
    results = []

    for i in range(len(dataset)):
        img, target = dataset[i]
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad(): out = model(**inputs)

        res = processor.post_process_object_detection(out, target_sizes=[target["orig_size"]], threshold=0.01)[0]

        boxes, scores, labels = res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if label >= NUM_CLASSES: continue
            x1, y1, x2, y2 = box
            results.append({
                "image_id": target["image_id"],
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })

    if not results:
        print("No predictions generated.")
        return

    coco_gt = COCO(dataset.annotation_file)
    coco_dt = coco_gt.loadRes(results) if len(results) > 0 else coco_gt.loadRes([])

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # The precision array has shape: (iouThrs, recThrs, catIds, areaRng, maxDets)
    # We plot the PR curve for IoU=0.50 (index 0) and all areas (index 0) and maxDets=100 (index 2)
    precisions = coco_eval.eval['precision'][0, :, :, 0, 2]
    recalls = coco_eval.params.recThrs

    plt.figure(figsize=(8, 6))
    for c in range(NUM_CLASSES):
        cat_id = coco_gt.getCatIds()[c] if c < len(coco_gt.getCatIds()) else c
        cat_idx = list(coco_gt.getCatIds()).index(cat_id)
        p = precisions[:, cat_idx]
        if p[p > -1].size > 0:
            plt.plot(recalls, p, label=f"Class {cat_id}")

    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Native COCO PR Curve (IoU=0.50)"); plt.legend(); plt.grid(True)
    plt.savefig(output_dir / "pr_curves.png"); plt.close()
    print(f"Metrics saved into {output_dir}")

def main():
    parser = argparse.ArgumentParser("DETR training via HF Trainer")
    parser.add_argument("--data-root", type=str, default="datasets", help="Path to base dataset directory (should contain train/, valid/, train.json, valid.json)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for backbone")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=0.1, help="Max gradient norm")
    parser.add_argument("--min-size", type=int, default=800, help="Min image edge size for upscaling")
    parser.add_argument("--max-size", type=int, default=1333, help="Max image edge size for upscaling")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    
    train_dataset = CocoDetectionDataset(
        root_dir=str(data_root / "train"), 
        annotation_file=str(data_root / "train.json"),
        is_train=True
    )
    
    eval_dataset = CocoDetectionDataset(
        root_dir=str(data_root / "valid"), 
        annotation_file=str(data_root / "valid.json"),
        is_train=False
    )

    model = DetrResnet50(num_classes=NUM_CLASSES)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        dataloader_num_workers=4,
        remove_unused_columns=False,
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
        data_collator=DetrCollator(min_size=args.min_size, max_size=args.max_size),
    )

    print(f"Starting training, saving models to {run_output_dir}")
    print(f"Using device: {trainer.args.device}")
    if trainer.args.device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(trainer.args.device)}")

    trainer.train()

    plot_losses(trainer.state.log_history, run_output_dir)
    evaluate_metrics(model, eval_dataset, run_output_dir, trainer.args.device)

if __name__ == '__main__':
    main()