import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import DeformableDetrImageProcessor
from models import DetrResnet50
from dataset import InferenceImageDataset
from torch.utils.data import DataLoader

NUM_CLASSES = 10


# Move Processor and Collate out of the inner scope so they can be pickled globally!
class InferenceCollator:
    def __init__(self, min_size, max_size):
        self.processor = DeformableDetrImageProcessor.from_pretrained(
            "SenseTime/deformable-detr",
            size={"shortest_edge": min_size, "longest_edge": max_size}
        )

    def __call__(self, batch):
        images, targets = zip(*batch)
        encoding = self.processor(images=list(images), return_tensors="pt")
        return encoding, targets


def predict():
    parser = argparse.ArgumentParser(description="Inference for DETR ResNet-50 on digit detection")
    parser.add_argument("--weights", "-w", type=str, required=True,
                        help="Path to trained model weights (.pt file)")
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Test images directory")
    parser.add_argument("--output", "-o", type=str,
                        help="Output file path (defaults to weights directory / pred.json)")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                        help="Batch size for generating predictions (speed up GPU)")
    parser.add_argument("--num-workers", "-n", type=int, default=2,
                        help="Number of CPU workers to load/process images")
    parser.add_argument("--threshold", "-t", type=float, default=0.1,
                        help="Confidence threshold (default: 0.1 for high sensitivity)")
    parser.add_argument("--min-size", type=int, default=200, help="Shortest image edge for preprocessing")
    parser.add_argument("--max-size", type=int, default=400, help="Longest image edge for preprocessing")
    args = parser.parse_args()

    if not args.output:
        args.output = Path(args.weights).parent / "pred.json"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = DetrResnet50(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    dataset = InferenceImageDataset(root_dir=args.data)
    collator = InferenceCollator(min_size=args.min_size, max_size=args.max_size)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    results = []

    # We still need the processor here for post_processing the outputs
    processor = DeformableDetrImageProcessor.from_pretrained(
        "SenseTime/deformable-detr",
        size={"shortest_edge": args.min_size, "longest_edge": args.max_size}
    )

    with torch.no_grad():
        for encoding, batch_targets in tqdm(dataloader, desc="Generating Predictions"):
            pixel_values = encoding["pixel_values"].to(device, non_blocking=True)
            pixel_mask = encoding.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device, non_blocking=True)

            outputs = model(pixel_values, pixel_mask=pixel_mask)

            target_sizes = [t["orig_size"] for t in batch_targets]
            results_hf = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=args.threshold
            )

            for target, result in zip(batch_targets, results_hf):
                image_id = target["image_id"]
                for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                    if label.item() >= NUM_CLASSES:
                        continue
                    x_min, y_min, x_max, y_max = box.tolist()
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label.item()) + 1,  # Map 0-9 back to 1-10 for COCO format
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": float(score.item()),
                    })

    with open(str(args.output), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    predict()
