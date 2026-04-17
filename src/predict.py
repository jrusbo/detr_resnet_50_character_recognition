import argparse
import json
from pathlib import Path

import torch
from transformers import DetrImageProcessor
from models import DetrResnet50
from dataset import InferenceImageDataset

NUM_CLASSES = 10
BATCH_SIZE = 4
THRESHOLD = 0.5


def predict():
    parser = argparse.ArgumentParser(description="Inference for DETR ResNet-50 on digit detection")
    parser.add_argument("--weights", "-w", type=str, required=True,
                        help="Path to trained model weights (.pt file)")
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Test images directory")
    parser.add_argument("--output", "-o", type=str,
                        help="Output file path (defaults to weights directory / pred.json)")
    args = parser.parse_args()

    if not args.output:
        args.output = Path(args.weights).parent / "pred.json"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = DetrResnet50(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    dataset = InferenceImageDataset(root_dir=args.data)
    results = []

    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(dataset))
            batch_images, batch_targets = zip(*[dataset[i] for i in range(start, end)])

            encoding = processor(images=list(batch_images), return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)
            pixel_mask = encoding.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device)

            outputs = model(pixel_values, pixel_mask=pixel_mask)

            target_sizes = [t["orig_size"] for t in batch_targets]
            results_hf = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=THRESHOLD
            )

            for target, result in zip(batch_targets, results_hf):
                image_id = target["image_id"]
                for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                    x_min, y_min, x_max, y_max = box.tolist()
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label.item()),
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": float(score.item()),
                    })

    with open(str(args.output), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    predict()
