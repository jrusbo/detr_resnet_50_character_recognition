import argparse
import json
from pathlib import Path
from transformers import DetrImageProcessor

import torch
from models import DetrResnet50
from torch.utils.data import DataLoader
from dataset import InferenceImageDataset, DetrCollator

NUM_CLASSES = 10
BATCH_SIZE = 4
THRESHOLD = 0.5

def predict():
    parser = argparse.ArgumentParser(description="Inference for DETR ResNet-50 on digit detection")
    parser.add_argument("--weights", "-w", type=str, required=True,
                        help="Path to trained model weights")
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
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=DetrCollator(processor)
    )

    results = []

    with torch.no_grad():
        for pixel_values, pixel_mask, _, raw_targets in data_loader:
            pixel_values = pixel_values.to(device)
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device)

            outputs = model(pixel_values, pixel_mask=pixel_mask)

            target_sizes = [t["orig_size"] for t in raw_targets]
            results_hf = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=THRESHOLD
            )

            for target, result in zip(raw_targets, results_hf):
                image_id = target["image_id"]
                for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                    # DETR outputs [x_min, y_min, x_max, y_max].
                    # Task requires [x_min, y_min, width, height] without normalization.
                    x_min, y_min, x_max, y_max = box.tolist()
                    w = x_max - x_min
                    h = y_max - y_min

                    results.append({
                        "image_id": image_id,
                        "category_id": int(label.item()),
                        "bbox": [x_min, y_min, w, h],
                        "score": float(score.item())
                    })

    with open(str(args.output), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    predict()
