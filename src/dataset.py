import json
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from transformers import DeformableDetrImageProcessor
import albumentations as A

TRAIN_TRANSFORMS = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=0, p=0.5),
    A.GaussNoise(std_range=(0.02, 0.08), per_channel=True, p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels', 'bbox_indices'], clip=True))

class CocoDetectionDataset(Dataset):
    def __init__(self, root_dir, annotation_file, is_train=False):
        self.root_dir = root_dir
        self.is_train = is_train
        self.annotation_file = annotation_file
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)

        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = defaultdict(list)
        for ann in self.coco['annotations']:
            # Map categories from 1-10 down to 0-9 to avoid index out-of-bounds
            mapped_ann = ann.copy()
            mapped_ann['category_id'] = mapped_ann['category_id'] - 1
            self.annotations[ann['image_id']].append(mapped_ann)

        self.image_ids = list(self.images.keys())

        if self.is_train:
            self.transform = TRAIN_TRANSFORMS
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _sanitize_annotations(anns, width, height):
        """Clip COCO xywh boxes to image bounds and drop invalid boxes."""
        sanitized = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x = max(0.0, float(x))
            y = max(0.0, float(y))
            w = float(w)
            h = float(h)

            if w <= 0.0 or h <= 0.0:
                continue

            x2 = min(float(width), x + w)
            y2 = min(float(height), y + h)
            clipped_w = x2 - x
            clipped_h = y2 - y
            if clipped_w <= 1e-3 or clipped_h <= 1e-3:
                continue

            ann_copy = ann.copy()
            ann_copy['bbox'] = [x, y, clipped_w, clipped_h]
            ann_copy['area'] = clipped_w * clipped_h
            ann_copy['iscrowd'] = int(ann_copy.get('iscrowd', 0))
            sanitized.append(ann_copy)
        return sanitized

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = Path(self.root_dir) / img_info['file_name']

        image = np.array(Image.open(img_path).convert("RGB"))
        anns = self._sanitize_annotations(self.annotations[img_id], img_info['width'], img_info['height'])

        if self.transform is not None and len(anns) > 0:
            bboxes = [ann['bbox'] for ann in anns]
            class_labels = [ann['category_id'] for ann in anns]
            bbox_indices = list(range(len(anns)))

            transformed = self.transform(
                image=image, bboxes=bboxes, class_labels=class_labels, bbox_indices=bbox_indices
            )
            image = transformed['image']

            new_anns = []
            for i, bbox in enumerate(transformed['bboxes']):
                orig_idx = int(transformed['bbox_indices'][i])
                ann_copy = anns[orig_idx].copy()
                ann_copy['bbox'] = list(bbox)
                ann_copy['area'] = bbox[2] * bbox[3]
                new_anns.append(ann_copy)
            anns = self._sanitize_annotations(new_anns, image.shape[1], image.shape[0])

        image = Image.fromarray(image)

        # Ensure area and iscrowd exist in all annotations
        for ann in anns:
            if 'area' not in ann:
                ann['area'] = ann['bbox'][2] * ann['bbox'][3]
            if 'iscrowd' not in ann:
                ann['iscrowd'] = 0

        target = {'image_id': img_id, 'annotations': anns, 'orig_size': image.size[::-1]}

        return image, target

class InferenceImageDataset(Dataset):
    """
    Dataset for running inference on a folder of images without annotations.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_files = [
            f for f in self.root_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(str(img_path)).convert("RGB")

        try:
            img_id = int(img_path.stem)
        except ValueError:
            img_id = img_path.stem

        target = {'image_id': img_id, 'orig_size': image.size[::-1]}

        return image, target

class DetrCollator:
    """
    Collator that applies the DeformableDetrImageProcessor to a batch of raw images and annotations.
    """
    def __init__(self, processor=None):
        if processor is None:
            self.processor = DeformableDetrImageProcessor.from_pretrained(
                "SenseTime/deformable-detr"
            )
        else:
            self.processor = processor

    def __call__(self, batch):
        images, targets = zip(*batch)

        if "annotations" in targets[0]:
            encoding = self.processor(
                images=list(images), annotations=list(targets), return_tensors="pt"
            )
            labels = encoding["labels"]
        else:
            encoding = self.processor(images=list(images), return_tensors="pt")
            labels = None

        pixel_values = encoding["pixel_values"]
        pixel_mask = encoding.get("pixel_mask", None)

        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
        }
