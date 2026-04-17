import json
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from transformers import DetrImageProcessor
import albumentations as A

TRAIN_TRANSFORMS = A.Compose([
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.8),
    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=0, p=0.6),
    A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.5),
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
            self.annotations[ann['image_id']].append(ann)

        self.image_ids = list(self.images.keys())

        if self.is_train:
            self.transform = TRAIN_TRANSFORMS
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = Path(self.root_dir) / img_info['file_name']

        image = np.array(Image.open(img_path).convert("RGB"))
        anns = self.annotations[img_id]

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
            anns = new_anns

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
    Collator that applies the DetrImageProcessor to a batch of raw images and annotations.
    """
    def __init__(self, processor=None, min_size=800, max_size=1333):
        if processor is None:
            self.processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                size={"shortest_edge": min_size, "longest_edge": max_size}
            )
        else:
            self.processor = processor

    def __call__(self, batch):
        images, targets = zip(*batch)

        if "annotations" in targets[0]:
            encoding = self.processor(images=list(images), annotations=list(targets), return_tensors="pt")
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
