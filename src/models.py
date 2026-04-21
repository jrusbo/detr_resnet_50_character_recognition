"""Model definitions for the DETR ResNet-50 digit detector."""

import torch.nn as nn
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

NUM_CLASSES = 10


class DetrResnet50(nn.Module):
    """
    Wrapper around `DeformableDetrForObjectDetection` configured for 10 classes.

    The backbone uses pretrained ResNet-50 weights while detection heads are initialized
    according to the Deformable DETR configuration.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        """Initialize model configuration and instantiate underlying transformer model."""
        super().__init__()
        config = DeformableDetrConfig(
            backbone="resnet50",
            use_pretrained_backbone=True,
            num_labels=num_classes,
            auxiliary_loss=True,
        )
        self.model = DeformableDetrForObjectDetection(config)

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        """Forward pass through the underlying Deformable DETR model."""
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)


def get_model(num_classes=NUM_CLASSES):
    """Return a configured `DetrResnet50` instance."""
    return DetrResnet50(num_classes=num_classes)
