"""
This file contains all model related code
"""
import torch.nn as nn
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

# HuggingFace DETR automatically adds +1 to num_labels for the "no object" (background) class.
# For 10 digit classes (0-9), set NUM_CLASSES = 10.
NUM_CLASSES = 10

class DetrResnet50(nn.Module):
    """

    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Initialize configuration with a pre-trained ResNet-50 backbone.
        # This initializes the Transformer encoder/decoder and detection heads
        # from scratch, complying with the restriction of only using pretrained
        # weights for the backbone and avoiding external data in the rest of the model.
        config = DeformableDetrConfig(
            backbone="resnet50",
            use_pretrained_backbone=True,
            num_labels=num_classes,
            auxiliary_loss=True,
        )
        self.model = DeformableDetrForObjectDetection(config)

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

def get_model(num_classes=NUM_CLASSES):
    return DetrResnet50(num_classes=num_classes)