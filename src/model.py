"""U-Net model for multi-organ abdominal MRI segmentation.

Architecture: U-Net with pretrained ResNet34 encoder (ImageNet weights).
Output: 5 channels (background + liver + right kidney + left kidney + spleen).
No final activation — raw logits fed directly to CrossEntropyLoss.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def create_model(num_classes: int = 5) -> nn.Module:
    """Create a U-Net with pretrained ResNet34 encoder.

    Args:
        num_classes: number of output channels (default 5: background + 4 organs)

    Returns:
        nn.Module ready for multi-class segmentation
    """
    model = smp.Unet(
        encoder_name    = "resnet50",
        encoder_weights = "imagenet",
        in_channels     = 3,          # greyscale repeated 3× to match ImageNet input
        classes         = num_classes,
        activation      = None,       # raw logits — CrossEntropyLoss applies softmax
    )
    return model


def get_parameter_groups(model: nn.Module) -> tuple[list, list]:
    """Split parameters into encoder and decoder groups for differential LRs.

    Encoder (pretrained ResNet34) gets a lower LR to preserve ImageNet features.
    Decoder (randomly initialised) gets a higher LR to learn fast.
    """
    encoder_params = list(model.encoder.parameters())
    decoder_params = (
        list(model.decoder.parameters()) +
        list(model.segmentation_head.parameters())
    )
    return encoder_params, decoder_params


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
