"""U-Net model for multi-organ abdominal MRI segmentation.

Architecture: U-Net with pretrained EfficientNet-B4 encoder (ImageNet weights).
EfficientNet-B4 uses compound scaling (depth + width + resolution) and squeeze-
excitation blocks, giving better feature extraction for small structures like the
spleen compared to ResNet's plain residual blocks.

Output: 5 channels (background + liver + right kidney + left kidney + spleen).
No final activation — raw logits fed directly to CrossEntropyLoss.
Input: 3 channels — consecutive MRI slices [n-1, n, n+1] (2.5D).
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def create_model(num_classes: int = 5) -> nn.Module:
    """Create a U-Net with pretrained EfficientNet-B4 encoder.

    Args:
        num_classes: number of output channels (default 5: background + 4 organs)

    Returns:
        nn.Module ready for multi-class segmentation
    """
    model = smp.Unet(
        encoder_name    = "efficientnet-b4",
        encoder_weights = "imagenet",
        in_channels     = 3,          # 2.5D: slices [n-1, n, n+1] as 3 channels
        classes         = num_classes,
        activation      = None,       # raw logits — CrossEntropyLoss applies softmax
    )
    return model


def get_parameter_groups(model: nn.Module) -> tuple[list, list]:
    """Split parameters into encoder and decoder groups for differential LRs.

    Encoder (pretrained EfficientNet-B4) gets a lower LR to preserve ImageNet features.
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
