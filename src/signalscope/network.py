"""One model and preprocessing definition shared by training, CLI and API."""

import torch
from torch import nn
from torch.nn import functional as F


def build_model(pretrained: bool = False) -> nn.Module:
    # Legacy ResNet releases need torchvision; the CLIP runtime does not.
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def preprocess_batch(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """uint8 BCHW RGB -> float ImageNet-normalized BCHW, with fixed antialiasing."""
    images = images.float().div(255)
    images = F.interpolate(images, size=(image_size, image_size), mode="bilinear",
                           align_corners=False, antialias=True)
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    return (images - mean) / std
