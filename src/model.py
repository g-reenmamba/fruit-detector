"""
model.py
--------
ResNet18 with transfer learning for VisionGuard.

Instead of training a CNN from scratch (which needs a lot of data and
time), we start from a ResNet18 already trained on ImageNet (1.2M images,
1000 classes) and adapt it to our much smaller classification problem.
"""

import os

import torch
import torch.nn as nn
from torchvision import models

# Local fallback cache for ImageNet-pretrained ResNet18 weights. Torchvision
# normally downloads these on demand from download.pytorch.org; some
# networks (including this project's sandbox) block that host specifically
# while still allowing github.com/pypi.org. scripts/fetch_pretrained_backbone.py
# populates this file from a verified GitHub mirror. Not committed to git
# (see .gitignore) since it's a large, easily re-obtainable binary.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_BACKBONE_CACHE = os.path.join(_PROJECT_ROOT, "checkpoints", "pretrained", "resnet18_imagenet.pth")


def _load_pretrained_resnet18():
    """
    Return a torchvision ResNet18 with genuine ImageNet-pretrained weights.

    Tries torchvision's normal online download first (works in most
    environments, and keeps this code portable). Falls back to a local
    cached copy of the same architecture's weights if that download fails
    -- see scripts/fetch_pretrained_backbone.py for how the cache is built
    and verified.
    """
    try:
        return models.resnet18(weights="DEFAULT")
    except Exception as download_error:
        if not os.path.exists(_LOCAL_BACKBONE_CACHE):
            raise RuntimeError(
                "Could not download pretrained ResNet18 weights, and no "
                f"local fallback cache exists at {_LOCAL_BACKBONE_CACHE}. "
                "Run scripts/fetch_pretrained_backbone.py, or connect to a "
                "network that can reach download.pytorch.org."
            ) from download_error

        model = models.resnet18(weights=None)  # architecture only, random init
        state_dict = torch.load(_LOCAL_BACKBONE_CACHE, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model


def build_model(num_classes, freeze_backbone=True):
    """
    Load a pretrained ResNet18 and adapt it to `num_classes` outputs.

    freeze_backbone=True freezes every existing ResNet18 weight, so the
    only thing that trains initially is the brand-new final layer. This
    is fast and hard to overfit with a small dataset. Once results look
    reasonable, layers can be selectively unfrozen (see unfreeze_layer4
    below) to fine-tune the backbone's higher-level features to this
    specific task.
    """
    model = _load_pretrained_resnet18()

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # model.fc is ResNet18's final linear layer, originally
    # in_features=512 -> out_features=1000 (ImageNet's 1000 classes).
    # Replacing it creates a brand-new layer with requires_grad=True by
    # default, sized for our actual number of classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def unfreeze_layer4(model):
    """Unfreeze the last ResNet block, for a fine-tuning pass after initial training."""
    for param in model.layer4.parameters():
        param.requires_grad = True
    return model


def get_device():
    """Device-agnostic helper: use a GPU if one is available, else fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")