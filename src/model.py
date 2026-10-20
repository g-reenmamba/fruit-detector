"""
model.py
--------
The network behind Fruit Detector: an ImageNet-pretrained backbone with a
small fresh/rotten head on top (transfer learning).

What changed vs. the first version, and why:

  * Backbone weights come from timm's GitHub release (a stable, public URL
    that works on a laptop, in CI and on Streamlit Cloud). No more local
    "pretrained cache" fallback to maintain.
  * The head has Dropout in front of the Linear layer. With few fruit types
    to learn from, a little dropout stops the head memorising them.
  * `freeze_backbone` now ALSO puts BatchNorm layers into eval mode
    (see `set_frozen_bn_eval`). requires_grad=False stops the weights from
    changing, but BatchNorm's running mean/variance still update every time
    model.train() runs a batch. In the old model those statistics drifted
    toward the 7 training screenshots -- a silent form of overfitting.
  * get_device() also picks Apple-Silicon GPUs (MPS), so training on a Mac
    is several times faster than on CPU.
"""

import timm
import torch
import torch.nn as nn

# Architecture name in timm -> (pretrained-weights URL, input size).
# All URLs are GitHub release assets of huggingface/pytorch-image-models.
_RELEASE = "https://github.com/huggingface/pytorch-image-models/releases/download/"
BACKBONES = {
    "efficientnet_b0": (_RELEASE + "v0.1-weights/efficientnet_b0_ra-3dd342df.pth", 224),
    "resnet18": (_RELEASE + "v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth", 224),
    "resnet50": (_RELEASE + "v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth", 224),
    "mobilenetv3_large_100": (_RELEASE + "v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth", 224),
    "efficientnet_b3": (_RELEASE + "v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth", 288),
}
DEFAULT_ARCH = "efficientnet_b0"


class FreshnessNet(nn.Module):
    """Backbone (feature extractor) + Dropout + Linear head."""

    def __init__(self, arch=DEFAULT_ARCH, num_classes=2, dropout=0.3, pretrained=True):
        super().__init__()
        if arch not in BACKBONES:
            raise ValueError(f"unknown arch {arch!r}; choose from {list(BACKBONES)}")
        self.arch = arch
        # num_classes=0 -> timm returns pooled features instead of ImageNet logits
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        if pretrained:
            url, _ = BACKBONES[arch]
            state = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
            # strict=False only because the ImageNet classifier weights have
            # nowhere to go (we removed that layer); everything else must load.
            missing, _ = self.backbone.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(f"pretrained weights did not match {arch}: missing {missing[:5]}")
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(num_classes=2, arch=DEFAULT_ARCH, pretrained=True, freeze_backbone=True):
    model = FreshnessNet(arch=arch, num_classes=num_classes, pretrained=pretrained)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model


def unfreeze_top(model, fraction=0.5):
    """
    Unfreeze the top `fraction` of the backbone's parameter tensors (the
    later, more task-specific layers) for the fine-tuning stage. The early
    layers -- edges, textures, colour blobs -- stay frozen: they are already
    general, and keeping them fixed is a strong guard against overfitting.
    """
    params = list(model.backbone.parameters())
    start = int(len(params) * (1 - fraction))
    for p in params[start:]:
        p.requires_grad = True
    return model


def set_frozen_bn_eval(model):
    """
    Call after model.train(): any BatchNorm layer whose weights are frozen
    goes back to eval mode, so its running statistics stay the ImageNet ones
    instead of drifting toward whatever is in the current batch.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()
    return model


def image_size(arch):
    return BACKBONES[arch][1]


def get_device():
    """Use an NVIDIA GPU, else an Apple-Silicon GPU (MPS), else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
