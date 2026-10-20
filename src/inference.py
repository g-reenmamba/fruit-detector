"""
inference.py
------------
Load the trained checkpoint and judge an image. app.py calls the same
load_model() / predict_image() functions, so the terminal and the web app
can never disagree.

    python3 src/inference.py path/to/photo.jpg
    python3 src/inference.py samples/          # a whole folder; files named
                                               # fresh_*.png / rotten_*.png are scored

What changed vs. the first version, and why:

  * The model is rebuilt with pretrained=False. All weights come from the
    checkpoint, so the app never downloads backbone weights it would
    immediately overwrite (faster cold start on Streamlit Cloud).
  * Logits are divided by the calibrated temperature saved at training
    time, so confidence numbers are honest.
  * Test-time augmentation: the image and its mirror image are both
    scored and averaged. Cheap, and steadier on odd photos.
  * verdict() turns probabilities into fresh / rotten / unsure. Below
    UNSURE_BELOW confidence the app says it isn't sure rather than guessing.
"""

import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import eval_transform
from model import build_model, get_device

UNSURE_BELOW = 0.70  # confidence under this -> "not sure"
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "checkpoints", "best_model.pth")


def load_model(checkpoint_path=DEFAULT_CHECKPOINT):
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = build_model(len(classes), arch=ckpt["arch"], pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.temperature = ckpt.get("temperature", 1.0)
    model.transform = eval_transform(ckpt.get("img_size", 224))
    model.to(device).eval()
    return model, classes, device


@torch.no_grad()
def predict_image(image, model, classes, device):
    """PIL image -> (predicted class, confidence, {class: probability})."""
    x = model.transform(image.convert("RGB")).unsqueeze(0).to(device)
    x = torch.cat([x, torch.flip(x, dims=[3])])            # original + mirror
    logits = model(x).mean(0) / model.temperature            # average, then calibrate
    probs = F.softmax(logits, dim=0).cpu()
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx]), dict(zip(classes, probs.tolist()))


def verdict(pred_class, confidence):
    return pred_class if confidence >= UNSURE_BELOW else "unsure"


def _main(target, checkpoint_path=DEFAULT_CHECKPOINT):
    model, classes, device = load_model(checkpoint_path)
    paths = ([os.path.join(target, f) for f in sorted(os.listdir(target))
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
             if os.path.isdir(target) else [target])
    scored = correct = 0
    for path in paths:
        pred, conf, probs = predict_image(Image.open(path), model, classes, device)
        name = os.path.basename(path)
        truth = name.split("_")[0] if name.split("_")[0] in classes else None
        mark = "" if truth is None else ("  OK" if pred == truth else "  WRONG")
        print(f"{name:32s} -> {verdict(pred, conf):7s} ({conf * 100:5.1f}% {pred}){mark}")
        if truth:
            scored += 1
            correct += pred == truth
    if scored:
        print(f"\n{correct}/{scored} correct")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 src/inference.py <image-or-folder>")
        sys.exit(1)
    _main(sys.argv[1])
