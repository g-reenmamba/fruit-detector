"""
inference.py
-------------
Command-line inference: given an image path, load the trained checkpoint
and print the predicted class + confidence. This is the same logic the
Streamlit app will call later -- proving it works from the terminal first.

Run from the project root:
    python3 src/inference.py path/to/some_photo.jpg
"""

import sys

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import eval_transform
from model import build_model, get_device


def load_model(checkpoint_path="checkpoints/best_model.pth"):
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]

    model = build_model(num_classes=len(classes), freeze_backbone=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, device


def predict_image(image, model, classes, device):
    """
    Same as predict() below, but takes an already-loaded PIL Image instead
    of a file path. Used by app.py so an uploaded image can go straight
    from memory into the model, with no temp file written to disk.
    """
    tensor = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)  # add a batch dimension of 1

    with torch.no_grad():
        logits = model(tensor)
        # CrossEntropyLoss (used during training) works directly on raw logits
        # and applies its own internal log-softmax. For a human-readable
        # "93.4% confident" number here, we apply softmax explicitly ourselves.
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = classes[pred_idx]
    confidence = probs[pred_idx].item()
    all_probs = dict(zip(classes, probs.tolist()))

    return pred_class, confidence, all_probs


def predict(image_path, model=None, classes=None, device=None, checkpoint_path="checkpoints/best_model.pth"):
    if model is None:
        model, classes, device = load_model(checkpoint_path)

    image = Image.open(image_path).convert("RGB")
    return predict_image(image, model, classes, device)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 src/inference.py <path-to-image>")
        sys.exit(1)

    pred_class, confidence, all_probs = predict(sys.argv[1])
    print(f"prediction: {pred_class}  (confidence: {confidence * 100:.1f}%)")
    print("all class probabilities:")
    for cls, p in all_probs.items():
        print(f"  {cls:12s} {p * 100:5.1f}%")