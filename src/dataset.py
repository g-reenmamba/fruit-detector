"""
dataset.py
----------
Data loading for Fruit Detector.

Expected layout (scripts/download_data.py builds it):

    data/
        <fruit>/fresh/*.jpg
        <fruit>/rotten/*.jpg
        ...

Three ideas in here fight the "only works on fruits it has seen" problem:

1. Validation by FRUIT, not by photo (`split_by_fruit`). If every fruit
   shows up in both train and val, a model can score 99% just by learning
   "this is a banana, and bananas in this dataset are mostly rotten". By
   holding out entire fruits, the validation score answers the real
   question: can it judge a fruit it has never seen?

2. Balanced sampling (`balanced_sampler`). Each (fruit, label) group is
   drawn equally often, so fruit identity carries no information about the
   label and the only way to lower the loss is to learn what rot looks
   like: mould, dark soft patches, wrinkles, collapsed skin.

3. Stronger augmentation. Random crops/zoom, flips, rotation, blur and
   lighting changes stop the model keying on framing, background or
   camera. Hue is only nudged slightly: colour shifts (green->brown,
   red->black) are a genuine freshness signal and we don't want to erase it.
"""

import os
import random
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ["fresh", "rotten"]  # fixed order; index 0 = fresh, 1 = rotten
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def train_transform(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.RandomApply([transforms.ColorJitter(0.35, 0.35, 0.3, 0.03)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_transform(size=224):
    # Whole image squeezed to size x size (no centre crop) so a fruit near
    # the edge of the photo is never cut out. Matches the old behaviour.
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def scan(data_dir):
    """Return [(path, label_idx, fruit), ...] for every image under data_dir."""
    samples = []
    for fruit in sorted(os.listdir(data_dir)):
        fruit_dir = os.path.join(data_dir, fruit)
        if not os.path.isdir(fruit_dir):
            continue
        for label_idx, label in enumerate(CLASSES):
            label_dir = os.path.join(fruit_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                if fname.lower().endswith(IMAGE_EXTS):
                    samples.append((os.path.join(label_dir, fname), label_idx, fruit))
    if not samples:
        raise FileNotFoundError(f"no images found under {data_dir}/<fruit>/(fresh|rotten)/")
    return samples


def split_by_fruit(samples, holdout_fruits):
    """Everything from `holdout_fruits` goes to validation, the rest to training."""
    holdout = set(holdout_fruits)
    train = [s for s in samples if s[2] not in holdout]
    val = [s for s in samples if s[2] in holdout]
    if not val:
        raise ValueError(f"holdout fruits {sorted(holdout)} not found in the data")
    return train, val


def split_random(samples, val_fraction=0.1, seed=0):
    """Stratified random split (same fraction from every fruit/label group)."""
    rng = random.Random(seed)
    groups = {}
    for s in samples:
        groups.setdefault((s[2], s[1]), []).append(s)
    train, val = [], []
    for key in sorted(groups):
        items = groups[key][:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def balanced_sampler(samples, num_samples):
    """Draw every (fruit, label) group equally often, with replacement."""
    counts = Counter((s[2], s[1]) for s in samples)
    weights = [1.0 / counts[(s[2], s[1])] for s in samples]
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


class FruitDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def describe(samples, name):
    counts = Counter((s[2], CLASSES[s[1]]) for s in samples)
    fruits = sorted({s[2] for s in samples})
    parts = [f"{f} {counts[(f, 'fresh')]}/{counts[(f, 'rotten')]}" for f in fruits]
    print(f"{name}: {len(samples)} images (fresh/rotten) -> " + ", ".join(parts))
