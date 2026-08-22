"""
dataset.py
----------
PyTorch Dataset / DataLoader / transforms for VisionGuard.

Expects images organized as:

    data/
        train/
            <class_a>/
                img1.jpg
                ...
            <class_b>/
                ...
        val/
            <class_a>/...
            <class_b>/...
        test/
            <class_a>/...
            <class_b>/...

Class names are read directly from the subfolder names, so this works
for any set of classes (fresh/spoiled, crack/scratch/dent, etc.) without
code changes.
"""

import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Stats ResNet18 was originally trained with on ImageNet. Since we're
# using ImageNet-pretrained weights, we normalize our images the same
# way the pretrained backbone expects to see them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transform: resize to what ResNet18 expects, then apply light
# random augmentation so a small dataset behaves like a slightly bigger,
# more varied one. Augmentation is randomized per epoch, so the model
# rarely sees the exact same image twice.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Validation/test transform: resize + normalize only, no randomness. Every
# evaluation run then measures the model against the same fixed images,
# so results are comparable and reproducible run to run.
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ImageClassificationDataset(Dataset):
    """
    A from-scratch PyTorch Dataset. Subclassing torch.utils.data.Dataset
    requires exactly two methods:

      __len__     -> how many samples exist
      __getitem__ -> given an index, return (image_tensor, label)

    DataLoader (see get_dataloaders below) wraps this to handle batching,
    shuffling, and (optionally) parallel loading.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []  # list of (filepath, label) built once at init
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(data_dir="data", batch_size=8, num_workers=0):
    """Build train/val/test DataLoaders from data_dir/{train,val,test}."""
    train_ds = ImageClassificationDataset(os.path.join(data_dir, "train"), transform=train_transform)
    val_ds = ImageClassificationDataset(os.path.join(data_dir, "val"), transform=eval_transform)
    test_ds = ImageClassificationDataset(os.path.join(data_dir, "test"), transform=eval_transform)

    assert train_ds.classes == val_ds.classes == test_ds.classes, \
        "train/val/test must have the same set of class subfolders"

    # shuffle=True for training: the model shouldn't learn anything from
    # the order images happen to be in. shuffle=False for val/test: order
    # doesn't matter for evaluation, and keeping it fixed makes runs
    # easier to compare against each other.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes