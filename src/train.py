"""
train.py
--------
Trains the Fruit Detector freshness model.

Run from the project root:

    # 1) Honest generalisation check: train on all fruits except one,
    #    validate on the one it never saw. Repeat for every fruit.
    python3 src/train.py --cv

    # 2) Single held-out fruit
    python3 src/train.py --holdout banana

    # 3) Final model for the app: train on every fruit
    python3 src/train.py

What changed vs. the first version, and why:

  * Two-stage transfer learning. Stage 1 trains only the new head on a
    frozen backbone; stage 2 unfreezes the top half of the backbone at a
    10x smaller learning rate. The old script only ever did stage 1.
  * Label smoothing (0.1) + weight decay. Stops the model from becoming
    100% sure about the training fruits, which is what made its verdicts
    on new fruits so overconfident.
  * Checkpoint chosen by lowest validation LOSS, with early stopping. The
    old `if val_acc >= best_val_acc` on a 2-image val set just saved the
    last epoch every time (2/2 correct = 100% again and again).
  * Balanced sampling and fruit-held-out validation (see dataset.py).
  * Temperature scaling after training, so "80% confident" means roughly
    80% -- which lets the app say "not sure" instead of guessing.
"""

import argparse
import copy
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from dataset import (CLASSES, FruitDataset, balanced_sampler, describe, eval_transform,
                     scan, split_by_fruit, split_random, train_transform)
from model import (DEFAULT_ARCH, build_model, get_device, image_size, set_frozen_bn_eval,
                   unfreeze_top)


def run_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    set_frozen_bn_eval(model)  # frozen BatchNorm layers keep their ImageNet statistics
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, samples, size, device, batch_size=64, num_workers=2):
    """Returns plain-CE loss, accuracy, balanced accuracy, per-fruit accuracy, raw logits."""
    model.eval()
    loader = DataLoader(FruitDataset(samples, eval_transform(size)), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    logits = torch.cat([model(x.to(device)).cpu() for x, _ in loader])
    labels = torch.tensor([s[1] for s in samples])
    preds = logits.argmax(1)
    loss = F.cross_entropy(logits, labels).item()
    per_class = [(preds[labels == c] == c).float().mean().item() for c in range(len(CLASSES))
                 if (labels == c).any()]
    per_fruit = {}
    fruits = [s[2] for s in samples]
    for fruit in sorted(set(fruits)):
        idx = torch.tensor([f == fruit for f in fruits])
        per_fruit[fruit] = (preds[idx] == labels[idx]).float().mean().item()
    return dict(loss=loss, acc=(preds == labels).float().mean().item(),
                bal_acc=sum(per_class) / len(per_class), per_fruit=per_fruit,
                logits=logits, labels=labels)


def cap_per_group(samples, cap, seed=0):
    rng = random.Random(seed)
    groups = {}
    for s in samples:
        groups.setdefault((s[2], s[1]), []).append(s)
    out = []
    for key in sorted(groups):
        items = groups[key][:]
        rng.shuffle(items)
        out += items[:cap]
    return out


def fit_temperature(logits, labels):
    """Find T > 0 minimising NLL of softmax(logits / T) on held-out data."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.2, 5.0))


def train(data_dir="data", holdout=None, arch=DEFAULT_ARCH, head_epochs=3, finetune_epochs=6,
          samples_per_epoch=3000, batch_size=32, lr=1e-3, patience=3, num_workers=2,
          checkpoint_path="checkpoints/best_model.pth", seed=0, val_cap=None, temperature=None):
    torch.manual_seed(seed)
    device = get_device()
    size = image_size(arch)
    samples = scan(data_dir)
    if holdout:
        train_s, val_s = split_by_fruit(samples, holdout)
    else:
        train_s, val_s = split_random(samples, val_fraction=0.1, seed=seed)
    if val_cap:  # optional: evaluate on at most val_cap images per (fruit, label) to save time
        val_s = cap_per_group(val_s, val_cap, seed)
    describe(train_s, "train")
    describe(val_s, "val  ")
    print(f"device={device} arch={arch} size={size}")

    train_loader = DataLoader(FruitDataset(train_s, train_transform(size)), batch_size=batch_size,
                              sampler=balanced_sampler(train_s, samples_per_epoch),
                              num_workers=num_workers, drop_last=True)

    model = build_model(len(CLASSES), arch=arch, pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = dict(loss=float("inf"))
    best_state, bad_epochs, epoch = None, 0, 0
    stages = [("head", head_epochs, lr), ("finetune", finetune_epochs, lr / 10)]
    for stage, n_epochs, stage_lr in stages:
        if n_epochs == 0:
            continue
        if stage == "finetune":
            unfreeze_top(model, fraction=0.5)
            bad_epochs = 0
        head_params = list(model.head.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": stage_lr})
        optimizer = AdamW(groups, weight_decay=0.05)
        scheduler = OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups],
                               total_steps=n_epochs * len(train_loader), pct_start=0.2)
        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            ev = evaluate(model, val_s, size, device, num_workers=num_workers)
            fruits = " ".join(f"{k}={v:.2f}" for k, v in ev["per_fruit"].items())
            print(f"[{stage:8s}] epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {ev['loss']:.3f} acc {ev['acc']:.3f} bal {ev['bal_acc']:.3f} | "
                  f"{fruits} | {time.time() - t0:.0f}s", flush=True)
            if ev["loss"] < best["loss"]:
                best, best_state, bad_epochs = ev, copy.deepcopy(model.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stop: val loss hasn't improved for {patience} epochs")
                    break

    fitted = fit_temperature(best["logits"], best["labels"])
    # A temperature fitted on fruits the model trained on is over-confident for
    # new fruits. Pass --temperature with the mean value printed by --cv to use
    # the one measured on unseen fruits instead.
    temperature = temperature if temperature is not None else fitted
    best["temperature"] = fitted
    print(f"best val: loss {best['loss']:.3f} acc {best['acc']:.3f} "
          f"bal_acc {best['bal_acc']:.3f} | fitted temperature {fitted:.2f} (using {temperature:.2f})")

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "classes": CLASSES,
            "arch": arch,
            "img_size": size,
            "temperature": temperature,
            "trained_on": sorted({s[2] for s in train_s}),
            "holdout": holdout or [],
            "val_acc": best["acc"],
            "val_bal_acc": best["bal_acc"],
        }, checkpoint_path)
        print(f"saved -> {checkpoint_path}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--arch", default=DEFAULT_ARCH)
    ap.add_argument("--holdout", nargs="*", help="fruit(s) to keep out of training and validate on")
    ap.add_argument("--cv", action="store_true", help="leave-one-fruit-out over every fruit")
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=6)
    ap.add_argument("--samples-per-epoch", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-cap", type=int, default=None,
                    help="evaluate on at most N images per (fruit, label) -- faster on CPU")
    ap.add_argument("--temperature", type=float, default=None,
                    help="calibration temperature to save (default: fit on the validation set)")
    ap.add_argument("--out", default="checkpoints/best_model.pth")
    args = ap.parse_args()

    common = dict(data_dir=args.data, arch=args.arch, head_epochs=args.head_epochs,
                  finetune_epochs=args.finetune_epochs, samples_per_epoch=args.samples_per_epoch,
                  batch_size=args.batch_size, num_workers=args.workers, val_cap=args.val_cap)
    if args.cv:
        fruits = sorted({s[2] for s in scan(args.data)})
        results = {}
        for fruit in fruits:
            print(f"\n===== holding out: {fruit} =====")
            results[fruit] = train(holdout=[fruit], checkpoint_path=None, **common)
        print("\nleave-one-fruit-out (model never saw this fruit during training):")
        for fruit, r in results.items():
            print(f"  {fruit:12s} accuracy {r['acc']:.3f}   balanced {r['bal_acc']:.3f}")
        mean = sum(r["bal_acc"] for r in results.values()) / len(results)
        print(f"  {'mean':12s} balanced {mean:.3f}")
        t = sum(r["temperature"] for r in results.values()) / len(results)
        print(f"\nmean temperature on unseen fruits: {t:.2f}  "
              f"-> train the final model with: python3 src/train.py --temperature {t:.2f}")
    else:
        train(holdout=args.holdout, checkpoint_path=args.out, temperature=args.temperature, **common)


if __name__ == "__main__":
    main()
