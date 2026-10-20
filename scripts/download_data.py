"""
download_data.py
----------------
Builds the training set for Fruit Detector from public datasets that are
mirrored on GitHub, and lays it out by FRUIT and by LABEL:

    data/
        apple/fresh/*.jpg      apple/rotten/*.jpg
        banana/fresh/*.jpg     banana/rotten/*.jpg
        orange/fresh/*.jpg     orange/rotten/*.jpg
        strawberry/fresh/*.jpg strawberry/rotten/*.jpg

Why organise by fruit? So training can hold out an entire fruit for
validation. That measures what we actually care about -- "can it judge a
fruit it has never seen?" -- instead of "can it recognise more photos of
the apples it already saw?".

Run from the project root (needs git, ~4 GB free disk while cloning):
    python3 scripts/download_data.py
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

# Each source: (repo, pinned commit, sparse path, {source_folder: (fruit, label)})
SOURCES = [
    (
        # Mirror of the Kaggle "Fruits fresh and rotten for classification"
        # dataset (Sriram Reddy Kalluri), ~13.6k images.
        "https://github.com/Bangkit-JKT2-D/fruits-fresh-rotten-classification.git",
        "780bcc08635e30e68b232671e8018fa5f01e975d",
        "Datasets",
        {
            "freshapples": ("apple", "fresh"), "rottenapples": ("apple", "rotten"),
            "freshbanana": ("banana", "fresh"), "rottenbanana": ("banana", "rotten"),
            "freshoranges": ("orange", "fresh"), "rottenoranges": ("orange", "rotten"),
        },
    ),
    (
        # Small web-scraped set with strawberries (plus a few apples/bananas).
        "https://github.com/1RaghavM/Fruit-Freshness-Classification.git",
        "35d424362a9efe8643ec5fff6982e074e9dec65c",
        "modules/data",
        {
            "Strawberry_Fresh": ("strawberry", "fresh"), "Strawberry_Rotten": ("strawberry", "rotten"),
            "Apple_Fresh": ("apple", "fresh"), "Apple_Rotten": ("apple", "rotten"),
            "Banana_Fresh": ("banana", "fresh"), "Banana_Rotten": ("banana", "rotten"),
        },
    ),
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def sparse_clone(url, commit, path, dest):
    run = lambda *args: subprocess.run(args, check=True, cwd=dest)
    os.makedirs(dest, exist_ok=True)
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", url)
    run("git", "sparse-checkout", "set", "--no-cone", path + "/")
    run("git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    run("git", "checkout", "-q", "FETCH_HEAD")


def main(out_dir="data"):
    seen = set()  # md5 of every image copied -> drops exact duplicates across splits/sources
    counts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, (url, commit, path, mapping) in enumerate(SOURCES):
            print(f"cloning {url} ...")
            repo_dir = os.path.join(tmp, f"src{i}")
            sparse_clone(url, commit, path, repo_dir)
            for root, _, files in os.walk(os.path.join(repo_dir, path)):
                folder = os.path.basename(root)
                if folder not in mapping:
                    continue
                fruit, label = mapping[folder]
                dst_dir = os.path.join(out_dir, fruit, label)
                os.makedirs(dst_dir, exist_ok=True)
                for fname in files:
                    if not fname.lower().endswith(IMAGE_EXTS):
                        continue
                    src = os.path.join(root, fname)
                    with open(src, "rb") as fh:
                        digest = hashlib.md5(fh.read()).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    ext = os.path.splitext(fname)[1].lower()
                    shutil.copy(src, os.path.join(dst_dir, f"s{i}_{digest[:12]}{ext}"))
                    counts[(fruit, label)] = counts.get((fruit, label), 0) + 1

    print("\nimages per fruit / label:")
    for (fruit, label), n in sorted(counts.items()):
        print(f"  {fruit:12s} {label:7s} {n:6d}")
    print(f"  total: {sum(counts.values())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
