"""
evaluate.py
-----------
Reproduces the test numbers in the README. Downloads three small sets of real
photos (none used to build or tune the model) and scores the app's detector:

  produce   34 photos of fresh fruit & vegetables, many cut or unusual-looking
            (junaiddbz/agrivision-produce-classifier sample images)
  mixed     fresh AND rotten photos, mostly produce types other than the usual
            apple/banana/orange (dhieu0901 demo images + Jayasree478 examples)
  samples   the photos bundled with the app (samples/)

    python3 scripts/evaluate.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import clip_detector  # noqa: E402

SETS = [
    ("produce", "https://github.com/junaiddbz/agrivision-produce-classifier.git",
     "72e169cf9deb7bdbe3fff80052fd06b638dcdb33", "assets/samples"),
    ("mixed", "https://github.com/dhieu0901/Fresh-and-Rotten-Fruit-Classification.git",
     "36b34b644c685683a086d8206a6d7b67531e2020", "demo/demo_images"),
    ("mixed", "https://github.com/Jayasree478/FoodSafe-AI.git",
     "255d3fe483108667b4417f2ab0469dae67a4f5ea", "."),
]
SKIP = ("paprika", "soy beans")  # a spice powder and dried beans aren't fresh produce


def label_for(set_name, fname):
    """Ground truth from the source's own file names."""
    n = fname.lower()
    if set_name == "produce":
        return "fresh"
    if "spoiled" in n or n.startswith("r "):
        return "rotten"
    if "fresh" in n or n.startswith("f "):
        return "fresh"
    return None


def fetch(url, commit, path, dest):
    run = lambda *a: subprocess.run(a, check=True, cwd=dest, capture_output=True)
    os.makedirs(dest)
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", url)
    if path != ".":
        run("git", "sparse-checkout", "set", "--no-cone", path + "/")
    run("git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    run("git", "checkout", "-q", "FETCH_HEAD")
    return os.path.join(dest, path)


def main():
    det = clip_detector.load_detector()
    cases = []  # (set, file name, truth, path)
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, url, commit, path) in enumerate(SETS):
            folder = fetch(url, commit, path, os.path.join(tmp, str(i)))
            for f in sorted(os.listdir(folder)):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")) or f.lower().startswith(SKIP):
                    continue
                truth = label_for(name, f)
                if truth:
                    cases.append((name, f, truth, os.path.join(folder, f)))
        for f in sorted(os.listdir(os.path.join(ROOT, "samples"))):
            cases.append(("samples", f, f.split("_")[0], os.path.join(ROOT, "samples", f)))

        totals = {}
        for set_name, fname, truth, path in cases:
            r = clip_detector.analyse(Image.open(path), det)
            pred = "rotten" if r["rotten"] > 0.5 else "fresh"
            conf = max(r["rotten"], r["fresh"])
            ok = pred == truth
            t = totals.setdefault(set_name, [0, 0, 0])
            t[0] += ok; t[1] += 1; t[2] += conf < clip_detector.UNSURE_BELOW
            if not ok:
                print(f"  miss  [{set_name}] {fname}: truth {truth}, said {pred} ({conf * 100:.0f}%)")
    print()
    for set_name, (ok, n, unsure) in totals.items():
        print(f"{set_name:8s} {ok:3d}/{n:<3d} correct   ({unsure} would show 'not sure')")
    ok = sum(t[0] for t in totals.values()); n = sum(t[1] for t in totals.values())
    print(f"{'total':8s} {ok:3d}/{n:<3d} correct")


if __name__ == "__main__":
    main()
