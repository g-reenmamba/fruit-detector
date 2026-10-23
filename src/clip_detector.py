"""
clip_detector.py
----------------
The model behind the app: CLIP's image encoder plus pre-encoded text prompts.

For ~60 kinds of produce we compare the photo with "a photo of a fresh X" and
"a photo of a rotten X" (several phrasings each). A softmax over all of those
prompts gives, at the same time, which produce it most likely is and how much
of the vote went to a "rotten ..." prompt. A handful of "not produce" prompts
(people, pets, screenshots...) let the app say when a photo isn't a fruit or
vegetable at all.

Why this instead of the fine-tuned CNN in model.py? See scripts/export_clip.py
and the README: the CNN only knew four fruits; CLIP generalises to almost any.
"""

import io
import os

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

ARCH = "ViT-B-32-quickgelu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

PRODUCE = [
    "bell pepper", "chili pepper", "dragon fruit", "passion fruit", "sweet potato",
    "apple", "apricot", "avocado", "banana", "beetroot", "blackberry", "blueberry", "broccoli",
    "cabbage", "cantaloupe", "carrot", "cauliflower", "celery", "cherry", "coconut", "corn",
    "cucumber", "date", "eggplant", "fig", "garlic", "ginger", "grape", "grapefruit", "guava",
    "kiwi", "lemon", "lettuce", "lime", "lychee", "mango", "melon", "mushroom", "nectarine",
    "okra", "onion", "orange", "papaya", "peach", "pear", "peas", "pepper", "persimmon",
    "pineapple", "plum", "pomegranate", "potato", "pumpkin", "radish", "raspberry", "spinach",
    "squash", "strawberry", "tamarillo", "tangerine", "tomato", "turnip", "watermelon", "zucchini",
]
FRESH_TEMPLATES = ["a photo of a fresh {}.", "a photo of fresh {}.", "a close-up photo of a fresh, healthy {}."]
ROTTEN_TEMPLATES = ["a photo of a rotten {}.", "a photo of a moldy, rotting {}.",
                    "a close-up photo of a spoiled, decaying {}."]
NOT_PRODUCE = [
    "a photo of a person.", "a selfie.", "a photo of a dog.", "a photo of a cat.", "a photo of an animal.",
    "a photo of a car.", "a photo of a building.", "a screenshot of a computer screen.",
    "a photo of a document with text.", "a cartoon drawing.", "a photo of furniture in a room.",
    "a photo of a phone or laptop.",
]
NOT_PRODUCE_ABOVE = 0.80   # only refuse when CLIP is quite sure it isn't produce
UNSURE_BELOW = 0.70        # fresh/rotten confidence under this -> "not sure"

# CLIP's own preprocessing: shortest side to 224 (bicubic), centre crop, CLIP colour stats
PREPROCESS = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def _vision_tower():
    import open_clip
    from open_clip.model import _build_vision_tower
    cfg = open_clip.get_model_config(ARCH)
    return _build_vision_tower(cfg["embed_dim"], cfg["vision_cfg"], quick_gelu=cfg.get("quick_gelu", False))


def load_detector(checkpoint_dir=CHECKPOINT_DIR):
    visual = _vision_tower()
    state = {}
    for part in ("part1", "part2"):
        state.update(load_file(os.path.join(checkpoint_dir, f"clip_visual.{part}.safetensors")))
    visual.load_state_dict({k: v.float() for k, v in state.items()})
    visual.eval()
    prompts = torch.load(os.path.join(checkpoint_dir, "clip_prompts.pt"), map_location="cpu")
    return {"visual": visual, **prompts}


@torch.no_grad()
def to_srgb(image):
    """
    RGB image in standard sRGB colours. Photos from Macs and iPhones often embed
    a wider colour profile (Display P3); browsers convert those to sRGB before
    showing them, so we do the same. Otherwise a saturated red tomato is
    literally different pixels here than in the browser version.
    """
    icc = image.info.get("icc_profile")
    if icc:
        try:
            from PIL import ImageCms
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            return ImageCms.profileToProfile(image.convert("RGB"), src, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            pass  # unreadable profile: fall back to the raw numbers
    return image.convert("RGB")


@torch.no_grad()
def analyse(image, det):
    """
    PIL image -> dict(
        rotten=P(rotten), fresh=P(fresh)      over produce prompts only
        produce=best-matching produce name,   produce_conf=its share of the vote
        not_produce=P(photo isn't produce at all))

    (Tried and dropped: also scoring zoomed-in crops to catch small rot spots.
    It helped on the Kaggle 4-fruit photos but not on real-world ones, was
    fragile across browsers, and tripled the time per check in the web app.)
    """
    img = to_srgb(image)
    x = torch.stack([PREPROCESS(img), PREPROCESS(img.transpose(Image.FLIP_LEFT_RIGHT))])
    feats = F.normalize(det["visual"](x), dim=-1)
    f = F.normalize(feats.sum(0), dim=0)                    # photo + its mirror image
    scale = det["logit_scale"]
    n = len(det["produce"])
    produce_logits = scale * torch.cat([det["fresh"], det["rotten"]]) @ f        # [2n]
    p = produce_logits.softmax(0)
    rotten = float(p[n:].sum())
    per_produce = p[:n] + p[n:]
    best = int(per_produce.argmax())
    other_logits = scale * det["not_produce"] @ f
    not_produce = float(torch.cat([produce_logits, other_logits]).softmax(0)[2 * n:].sum())
    return {"rotten": rotten, "fresh": 1 - rotten, "produce": det["produce"][best],
            "produce_conf": float(per_produce[best]), "not_produce": not_produce}


if __name__ == "__main__":
    import sys
    det = load_detector()
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "samples")
    paths = ([os.path.join(target, f) for f in sorted(os.listdir(target))
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))] if os.path.isdir(target) else [target])
    ok = scored = 0
    for path in paths:
        r = analyse(Image.open(path), det)
        name = os.path.basename(path)
        label = "rotten" if r["rotten"] > 0.5 else "fresh"
        truth = name.split("_")[0] if name.split("_")[0] in ("fresh", "rotten") else None
        if truth:
            scored += 1; ok += label == truth
        flag = "" if truth is None else ("  OK" if label == truth else "  WRONG")
        print(f"{name:34s} {label:6s} {max(r['rotten'], r['fresh']) * 100:5.1f}%  looks like {r['produce']:12s}"
              f" not-produce {r['not_produce'] * 100:4.1f}%{flag}")
    if scored:
        print(f"\n{ok}/{scored} correct")
