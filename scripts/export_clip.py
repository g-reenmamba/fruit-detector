"""
export_clip.py
--------------
One-time step that turns a public CLIP checkpoint into the small, self-
contained files the app loads (no internet needed at app start):

    checkpoints/clip_visual.part1.safetensors   } CLIP's image encoder, fp16,
    checkpoints/clip_visual.part2.safetensors   } split to stay < 100 MB per file
    checkpoints/clip_prompts.pt                 text side, already encoded

Why CLIP? The fine-tuned EfficientNet (src/train.py) only ever saw four fruits,
so it learned "brown / dark / irregular = rotten" and called fresh potatoes,
beetroot and cut watermelon rotten (21/36 on real fresh-produce photos). CLIP was
trained on hundreds of millions of image-caption pairs, so it already knows what
fresh and rotten look like for almost any produce. We ask it, for ~60 kinds of
produce, "a photo of a fresh X" vs "a photo of a rotten X" and add up the votes.

Model: OpenCLIP ViT-B/32 (quickgelu) trained on LAION-400M, MIT licence.
Run from the project root:
    python3 scripts/export_clip.py
"""

import os
import sys

import open_clip
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clip_detector import ARCH, CHECKPOINT_DIR, NOT_PRODUCE, PRODUCE, FRESH_TEMPLATES, ROTTEN_TEMPLATES  # noqa: E402

WEIGHTS_URL = ("https://github.com/mlfoundations/open_clip/releases/download/"
               "v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt")


def main(local_weights=None):
    model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=None)
    if local_weights:
        state = torch.load(local_weights, map_location="cpu", weights_only=False)
    else:
        state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", progress=True,
                                                   weights_only=False)
    state = state.get("state_dict", state)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    tok = open_clip.get_tokenizer(ARCH)

    def encode(prompts):
        with torch.no_grad():
            e = torch.nn.functional.normalize(model.encode_text(tok(prompts)), dim=-1).mean(0)
        return torch.nn.functional.normalize(e, dim=0)

    fresh = torch.stack([encode([t.format(n) for t in FRESH_TEMPLATES]) for n in PRODUCE])
    rotten = torch.stack([encode([t.format(n) for t in ROTTEN_TEMPLATES]) for n in PRODUCE])
    other = torch.stack([encode([p]) for p in NOT_PRODUCE])

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({"produce": PRODUCE, "fresh": fresh, "rotten": rotten, "not_produce": other,
                "not_produce_prompts": NOT_PRODUCE, "logit_scale": float(model.logit_scale.exp())},
               os.path.join(CHECKPOINT_DIR, "clip_prompts.pt"))

    visual = {k: v.half().contiguous() for k, v in model.visual.state_dict().items()}
    keys = list(visual)
    half, total, part1 = sum(v.numel() for v in visual.values()) / 2, 0, {}
    for k in keys:
        if total >= half:
            break
        part1[k] = visual[k]
        total += visual[k].numel()
    part2 = {k: visual[k] for k in keys if k not in part1}
    for name, part in (("part1", part1), ("part2", part2)):
        out = os.path.join(CHECKPOINT_DIR, f"clip_visual.{name}.safetensors")
        save_file(part, out)
        print(f"saved {out}  {os.path.getsize(out) / 1e6:.1f} MB")
    print(f"saved {CHECKPOINT_DIR}/clip_prompts.pt  ({len(PRODUCE)} produce types)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
