# Training log for checkpoints/best_model.pth

Trained on CPU (2 cores), EfficientNet-B0 (timm, RA weights), 224 px.

Leave-one-fruit-out:
    python3 src/train.py --cv --head-epochs 3 --finetune-epochs 4 --samples-per-epoch 2000 --val-cap 400
      apple 0.888 | banana 0.939 | orange 0.896 | strawberry 0.908 | mean 0.908

Final model (all four fruits, stratified 10% val):
    python3 src/train.py --head-epochs 3 --finetune-epochs 3 --samples-per-epoch 2500 --val-cap 150 --temperature 0.55
      val acc 0.991 (seen fruits), fitted T on seen fruits 0.32 -> saved T 0.55 (measured on unseen fruits)

    python3 src/inference.py samples/   -> 11/11 correct

## v3: CLIP zero-shot (what the app uses now)

The deployed EfficientNet called a fresh cut watermelon rotten. Testing it on real
photos it had never seen showed the problem was general: 21/34 fresh produce photos
(potatoes, beetroot, ginger, eggplant...) came out "rotten". With only four fruits as
fresh examples, it learned "brown / dark / irregular = rotten".

Tried and rejected:
- Adding ~3k fresh Fruits-360 images of 100+ other produce types: no gain on real
  photos (22/36) and rot detection dropped. Those are 100x100 studio shots on white,
  so the model learned "studio photo = fresh", a new shortcut.
- A linear probe on CLIP features trained on our 4 fruits: overfits to them (21/36).
- Extra "dark spots / mould" rotten prompts: fixed one tomato, broke four fresh photos.

Chosen: CLIP ViT-B/32 (OpenCLIP, LAION-400M) with per-produce prompts, no training:
    python3 scripts/export_clip.py      # builds checkpoints/clip_*
    python3 scripts/evaluate.py
      produce 34/34 | mixed 24/25 | samples 12/12 | total 70/71
      (samples were 11/12 before the 3-tomato sample was cropped to one tomato)
    same 71 photos with the v2 EfficientNet: 47/71
    4-fruit Kaggle set (4,000 photos, zero-shot): balanced accuracy 0.912

## Colour profiles and the zoom-in experiment

The browser version called the tomato sample fresh while Python called it rotten.
Cause: Mac screenshots embed a Display P3 colour profile; browsers convert it to
sRGB, Python was reading the raw numbers. clip_detector now converts to sRGB too.

Zoom-in crops (whole photo + 5 crops, colour-managed), whole-photo vs best rule:
    1,200 held-out Kaggle photos: 1135 -> 1141 to 1147 correct
    71 real-world photos:          69 -> 67 to 68 correct, and 3x slower in the browser
Dropped. The 3-tomato sample was cropped to the single rotten tomato instead (99.9% rotten).
