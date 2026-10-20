# 🍎 Fruit Detector

**Is it fresh or rotten?** Upload a photo of a fruit or vegetable and get a verdict
with a confidence score. It says *"not sure"* when it doesn't know, and *"that's not
a fruit"* when it isn't one.

**Live demo:** https://fruit-detector-jywf8hr9vrvfnj8appppdlxl.streamlit.app

![screenshot](docs/screenshot.png)

## How it works

The app uses **CLIP** (OpenCLIP ViT-B/32, trained on 400 million captioned images).
For 64 kinds of produce it compares the photo with captions like *"a photo of a
fresh watermelon"* and *"a photo of a rotten watermelon"*. A single softmax over all
128 captions says both what the produce probably is and how much of the vote went to
"rotten". A few extra captions (people, pets, screenshots...) catch photos that aren't
produce at all. Nothing is trained on our photos, which is exactly why it generalises.

## The interesting part: three versions, and what each one got wrong

| Version | Idea | What broke it |
| --- | --- | --- |
| v1 | ResNet18 fine-tuned on 7 screenshots | Learned the **background** (fresh on white, rotten on grey). Its "best" checkpoint scored 50% on its own validation set. |
| v2 | EfficientNet-B0 fine-tuned on 14k photos of 4 fruits, with fruit-held-out validation, balanced sampling, strong augmentation, calibration | 91% on held-out *fruits*, but called fresh **potatoes, beetroot, ginger and a cut watermelon rotten**. With four fruits as its only fresh examples, it learned "brown / dark / irregular = rotten". |
| **v3** | **CLIP zero-shot, per-produce prompts** | Two misses in 71 test photos (a subtly mouldy lemon, a tomato with black spots). |

### Results on 71 real photos the model never saw

Reproduce with `python3 scripts/evaluate.py` (downloads the test photos from pinned repos).

| Test set | v2 EfficientNet | **v3 CLIP** |
| --- | --- | --- |
| 34 fresh fruits & vegetables, many cut (watermelon slice, pomegranate, kiwi, beetroot...) | 21 / 34 | **34 / 34** |
| 25 fresh *and* rotten, mostly other types (lemon, mango, tamarillo, lulo, potato, tomato...) | 15 / 25 | **24 / 25** |
| 12 app samples | 11 / 12 | 11 / 12 |
| **Total** | 47 / 71 (66%) | **69 / 71 (97%)** |

On the 4,000-photo apple / banana / orange / strawberry test set, CLIP with no training
gets **91.2%** balanced accuracy, about the same as v2's 90.8% leave-one-fruit-out score.
None of the 394 real produce photos checked were wrongly refused as "not a fruit"; all
6 non-produce test images (screenshots, a document, landscapes) were.

### Things I tried that didn't work (also worth knowing)

- **More fresh variety for v2** (3k Fruits-360 images of 100+ types): no gain on real
  photos, worse rot detection. Those are 100x100 studio shots on white, so the model
  learned *"studio photo = fresh"*, just a new shortcut.
- **Training a classifier on CLIP features** using our four fruits: it re-learned v2's
  bias (21/36 on fresh produce).
- **Extra "dark spots / mould" prompts**: fixed the tomato but broke four fresh photos,
  so the original prompts stayed.

v2's leave-one-fruit-out numbers, backbone comparison and training commands are in
[`docs/training_log.md`](docs/training_log.md).

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py                          # the app
python3 src/clip_detector.py samples/         # score photos from the terminal
python3 scripts/evaluate.py                   # reproduce the 71-photo results
```

Rebuild the CLIP files (`checkpoints/clip_*`) from the public checkpoint with
`python3 scripts/export_clip.py`. To retrain the v2 EfficientNet:
`python3 scripts/download_data.py && python3 src/train.py --cv`.

## Project layout

```
app.py                    Streamlit app (two-column pixel-art "desktop")
src/clip_detector.py      v3: CLIP image encoder + pre-encoded fresh/rotten prompts
src/ui.py                 pixel-art look: colours, fonts, windows, sprites, verdict card
src/dataset.py, model.py, train.py, inference.py    v2 EfficientNet pipeline
scripts/export_clip.py    builds checkpoints/clip_* from the OpenCLIP release
scripts/evaluate.py       71-photo test on images the model never saw
scripts/download_data.py  v2 training data from public GitHub mirrors
scripts/make_background.py  draws the pixel landscape -> assets/background.png
checkpoints/              CLIP encoder (2 fp16 shards < 100 MB) + prompts; v2 weights
samples/                  one-click examples in the app
static/fonts/             Pixelify Sans + VT323 (served by Streamlit, no CDN)
```

## Credits

- Model: [OpenCLIP](https://github.com/mlfoundations/open_clip) ViT-B/32 (LAION-400M), MIT licence
- v2 data: *Fruits fresh and rotten for classification* (Sriram Reddy Kalluri, Kaggle) via the
  [Bangkit-JKT2-D mirror](https://github.com/Bangkit-JKT2-D/fruits-fresh-rotten-classification), and
  [1RaghavM/Fruit-Freshness-Classification](https://github.com/1RaghavM/Fruit-Freshness-Classification);
  backbone weights from [timm](https://github.com/huggingface/pytorch-image-models)
- Test photos: [junaiddbz/agrivision-produce-classifier](https://github.com/junaiddbz/agrivision-produce-classifier),
  [dhieu0901/Fresh-and-Rotten-Fruit-Classification](https://github.com/dhieu0901/Fresh-and-Rotten-Fruit-Classification),
  [Jayasree478/FoodSafe-AI](https://github.com/Jayasree478/FoodSafe-AI)
- Fonts: [Pixelify Sans](https://github.com/google/fonts/tree/main/ofl/pixelifysans) and
  [VT323](https://github.com/google/fonts/tree/main/ofl/vt323), SIL Open Font License (copies in `static/fonts/`)
- Pixel-art background and sprites: original, drawn in code

## Limitations

Best on one piece of produce filling most of a well-lit photo. Very early or subtle
rot (a slightly soft lemon, a few dark spots) can read as fresh, and rotten
strawberries are its weakest case (59% caught on the 4-fruit test set). It reports
"not sure" below 70% confidence rather than guessing.
