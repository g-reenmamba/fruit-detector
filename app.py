"""
app.py
------
Fruit Detector's interface: upload a photo of a fruit or vegetable and get
a fresh / rotten verdict with a confidence score, or an honest "not sure".
The pixel-art look lives in src/ui.py; predictions come from CLIP via
src/clip_detector.py (also runnable from the terminal).

Run from the project root:
    streamlit run app.py
"""

import os
import sys

import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import ui  # noqa: E402
import json  # noqa: E402

import clip_detector  # noqa: E402
from clip_detector import UNSURE_BELOW  # noqa: E402

CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "clip_prompts.pt")
SAMPLES_DIR = os.path.join(ROOT, "samples")
REPO_URL = "https://github.com/g-reenmamba/fruit-detector"
with open(os.path.join(ROOT, "src", "produce_info.json"), encoding="utf-8") as fh:
    PRODUCE_INFO = json.load(fh)["produce"]   # facts + storage / spoilage advice per produce type

st.set_page_config(page_title="Fruit Detector", page_icon="🍎", layout="wide")
st.markdown(ui.css(), unsafe_allow_html=True)
st.markdown(ui.hero(), unsafe_allow_html=True)

if not os.path.exists(CHECKPOINT_PATH):
    st.error("Model files are missing. Run `python3 scripts/export_clip.py`, then restart the app.")
    st.stop()


@st.cache_resource
def get_detector():
    return clip_detector.load_detector()


detector = get_detector()


def judge(image):
    r = clip_detector.analyse(image, detector)
    if r["not_produce"] > clip_detector.NOT_PRODUCE_ABOVE:
        return dict(label="not_produce", pred=None, conf=r["not_produce"], probs={}, guess=None)
    pred = "rotten" if r["rotten"] > 0.5 else "fresh"
    conf = max(r["rotten"], r["fresh"])
    return dict(label=pred if conf >= UNSURE_BELOW else "unsure", pred=pred, conf=conf,
                probs={"fresh": r["fresh"], "rotten": r["rotten"]},
                guess=r["produce"] if r["produce_conf"] >= 0.3 else None)

sample_files = sorted(
    f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
) if os.path.isdir(SAMPLES_DIR) else []


def sample_name(fname):
    state, _, fruit = os.path.splitext(fname)[0].partition("_")
    fruit = fruit.replace("_2", "").replace("_", " ")
    return f"{state.capitalize()} {fruit}" + (" (another one)" if fname.endswith("_2.png") else "")


# Two windows side by side on wide screens (they stack on phones). Streamlit can't
# wrap widgets in raw HTML, so each window is a keyed container; Streamlit gives it
# the CSS class .st-key-win_<name>, which src/ui.py styles as a retro window.
left, right = st.columns([1.12, 1], gap="large")

with left, st.container(key="win_main"):
    st.markdown(ui.titlebar("fruit_detector.exe"), unsafe_allow_html=True)
    st.markdown('<p class="fd-label">Upload a photo of one fruit</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a photo of one fruit", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
    )

    image, image_key = None, None
    if uploaded_file is not None:
        image, image_key = Image.open(uploaded_file).convert("RGB"), f"upload:{uploaded_file.file_id}"
    elif sample_files:
        choice = st.selectbox("No photo handy? Try one of ours", sample_files, index=None,
                              placeholder="Pick a sample fruit", format_func=sample_name)
        if choice:
            image, image_key = Image.open(os.path.join(SAMPLES_DIR, choice)).convert("RGB"), f"sample:{choice}"

    if image is None:
        st.markdown(ui.photo_slot(), unsafe_allow_html=True)
    else:
        st.image(image, width="stretch")
        if st.button("Check freshness"):
            with st.spinner("Looking closely..."):
                # keep the result until a different photo is chosen, so it survives reruns
                st.session_state.result = dict(key=image_key, **judge(image))

with right:
    with st.container(key="win_verdict"):
        st.markdown(ui.titlebar("verdict.txt"), unsafe_allow_html=True)
        r = st.session_state.get("result")
        if r and image_key and r["key"] == image_key:
            st.markdown(ui.verdict_card(r["label"], r["pred"], r["conf"], r["probs"], r["guess"]),
                        unsafe_allow_html=True)
        else:
            st.markdown(ui.waiting(), unsafe_allow_html=True)

    with st.container(key="win_info"):
        st.markdown(ui.titlebar("fruit_info.txt"), unsafe_allow_html=True)
        shown = r if (r and image_key and r["key"] == image_key) else None
        st.markdown(ui.fruit_info(shown, PRODUCE_INFO, REPO_URL), unsafe_allow_html=True)
