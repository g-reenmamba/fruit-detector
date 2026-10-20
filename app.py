"""
app.py
------
Fruit Detector's interface: upload a photo of a fruit or vegetable and get
a fresh / rotten verdict with a confidence score, or an honest "not sure".
The pixel-art look lives in src/ui.py; predictions come from the same
predict_image() used on the command line in src/inference.py.

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
from inference import load_model, predict_image, verdict  # noqa: E402

CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "best_model.pth")
SAMPLES_DIR = os.path.join(ROOT, "samples")
REPO_URL = "https://github.com/g-reenmamba/fruit-detector"
ARCH_LABELS = {"efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet18", "resnet50": "ResNet50",
               "mobilenetv3_large_100": "MobileNetV3", "efficientnet_b3": "EfficientNet-B3"}

st.set_page_config(page_title="Fruit Detector", page_icon="🍎", layout="wide")
st.markdown(ui.css(), unsafe_allow_html=True)
st.markdown(ui.hero(), unsafe_allow_html=True)

if not os.path.exists(CHECKPOINT_PATH):
    st.error("No trained model yet. Run `python3 scripts/download_data.py`, then "
             "`python3 src/train.py`, then restart the app.")
    st.stop()


@st.cache_resource
def get_model():
    return load_model(CHECKPOINT_PATH)


model, classes, device = get_model()

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
                pred_class, confidence, all_probs = predict_image(image, model, classes, device)
            # keep the result until a different photo is chosen, so it survives reruns
            st.session_state.result = dict(key=image_key, pred=pred_class, conf=confidence, probs=all_probs)

with right:
    with st.container(key="win_verdict"):
        st.markdown(ui.titlebar("verdict.txt"), unsafe_allow_html=True)
        r = st.session_state.get("result")
        if r and image_key and r["key"] == image_key:
            st.markdown(ui.verdict_card(verdict(r["pred"], r["conf"]), r["pred"], r["conf"], r["probs"]),
                        unsafe_allow_html=True)
        else:
            st.markdown(ui.waiting(), unsafe_allow_html=True)

    with st.container(key="win_about"):
        st.markdown(ui.titlebar("about.txt"), unsafe_allow_html=True)
        st.markdown(ui.about(ARCH_LABELS.get(model.arch, "CNN"), REPO_URL), unsafe_allow_html=True)
