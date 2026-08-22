"""
app.py
------
VisionGuard's interface: upload an image, get a fresh/rotten prediction
with a confidence score. Minecraft-themed skin on top of the exact same
predict_image() logic already verified from the terminal in inference.py
-- no separate/duplicate prediction code to drift out of sync.

Run from the project root:
    streamlit run app.py
"""

import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from inference import load_model, predict_image  # noqa: E402

CHECKPOINT_PATH = "checkpoints/best_model.pth"

st.set_page_config(page_title="VisionGuard", page_icon="🟩", layout="centered")

MINECRAFT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

html, body, [class*="css"] {
    font-family: 'Press Start 2P', monospace;
}

.stApp {
    background: linear-gradient(180deg, #7EC0EE 0%, #7EC0EE 32%, #6ABE30 32%, #4F8F26 100%);
    background-attachment: fixed;
}

.block-container {
    padding-top: 2.5rem;
    max-width: 640px;
}

.mc-title {
    text-align: center;
    color: #FFFFFF;
    font-size: 26px;
    line-height: 1.4;
    text-shadow: 3px 3px 0px #3F3F3F;
    margin-bottom: 6px;
    letter-spacing: 1px;
}

.mc-subtitle {
    text-align: center;
    color: #FFFFFF;
    font-size: 10px;
    text-shadow: 2px 2px 0px #3F3F3F;
    margin-bottom: 22px;
}

.mc-panel {
    background: #C6C6C6;
    border-top: 4px solid #FFFFFF;
    border-left: 4px solid #FFFFFF;
    border-right: 4px solid #565656;
    border-bottom: 4px solid #565656;
    box-shadow: 0 0 0 4px #000000;
    padding: 22px;
    margin-bottom: 20px;
}

.mc-label {
    color: #2B2B2B;
    font-size: 11px;
    margin-bottom: 10px;
    text-shadow: 1px 1px 0px #FFFFFF;
}

[data-testid="stFileUploader"] {
    background: #8B8B8B;
    border: 4px dashed #3F3F3F;
    padding: 14px;
}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    font-family: 'Press Start 2P', monospace;
}

.stButton > button {
    font-family: 'Press Start 2P', monospace;
    background: #6ABE30;
    color: #FFFFFF;
    border-top: 4px solid #A0E86B;
    border-left: 4px solid #A0E86B;
    border-right: 4px solid #3F7A1A;
    border-bottom: 4px solid #3F7A1A;
    border-radius: 0;
    padding: 12px 18px;
    font-size: 11px;
    text-shadow: 2px 2px 0px #3F3F3F;
    box-shadow: 0 0 0 2px #000000;
    width: 100%;
    transition: none;
}

.stButton > button:hover {
    background: #7ECF41;
    color: #FFFFFF;
    border-color: #A0E86B #3F7A1A #3F7A1A #A0E86B;
}

.stButton > button:active {
    border-top: 4px solid #3F7A1A;
    border-left: 4px solid #3F7A1A;
    border-right: 4px solid #A0E86B;
    border-bottom: 4px solid #A0E86B;
}

.mc-result-fresh, .mc-result-spoiled {
    border: 4px solid #000000;
    color: #FFFFFF;
    padding: 18px;
    text-align: center;
    font-size: 13px;
    line-height: 1.8;
    text-shadow: 2px 2px 0px #000000;
    margin-top: 18px;
}

.mc-result-fresh {
    background: #3F7A1A;
    box-shadow: inset 0 0 0 3px #6ABE30;
}

.mc-result-spoiled {
    background: #7A1A1A;
    box-shadow: inset 0 0 0 3px #BE3030;
}

.mc-bar-bg {
    background: #3F3F3F;
    border: 3px solid #000000;
    height: 22px;
    margin-top: 14px;
    padding: 2px;
}

.mc-bar-fill-fresh {
    background: repeating-linear-gradient(90deg, #6ABE30, #6ABE30 8px, #7ECF41 8px, #7ECF41 16px);
    height: 100%;
}

.mc-bar-fill-spoiled {
    background: repeating-linear-gradient(90deg, #BE3030, #BE3030 8px, #CF4141 8px, #CF4141 16px);
    height: 100%;
}

.mc-footer {
    text-align: center;
    color: #FFFFFF;
    font-size: 8px;
    line-height: 1.8;
    text-shadow: 1px 1px 0px #3F3F3F;
    margin-top: 22px;
    opacity: 0.9;
}
</style>
"""

st.markdown(MINECRAFT_CSS, unsafe_allow_html=True)

st.markdown('<div class="mc-title">⛏️ VISIONGUARD ⛏️</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mc-subtitle">FRUIT &amp; VEGGIE FRESHNESS INSPECTOR</div>',
    unsafe_allow_html=True,
)

if not os.path.exists(CHECKPOINT_PATH):
    st.error(
        "No trained model found yet. Run `python3 src/train.py` first "
        "(from the project root, in your terminal), then restart this app."
    )
    st.stop()


@st.cache_resource
def get_model():
    return load_model(CHECKPOINT_PATH)


model, classes, device = get_model()

st.markdown('<div class="mc-panel">', unsafe_allow_html=True)
st.markdown('<div class="mc-label">&gt; DROP AN ITEM INTO THE SLOT:</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ITEM IN INVENTORY", use_container_width=True)

    if st.button("🔍 INSPECT ITEM"):
        with st.spinner("MINING FOR CLUES..."):
            pred_class, confidence, all_probs = predict_image(image, model, classes, device)

        if pred_class == "fresh":
            st.markdown(
                f'<div class="mc-result-fresh">✅ CERTIFIED FRESH<br>{confidence * 100:.1f}% CONFIDENCE</div>',
                unsafe_allow_html=True,
            )
            bar_class = "mc-bar-fill-fresh"
        else:
            st.markdown(
                f'<div class="mc-result-spoiled">☠️ ROTTEN — DO NOT EAT<br>{confidence * 100:.1f}% CONFIDENCE</div>',
                unsafe_allow_html=True,
            )
            bar_class = "mc-bar-fill-spoiled"

        st.markdown(
            f'<div class="mc-bar-bg"><div class="{bar_class}" style="width:{confidence * 100:.0f}%"></div></div>',
            unsafe_allow_html=True,
        )

        with st.expander("📊 FULL PROBABILITY BREAKDOWN"):
            for cls, p in all_probs.items():
                st.write(f"{cls.upper()}: {p * 100:.1f}%")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="mc-footer">POWERED BY A FINE-TUNED RESNET18 &middot; TRANSFER LEARNING &middot; PYTORCH</div>',
    unsafe_allow_html=True,
)