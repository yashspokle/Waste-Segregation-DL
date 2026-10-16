
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "efficientnet_lite" / "best_model.keras"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

INFO = {
    "cardboard": ("📦", "Paper/cardboard recycling", "Keep dry and flatten boxes."),
    "glass": ("🫙", "Glass recycling", "Handle broken glass carefully."),
    "metal": ("🥫", "Metal recycling", "Empty containers before disposal."),
    "paper": ("📄", "Paper recycling", "Keep paper clean and dry."),
    "plastic": ("🧴", "Plastic recycling", "Check local plastic recycling rules."),
    "trash": ("🗑️", "General waste", "Check whether the item can be reused."),
}

st.set_page_config(
    page_title="WasteWise",
    page_icon="♻️",
    layout="wide",
)

# Simple dark theme
st.markdown("""
<style>
.stApp {
    background: #111315;
    color: #F0F0F0;
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
h1, h2, h3, p, label {
    color: #F0F0F0;
}

div[data-testid="stFileUploader"] {
    background: #202326;
    border: 1px solid #41464B;
    border-radius: 12px;
    padding: 12px;
}
div[data-testid="stFileUploader"] section {
    background: #202326;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: #292D31 !important;
    border: 1px solid #50565B !important;
    border-radius: 8px !important;
}
.stButton > button {
    width: 100%;
    min-height: 48px;
    background: #347653 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}
.stButton > button:hover {
    background: #419565 !important;
}
div[data-testid="stMetric"] {
    background: #202326;
    border: 1px solid #41464B;
    padding: 14px;
    border-radius: 12px;
}

/* Larger, more visual scanning mode buttons */
div[role="radiogroup"] {
    display: flex;
    gap: 16px;
    margin: 16px 0 24px;
    flex-wrap: wrap;
}

div[role="radiogroup"] > label {
    background: #202820;
    border: 2px solid #46574A;
    border-radius: 14px;
    padding: 18px 26px;
    min-height: 64px;
    flex: 1;
    justify-content: center;
    transition: all 0.2s ease;
}


/* Large, clickable scanning mode options */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 16px !important;
    width: 100% !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background-color: #202820 !important;
    border: 2px solid #46574A !important;
    border-radius: 16px !important;
    padding: 22px 24px !important;
    min-height: 76px !important;
    flex: 1 1 0 !important;
    cursor: pointer !important;
    align-items: center !important;
    gap: 12px !important;
    transition: 0.2s ease !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    background-color: #293B2E !important;
    border-color: #75B783 !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label p {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #F1F5EF !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    transform: scale(1.25);
}

@media (max-width: 600px) {
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        flex-direction: column !important;
    }
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def predict(image):
    model = load_model()
    image = image.convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32)[None, ...]
    scores = np.asarray(model(array, training=False))[0]

    index = int(np.argmax(scores))
    return CLASSES[index], float(scores[index]), scores


def show_result(result):
    category, confidence, scores = result
    icon, bin_name, tip = INFO[category]

    st.divider()
    st.subheader("🔎 Detection Result")

    st.success(f"{icon} Predicted waste: **{category.title()}**")
    st.metric("Model score", f"{confidence * 100:.2f}%")

    st.markdown(f"### ♻️ Recommended destination")
    st.info(bin_name)
    st.caption(f"💡 {tip}")

    with st.expander("View scores for all categories"):
        for name, score in zip(CLASSES, scores):
            st.write(f"{name.title()}: {score * 100:.2f}%")
            st.progress(float(np.clip(score, 0, 1)))

    st.caption(
        "Model score does not guarantee correctness. "
        "Follow your local waste disposal guidelines."
    )


st.title("♻️ WasteWise")
st.write("### Waste Segregation System")
st.write("Upload a photo or capture one using your camera to classify waste in 6 categories.")

try:
    load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

mode = st.radio(
    "Choose scanning method",
    ["📁 Upload Image", "📷 Camera Scan"],
    horizontal=True,
)

if mode == "📁 Upload Image":
    st.subheader("Upload waste image")

    uploaded = st.file_uploader(
        "Choose JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        key="waste_upload",
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Selected image", width=450)

            # Always displayed whenever an image is uploaded
            if st.button(
                "🔍 DETECT WASTE",
                key="upload_detect",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Detecting waste..."):
                    result = predict(image)
                    st.session_state["upload_result"] = result

            if "upload_result" in st.session_state:
                show_result(st.session_state["upload_result"])

        except Exception as error:
            st.error(f"Could not process image: {error}")
    else:
        st.info("Upload an image first. The Detect Waste button appears below it.")

else:
    st.subheader("Camera Scan")
    st.write("Allow camera permission, point your camera at the item, and capture it.")

    camera_photo = st.camera_input(
        "Capture waste image",
        key="waste_camera",
    )

    if camera_photo is not None:
        try:
            camera_image = Image.open(camera_photo).convert("RGB")

            # Always displayed after capturing a photo
            if st.button(
                "📷 DETECT CAPTURED WASTE",
                key="camera_detect",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Analysing captured image..."):
                    result = predict(camera_image)
                    st.session_state["camera_result"] = result

            if "camera_result" in st.session_state:
                show_result(st.session_state["camera_result"])

        except Exception as error:
            st.error(f"Could not process camera image: {error}")

st.divider()
st.caption("WasteWise | Waste Segregation ")
