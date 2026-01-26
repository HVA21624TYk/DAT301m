import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

from huggingface_hub import hf_hub_download
from models import build_densenet121, build_vit

# =========================
# CONFIG
# =========================
CLASSES = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
IMG_SIZE = 224

# Hugging Face model repo (your repo)
HF_REPO = "Anh2162004/DAT301m"

# Checkpoint filenames on Hugging Face
DENSENET_FILE = "best_densenet121_4class.pth"
VIT_FILE      = "best_vit21k_4class.pth"

# ViT model name (MUST match training)
VIT_NAME = "vit_base_patch16_224_in21k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

def get_hf_token():
    """
    If the Hugging Face repo is public, no token is required.
    If the repo is private, add HF_TOKEN to Streamlit Secrets.
    """
    try:
        return st.secrets.get("HF_TOKEN", None)
    except Exception:
        return None

@st.cache_resource
def load_densenet():
    token = get_hf_token()
    ckpt_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=DENSENET_FILE,
        token=token
    )

    model = build_densenet121(num_classes=len(CLASSES))
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_vit():
    token = get_hf_token()
    ckpt_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=VIT_FILE,
        token=token
    )

    model = build_vit(VIT_NAME, num_classes=len(CLASSES))
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def predict(model, pil_img: Image.Image):
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

def probs_table(probs):
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

# =========================
# UI
# =========================
st.set_page_config(
    page_title="X-Ray Demo (DenseNet + ViT)",
    layout="centered"
)

st.title("🫁 X-Ray Classification Demo")
st.write("Demo **2 model**: DenseNet121 and ViT (pretrained IN-21K). Upload images X-ray to predict.")

with st.expander("Current configuration", expanded=False):
    st.code(
        f"HF_REPO = {HF_REPO}\n"
        f"DENSENET_FILE = {DENSENET_FILE}\n"
        f"VIT_FILE = {VIT_FILE}\n"
        f"VIT_NAME = {VIT_NAME}\n"
        f"DEVICE = {DEVICE}"
    )

model_choice = st.selectbox(
    "Select model:",
    ["DenseNet121", "ViT-21K", "Compare both models"]
)

uploaded = st.file_uploader(
    "Upload X-ray image (.png / .jpg / .jpeg)",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Input image", use_container_width=True)

    if model_choice == "DenseNet121":
        with st.spinner("Loading DenseNet121 (weights will be downloaded on first run)..."):
            model = load_densenet()

        pred_idx, probs = predict(model, img)

        st.subheader("Result (DenseNet121)")
        st.metric("Predicted class", CLASSES[pred_idx])
        st.dataframe(probs_table(probs))
        st.bar_chart(probs_table(probs))

    elif model_choice == "ViT-21K":
        with st.spinner("Loading ViT-21K (weights will be downloaded on first run)..."):
            model = load_vit()

        pred_idx, probs = predict(model, img)

        st.subheader(f"Result (ViT-21K: {VIT_NAME})")
        st.metric("Predicted class", CLASSES[pred_idx])
        st.dataframe(probs_table(probs))
        st.bar_chart(probs_table(probs))

    else:
        with st.spinner("Loading both models (weights will be downloaded on first run)..."):
            densenet = load_densenet()
            vit = load_vit()

        d_pred, d_probs = predict(densenet, img)
        v_pred, v_probs = predict(vit, img)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("DenseNet121")
            st.metric("Predicted class", CLASSES[d_pred])
            st.dataframe(probs_table(d_probs))
            st.bar_chart(probs_table(d_probs))

        with col2:
            st.subheader("ViT-21K")
            st.metric("Predicted class", CLASSES[v_pred])
            st.dataframe(probs_table(v_probs))
            st.bar_chart(probs_table(v_probs))

    st.caption(f"Running on: {DEVICE}")
