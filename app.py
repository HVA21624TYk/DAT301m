# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

from models import build_densenet121, build_vit


# =========================
# CONFIG (khớp lúc train)
# =========================
CLASSES = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
IMG_SIZE = 224

# ViT bạn train (đổi đúng tên đã train)
VIT_NAME = "vit_base_patch16_224_in21k"

# Checkpoints (đổi đúng tên file của bạn)
CKPT_DENSENET = "best_densenet121_4class.pth"
CKPT_VIT      = "best_vit21k_4class.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# preprocess giống code train của bạn
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


@st.cache_resource
def load_densenet():
    model = build_densenet121(num_classes=len(CLASSES))
    state = torch.load(CKPT_DENSENET, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_vit():
    model = build_vit(VIT_NAME, num_classes=len(CLASSES))
    state = torch.load(CKPT_VIT, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def predict(model, pil_img: Image.Image):
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def probs_table(probs):
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


# =========================
# UI
# =========================
st.set_page_config(page_title="X-Ray Demo (DenseNet + ViT)", layout="centered")
st.title("🫁 X-Ray Classification Demo")
st.write("Demo **2 model**: DenseNet121 and ViT (pretrained IN-21K). Upload images X-ray to predict.")

model_choice = st.selectbox(
    "Chosse model to use:",
    ["DenseNet121", "ViT-21K", "So sánh cả hai"]
)

uploaded = st.file_uploader("Upload images (.png/.jpg/.jpeg)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="inputs images", use_container_width=True)

    if model_choice == "DenseNet121":
        model = load_densenet()
        pred_idx, probs = predict(model, img)

        st.subheader("Results (DenseNet121)")
        st.metric("Predicted class", CLASSES[pred_idx])
        st.dataframe(probs_table(probs))
        st.bar_chart(probs_table(probs))

    elif model_choice == "ViT-21K":
        model = load_vit()
        pred_idx, probs = predict(model, img)

        st.subheader(f"Results (ViT-21K: {VIT_NAME})")
        st.metric("Predicted class", CLASSES[pred_idx])
        st.dataframe(probs_table(probs))
        st.bar_chart(probs_table(probs))

    else:
        densenet = load_densenet()
        vit = load_vit()

        d_pred, d_probs = predict(densenet, img)
        v_pred, v_probs = predict(vit, img)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("DenseNet121")
            st.metric("Predicted", CLASSES[d_pred])
            st.dataframe(probs_table(d_probs))
            st.bar_chart(probs_table(d_probs))

        with col2:
            st.subheader("ViT-21K")
            st.metric("Predicted", CLASSES[v_pred])
            st.dataframe(probs_table(v_probs))
            st.bar_chart(probs_table(v_probs))

    st.caption(f"Device: {DEVICE}")
