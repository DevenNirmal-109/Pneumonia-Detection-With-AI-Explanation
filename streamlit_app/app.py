import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import PneumoniaCNN

st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩺 AI Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to detect pneumonia and localize it.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: **{device}**")

@st.cache_resource
def load_model():
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(r"D:\D drive\Project\AI That Explains Medical Images Like a Doctor\model creation\pneumonia_cnn.pth", map_location=device))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = image.size

    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        obj_logit, bbox = model(img_tensor)
        prob = torch.sigmoid(obj_logit).item()
        bbox = bbox.squeeze().cpu().numpy()

threshold = st.slider(
    "Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.markdown(f"### 🧪 Pneumonia Probability: **{prob:.2f}**")

if prob < threshold:
    st.success("✅ No Pneumonia Detected")
else:
    st.error("⚠️ Pneumonia Detected")

    # Convert bbox to pixel coordinates
    x, y, w, h = bbox
    x *= orig_w
    y *= orig_h
    w *= orig_w
    h *= orig_h

    # Draw bounding box
    img_np = np.array(image)
    fig, ax = plt.subplots()
    ax.imshow(img_np)

    rect = plt.Rectangle(
        (x, y), w, h,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rect)

    ax.set_title("Detected Pneumonia Region")
    ax.axis("off")

    st.pyplot(fig)
