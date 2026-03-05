# import streamlit as st
# import torch
# import numpy as np
# import cv2
# from PIL import Image
# from torchvision import transforms

# from model import PneumoniaCNN
# from gradcam import GradCAM
# from inference import predict
# from explanation_features import build_explanation

# # -------------------- CONFIG --------------------
# st.set_page_config(
#     page_title="AI Pneumonia Explanation System",
#     layout="wide"
# )

# st.title("🩺 AI That Explains Pneumonia Like a Doctor")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if device.type == "cuda":
#     torch.cuda.init()

# # -------------------- LOAD MODEL --------------------
# @st.cache_resource
# def load_model():
#     model = PneumoniaCNN().to(device)
#     model.load_state_dict(
#         torch.load(
#             r"D:\D drive\Project\AI That Explains Medical Images Like a Doctor\Another way\model creation\pneumonia_cnn.pth",
#             map_location=device,
#             weights_only=True
#         )
#     )
#     model.eval()
#     return model

# model = load_model()
# gradcam = GradCAM(model, model.features[6])

# # -------------------- TRANSFORM --------------------
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor()
# ])

# # -------------------- UI --------------------
# uploaded_file = st.file_uploader(
#     "Upload Chest X-ray Image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file:
#     # ---------- Load Image ----------
#     image = Image.open(uploaded_file).convert("RGB")
#     image_resized = image.resize((224, 224))
#     image_np = np.array(image_resized)

#     image_tensor = transform(image).unsqueeze(0).to(device)

#     # ---------- Prediction ----------
#     prob, bbox = predict(model, image_tensor)
#     label = "PNEUMONIA" if prob >= 0.5 else "NO PNEUMONIA"

#     st.subheader("🧪 Prediction Result")
#     st.write(f"**Prediction:** {label}")
#     st.write(f"**Confidence:** {prob:.2f}")

#     # ---------- Grad-CAM ----------
#     heatmap = gradcam.generate(image_tensor)

#     heatmap_color = cv2.applyColorMap(
#         np.uint8(255 * heatmap),
#         cv2.COLORMAP_JET
#     )

#     gradcam_overlay = cv2.addWeighted(
#         image_np, 0.6, heatmap_color, 0.4, 0
#     )

#     # ---------- Bounding Box ----------
#     h, w, _ = image_np.shape
#     x, y, bw, bh = bbox

#     x1 = int(x * w)
#     y1 = int(y * h)
#     x2 = int((x + bw) * w)
#     y2 = int((y + bh) * h)

#     bbox_image = image_np.copy()
#     cv2.rectangle(
#         bbox_image,
#         (x1, y1),
#         (x2, y2),
#         (0, 255, 0),
#         2
#     )

#     # ---------- Combined ----------
#     combined_image = gradcam_overlay.copy()
#     cv2.rectangle(
#         combined_image,
#         (x1, y1),
#         (x2, y2),
#         (255, 255, 255),
#         2
#     )

#     # ---------- Display ----------
#     st.subheader("🖼️ Visual Explanation")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.image(image_np, caption="Original X-ray", use_container_width=True)

#     with col2:
#         st.image(gradcam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)

#     with col3:
#         st.image(bbox_image, caption="Bounding Box Localization", use_container_width=True)

#     st.subheader("🔥 Combined View")
#     st.image(
#         combined_image,
#         caption="Grad-CAM + Bounding Box",
#         use_container_width=True
#     )

#     # ---------- Explanation JSON ----------
#     explanation = build_explanation(
#         uploaded_file.name,
#         prob,
#         bbox,
#         heatmap
#     )

#     st.subheader("🩺 AI Medical Reasoning (Structured)")
#     st.json(explanation)

#     st.warning(
#         "⚠️ This AI output is for educational and research purposes only. "
#         "Consult a qualified medical professional for diagnosis."
#     )


import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import ollama

from model import PneumoniaCNN
from gradcam import GradCAM
from inference import predict
from explanation_features import build_explanation

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="AI Pneumonia Explanation System",
    layout="wide"
)

st.title("🩺 AI That Explains Pneumonia Like a Doctor")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.init()

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = PneumoniaCNN().to(device)
    model.load_state_dict(
        torch.load(
            r"D:\D drive\Project\AI That Explains Medical Images Like a Doctor\model creation\pneumonia_cnn.pth",
            map_location=device,
            weights_only=True
        )
    )
    model.eval()
    return model

model = load_model()
gradcam = GradCAM(model, model.features[6])

# -------------------- TRANSFORM --------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# -------------------- UI --------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # ---------- Load Image ----------
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized)

    image_tensor = transform(image).unsqueeze(0).to(device)

    # ---------- Prediction ----------
    prob, bbox = predict(model, image_tensor)
    label = "PNEUMONIA" if prob >= 0.34011123 else "NO PNEUMONIA"

    st.subheader("🧪 Prediction Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {prob:.2f}")

    # ---------- Grad-CAM ----------
    heatmap = gradcam.generate(image_tensor)

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    gradcam_overlay = cv2.addWeighted(
        image_np, 0.6, heatmap_color, 0.4, 0
    )

    # ---------- Bounding Box ----------
    h, w, _ = image_np.shape
    x, y, bw, bh = bbox

    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + bw) * w)
    y2 = int((y + bh) * h)

    bbox_image = image_np.copy()
    cv2.rectangle(
        bbox_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # ---------- Combined View ----------
    combined_image = gradcam_overlay.copy()
    cv2.rectangle(
        combined_image,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    # ---------- Display Images ----------
    st.subheader("🖼️ Visual Explanation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_np, caption="Original X-ray", use_container_width=True)

    with col2:
        st.image(gradcam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)

    with col3:
        st.image(bbox_image, caption="Bounding Box Localization", use_container_width=True)

    st.subheader("🔥 Combined View (Grad-CAM + Bounding Box)")
    st.image(
        combined_image,
        caption="Model Attention + Localization",
        use_container_width=True
    )

    # ---------- Structured Explanation ----------
    explanation = build_explanation(
        uploaded_file.name,
        prob,
        bbox,
        heatmap
    )

    st.subheader("📊 Model Output (Structured Features)")
    st.json(explanation)

    # ---------- LLM EXPLANATION ----------
    st.subheader("🧠 AI Medical Explanation")

    with st.spinner("Generating medical explanation using LLM..."):
        llm_prompt = f"""
You are a medical AI assistant.

Explain the following AI-based pneumonia detection result
in simple, clinically accurate language.

Do NOT provide a diagnosis.
Add a short medical disclaimer.

also provide the explanation in a structured format.
make sure the normal human readable explanation is at least 100 words long.

Structured findings:
{explanation}
"""

        response = ollama.chat(
            model="llama3.1",
            messages=[
                {"role": "user", "content": llm_prompt}
            ]
        )

    st.markdown(response["message"]["content"])

    st.warning(
        "⚠️ This AI-generated explanation is for educational purposes only "
        "and must not be used as a medical diagnosis."
    )
