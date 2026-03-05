import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import PneumoniaCNN
from gradcam import GradCAM
from inference import predict
from explanation_features import build_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load model once ----------
model = PneumoniaCNN().to(device)
model.load_state_dict(
    torch.load(r"D:\D drive\Project\AI That Explains Medical Images Like a Doctor\model creation\pneumonia_cnn.pth", map_location=device, weights_only=True)
)
model.eval()

gradcam = GradCAM(model, model.features[6])

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# ---------- Main API ----------
def predict_xray(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    prob, bbox = predict(model, image_tensor)
    heatmap = gradcam.generate(image_tensor)

    explanation = build_explanation(
        image_name=image_path.split("/")[-1],
        prob=prob,
        bbox=bbox,
        heatmap=heatmap
    )

    return explanation
