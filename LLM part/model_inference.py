import torch
from torchvision import transforms
from PIL import Image
from model_defiantion import PneumoniaCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PneumoniaCNN().to(device)
state_dict = torch.load(
    r"D:\D drive\Project\AI That Explains Medical Images Like a Doctor\model creation\pneumonia_cnn.pth",
    map_location=device,
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_xray(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        obj_logit, bbox = model(image)
        prob = torch.sigmoid(obj_logit).item()

    prediction = "Pneumonia" if prob >= 0.5 else "Normal"

    return {
        "prediction": prediction,
        "confidence": prob,
        "bbox": bbox.squeeze().cpu().numpy().tolist()
    }
