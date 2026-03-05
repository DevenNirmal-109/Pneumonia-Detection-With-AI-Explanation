import torch

def predict(model, image_tensor):
    obj_logit, bbox = model(image_tensor)
    prob = torch.sigmoid(obj_logit).item()
    bbox = bbox.squeeze(0).detach().cpu().numpy()
    return prob, bbox
