import numpy as np
import cv2

def detect_affected_lung(heatmap):
    h, w = heatmap.shape
    left = heatmap[:, :w//2].mean()
    right = heatmap[:, w//2:].mean()

    if abs(left - right) < 0.05:
        return "Both lungs"
    return "Right lung" if left > right else "Left lung"


def detect_opacity(heatmap, threshold=0.4):
    ratio = (heatmap > threshold).sum() / heatmap.size
    return bool(ratio > 0.05)


def detect_spread(heatmap, threshold=0.4):
    binary = np.uint8(heatmap > threshold)
    components, _ = cv2.connectedComponents(binary)
    return "Localized" if components <= 3 else "Diffuse"


def build_explanation(image_name, prob, bbox, heatmap):
    explanation = {
        "image_name": str(image_name),
        "prediction": "PNEUMONIA" if prob >= 0.5 else "NO_PNEUMONIA",
        "confidence": float(round(prob, 3)),
        "affected_lung": detect_affected_lung(heatmap),
        "opacity_detected": detect_opacity(heatmap),
        "spread_type": detect_spread(heatmap),
        "bounding_box": {
            "x": float(bbox[0]),
            "y": float(bbox[1]),
            "width": float(bbox[2]),
            "height": float(bbox[3])
        },
        "visual_patterns": [],
        "disclaimer": (
            "This AI output is for educational purposes only "
            "and must not be used as a medical diagnosis."
        )
    }

    if explanation["opacity_detected"]:
        explanation["visual_patterns"].extend([
            "increased opacity",
            "loss of normal lung texture"
        ])

    explanation["visual_patterns"].append(
        "localized lung involvement"
        if explanation["spread_type"] == "Localized"
        else "diffuse lung involvement"
    )

    return explanation
