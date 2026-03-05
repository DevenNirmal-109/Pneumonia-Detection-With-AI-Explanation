def bbox_region_description(bbox):
    x, y, w, h = bbox
    area = w * h

    if area < 0.05:
        return "small localized lung region"
    elif area < 0.15:
        return "moderate-sized lung region"
    else:
        return "large lung region"


def severity_from_confidence(conf):
    if conf < 0.6:
        return "Mild"
    elif conf < 0.8:
        return "Moderate"
    else:
        return "Severe"


def generate_medical_summary(result):
    pred = result["prediction"]
    conf = result["confidence"]
    bbox = result["bbox"]

    if pred == "Pneumonia":
        region = bbox_region_description(bbox)
        severity = severity_from_confidence(conf)

        summary = (
            f"Condition: Pneumonia suspected\n"
            f"Confidence: {conf*100:.1f}%\n"
            f"Lung opacity: Detected\n"
            f"Affected region: {region}\n"
            f"Severity assessment: {severity}"
        )
    else:
        summary = (
            f"Condition: Normal chest X-ray\n"
            f"Confidence: {(1-conf)*100:.1f}%\n"
            f"Lung opacity: Not detected"
        )

    return summary
