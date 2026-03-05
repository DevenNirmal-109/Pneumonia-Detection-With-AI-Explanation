def generate_medical_summary(result: dict) -> str:
    lines = []

    lines.append(f"Prediction: {result['prediction']}")
    lines.append(f"Confidence: {result['confidence']}")

    lines.append(f"Affected Lung: {result['affected_lung']}")
    lines.append(f"Opacity Detected: {result['opacity_detected']}")
    lines.append(f"Spread Type: {result['spread_type']}")

    bbox = result["bounding_box"]
    lines.append(
        f"Localized Region (bounding box): "
        f"x={bbox['x']}, y={bbox['y']}, "
        f"width={bbox['width']}, height={bbox['height']}"
    )

    if result["visual_patterns"]:
        lines.append("Observed Visual Patterns:")
        for p in result["visual_patterns"]:
            lines.append(f"- {p}")

    lines.append(
        "\nNote: This is an AI-assisted analysis and not a clinical diagnosis."
    )

    return "\n".join(lines)
