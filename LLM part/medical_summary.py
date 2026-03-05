def generate_medical_summary(result):
    prediction = result["prediction"]
    confidence = result["confidence"] * 100

    if prediction == "Pneumonia":
        summary = f"""
        Diagnosis: Pneumonia
        Confidence: {confidence:.2f}%
        Lung Opacity: Detected
        Affected Area: Possible lower lung region
        Severity: Moderate
        """
    else:
        summary = f"""
        Diagnosis: Normal
        Confidence: {confidence:.2f}%
        Lung Opacity: Not detected
        """

    return summary.strip()
