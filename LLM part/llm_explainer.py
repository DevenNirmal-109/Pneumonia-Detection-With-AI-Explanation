import ollama

def generate_explanation(medical_summary):
    prompt = f"""
You are a medical AI assistant specialized in chest X-ray analysis.

Explain the findings below in simple, calm, doctor-like language
that a patient can understand.

Rules:
- Do NOT give absolute diagnosis
- Mention confidence and affected region
- Avoid alarming language
- Add a medical disclaimer

Findings:
{medical_summary}
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]
