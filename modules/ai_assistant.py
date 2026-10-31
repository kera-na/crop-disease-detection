import google.generativeai as genai


def ask_farmer_assistant(
        api_key,
        disease,
        confidence,
        question):

    genai.configure(
        api_key=api_key
    )

    models_to_try = [

        "gemini-3.1-flash-lite",

        "gemini-1.5-pro",

        "gemini-pro"
    ]

    last_error = None

    for model_name in models_to_try:

        try:

            model = genai.GenerativeModel(
                model_name
            )

            prompt = f"""
You are an agricultural expert.

Disease:
{disease}

Model Confidence:
{confidence*100:.2f}%

Farmer Question:
{question}

Provide:

1. Disease explanation
2. Cause
3. Treatment
4. Prevention
5. Organic alternatives
6. Farmer recommendations

Keep the answer practical and easy to understand.
"""

            response = model.generate_content(
                prompt
            )

            return response.text

        except Exception as e:

            last_error = e

            continue

    return f"AI Assistant Error: {last_error}"