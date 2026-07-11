def get_diabetes_recommendations(risk_score):

    if risk_score >= 80:
        return [
            "Consult a healthcare professional",
            "Monitor blood glucose regularly",
            "Reduce sugar intake",
            "Follow a structured exercise plan"
        ]

    elif risk_score >= 50:
        return [
            "Maintain a healthy diet",
            "Increase physical activity",
            "Monitor glucose periodically"
        ]

    return [
        "Continue healthy lifestyle habits",
        "Annual diabetes screening"
    ]