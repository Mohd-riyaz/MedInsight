def get_heart_recommendations(risk_score):

    if risk_score >= 80:
        return [
            "Consult a cardiologist",
            "Monitor blood pressure",
            "Reduce sodium intake",
            "Increase physical activity"
        ]

    elif risk_score >= 50:
        return [
            "Follow a heart-healthy diet",
            "Exercise regularly",
            "Schedule routine cardiac checkups"
        ]

    return [
        "Maintain healthy lifestyle habits",
        "Annual heart screening"
    ]