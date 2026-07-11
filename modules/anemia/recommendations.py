def get_anemia_recommendations(risk_score):

    recommendations = []

    if risk_score >= 80:
        recommendations.extend([
            "Consult a physician immediately",
            "Perform complete blood count test",
            "Increase iron-rich foods"
        ])

    elif risk_score >= 50:
        recommendations.extend([
            "Monitor hemoglobin levels",
            "Consume spinach and dates",
            "Increase dietary iron intake"
        ])

    else:
        recommendations.extend([
            "Maintain balanced nutrition",
            "Continue regular health checkups"
        ])

    return recommendations