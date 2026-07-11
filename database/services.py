from database.db import get_connection

def save_prediction(patient_name, age, disease, prediction, risk_score):
    print("🚀 save_prediction() called")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (patient_name, age, disease, prediction, risk_score)
        VALUES (?, ?, ?, ?, ?)
    """, (
        patient_name,
        age,
        disease,
        prediction,
        risk_score
    ))

    conn.commit()

    print("✅ Rows inserted:", cursor.rowcount)

    conn.close()