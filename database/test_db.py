from db import connection

cursor = connection.cursor()

cursor.execute("""
INSERT INTO predictions
(patient_name, disease, prediction, risk_score)
VALUES (?, ?, ?, ?)
""",
("John", "Diabetes", 1, 91.3)
)

connection.commit()

print("✅ Record inserted successfully!")