from db import connection

cursor = connection.cursor()

cursor.execute("SELECT * FROM predictions")

rows = cursor.fetchall()

print(f"Total Records: {len(rows)}")

for row in rows:
    print(row)