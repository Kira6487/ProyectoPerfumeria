import psycopg2

try:
    conexion = psycopg2.connect(
        host="localhost",
        database="PerfumeDB",
        user="postgres",
        password="admin"
    )
    print("Conexión exitosa a la base de datos")
    cursor = conexion.cursor()
    cursor.execute("SELECT datname FROM pg_database;")
    row = cursor.fetchone()
    print(row)
    cursor.execute("SELECT datname FROM pg_database;")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
except Exception as e:
    print("Error al conectar a la base de datos:", e)
finally:
    if 'conexion' in locals():
        conexion.close()
        print("Conexión cerrada")