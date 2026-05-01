import os

import psycopg2


def obtener_conexion():
    return psycopg2.connect(
        host=os.getenv("PERFUME_DB_HOST", "localhost"),
        port=os.getenv("PERFUME_DB_PORT", "5432"),
        database=os.getenv("PERFUME_DB_NAME", "PerfumeDB"),
        user=os.getenv("PERFUME_DB_USER", "postgres"),
        password=os.getenv("PERFUME_DB_PASSWORD", "admin"),
    )
