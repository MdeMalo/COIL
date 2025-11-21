"""
Script para inicializar una base de datos SQLite de ejemplo para desarrollo.
Crea `data/sample.db` y una tabla `Reportes` con el esquema usado por la aplicación,
además inserta algunas filas de ejemplo para poder probar el dashboard y el historial.

Ejecución (PowerShell):
    python scripts\init_db.py

"""
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
DB_PATH = DATA_DIR / 'sample.db'

DATA_DIR.mkdir(parents=True, exist_ok=True)

schema = """
CREATE TABLE IF NOT EXISTS Reportes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_generacion TEXT DEFAULT (datetime('now')),
    profundidad_mm REAL,
    longitud_cm REAL,
    temp_ambiente_C REAL,
    humedad_relativa REAL,
    patron_fisura TEXT,
    columna_ensanchada INTEGER,
    edad_concreto_horas REAL,
    exposicion_viento_kmh REAL,
    resultado TEXT,
    confianza REAL,
    ruta_pdf TEXT
);
"""

sample_rows = [
    (12.0, 45.0, 5.0, 60.0, 'cruz', 0, 24.0, 12.0, 'FÍSURA PLÁSTICA: HELADA TEMPRANA ❄️', 82.5, None),
    (38.5, 120.0, 18.0, 55.0, 'longitudinal', 1, 240.0, 6.0, 'FÍSURA ESTRUCTURAL: POR COMPRESIÓN SIMPLE 🧱', 74.2, None),
    (8.0, 22.0, 3.0, 80.0, 'stellate', 0, 12.0, 20.0, 'FÍSURA PLÁSTICA: HELADA TEMPRANA ❄️', 66.1, None),
    (25.0, 90.0, 22.0, 40.0, 'transversal', 1, 720.0, 3.0, 'FÍSURA ESTRUCTURAL: POR COMPRESIÓN SIMPLE 🧱', 88.0, None),
]

print(f"Creando base de datos de ejemplo en: {DB_PATH}")
conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

cur.executescript(schema)

# Insert sample rows
insert_sql = """
INSERT INTO Reportes (
    profundidad_mm, longitud_cm, temp_ambiente_C, humedad_relativa,
    patron_fisura, columna_ensanchada, edad_concreto_horas, exposicion_viento_kmh,
    resultado, confianza, ruta_pdf
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

for row in sample_rows:
    cur.execute(insert_sql, row)

conn.commit()
conn.close()
print("Base de datos inicializada con filas de ejemplo.")
print("Ruta del archivo DB:", DB_PATH)
print("Para usarla localmente, configura tu helper de conexión para apuntar a este archivo SQLite.")
