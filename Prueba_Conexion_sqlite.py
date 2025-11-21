"""
Helper de conexión SQLite para desarrollo local.
Exporta `get_connection()` que devuelve un objeto compatible con DB-API
(similar a pyodbc connections used in the project).

Uso: en desarrollo puedes renombrar temporalmente tu `Prueba_Conexion.py`
original (o importar este módulo desde `app.py`) para apuntar a la DB de ejemplo.
"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'data' / 'sample.db'

# A small wrapper to provide cursor() with similar behavior
def get_connection():
    # sqlite3 connection; note: param style is '?'
    conn = sqlite3.connect(str(DB_PATH))
    # Make returned rows accessible like tuples; user code uses cursor.fetchall() and cursor.description
    conn.row_factory = None
    return conn

if __name__ == '__main__':
    print('DB path:', DB_PATH)
    if not DB_PATH.exists():
        print('Base de datos no encontrada. Ejecuta scripts/init_db.py para generarla.')
    else:
        print('Base de datos lista para usar.')
