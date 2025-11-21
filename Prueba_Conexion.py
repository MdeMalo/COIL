import pyodbc

def get_connection():
    try:
        return pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=localhost;'
            'DATABASE=FisurasDB;'
            'Trusted_Connection=yes;'
        )
        
    except pyodbc.Error as e:
        print(f"Error al conectar a la base de datos: {e}")
        raise

if __name__ == "__main__":
    try:
        conn = get_connection()
        print("Conexión exitosa a la base de datos.")
        conn.close()
    except Exception as e:
        print(f"No se pudo establecer la conexión: {e}")