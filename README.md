# Flask Fisuras — COIL

Proyecto Flask para predicción y reporte de fisuras en concreto.

**Contenido**
- `app.py` — aplicación Flask principal (rutas: predict, historial, dashboard, export).
- `templates/` — plantillas Jinja2 (`index.html`, `historial.html`, `dashboard.html`, `reporte_detalle.html`, ...).
- `static/` — CSS y recursos (p. ej. `styles.css`, `logo.png`).
- Modelos serializados: `modelo_fisuras.pkl`, `columnas.pkl` (deben estar en la raíz o ruta esperada por `app.py`).

## Requisitos
- Python 3.9+ (recomendado 3.10+)
- Paquetes Python (sugerencia):
  - Flask
  - pandas
  - numpy
  - pyodbc
  - reportlab
  - qrcode
  - matplotlib
  - openpyxl / xlsxwriter (para exportar Excel)
  - shap (opcional, usado para explicación en PDFs)

Puedes crear un `venv` y luego instalar:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install flask pandas numpy pyodbc reportlab qrcode matplotlib xlsxwriter shap openpyxl
```

Nota: en Windows, `pyodbc` puede requerir instalar el driver ODBC de SQL Server (Microsoft ODBC Driver for SQL Server). Instálalo desde el sitio oficial si falta.

## Variables y conexiones
- La conexión a la base de datos SQL Server la gestiona `Prueba_Conexion.get_connection()` en tu código. Asegúrate de que esa función esté configurada con el DSN/host/usuario/contraseña correctos.
- Archivos del modelo: `modelo_fisuras.pkl` y `columnas.pkl` deben existir y ser legibles por el proceso (mismo formato `pickle`).

## Base de datos de ejemplo (SQLite)
Si quieres incluir una base de datos de ejemplo en el repositorio (útil para demos o pruebas locales), hay dos enfoques recomendados:

- 1) Script de inicialización (recomendado): se incluye un script `scripts/init_db.py` que crea `data/sample.db` con la tabla `Reportes` y filas de ejemplo. Ejecuta:

```powershell
python scripts\init_db.py
```

Después de ejecutar, el archivo `data/sample.db` contendrá datos de ejemplo. Para usarlo en desarrollo puedes apuntar temporalmente la aplicación al helper SQLite.

- 2) Exportar un dump SQL / archivo .db: puedes añadir el archivo binario `data/sample.db` al repo si quieres compartir directamente la base de datos. Ten en cuenta que no es recomendable subir bases de datos de producción o con datos sensibles.

### Usar el helper SQLite local
Se incluye `Prueba_Conexion_sqlite.py` que exporta `get_connection()` apuntando a `data/sample.db`. Para desarrollar con la base SQLite, cambia en `app.py` la línea de import (temporalmente):

```python
# from Prueba_Conexion import get_connection
from Prueba_Conexion_sqlite import get_connection
```

Esto evita tocar tu configuración de SQL Server y permite que `/historial`, `/dashboard` y `/export_excel` usen la DB de ejemplo.

> Seguridad: no subas credenciales ni bases de datos con datos reales al repositorio público. Usa datos sintéticos o anonimiza antes de compartir.

## Ejecutar la aplicación (desarrollo)
En PowerShell (desde la carpeta del proyecto):

```powershell
# activar entorno si usas venv
.\.venv\Scripts\Activate.ps1
# variables de entorno (opcional)
$env:FLASK_APP = 'app.py'
$env:FLASK_ENV = 'development'
# arrancar Flask
python -m flask run
# o alternativamente
python app.py
```

Abre en el navegador: `http://127.0.0.1:5000/`

## Endpoints principales
- `/` — formulario principal para ingresar datos y generar predicción / PDF.
- `/predecir` — POST que procesa la predicción y guarda en la base de datos.
- `/reporte_pdf` — POST que genera/descarga el PDF del reporte (usa ReportLab + QR).
- `/historial` — vista con registros guardados (filtros, paginación, borrar, exportar).
- `/export_excel` — genera un archivo Excel con los registros filtrados.
- `/dashboard` — dashboard interactivo; usa `/api/dashboard_data` para datos JSON.
- `/api/dashboard_data` — devuelve JSON con `conteos`, `promedios` y `serie` para las gráficas.
- `/reporte/<id>` — vista detallada de un registro individual.

## Generar un PDF de prueba
1. Rellena el formulario en `/` y usa el botón "Generar reporte (PDF)".
2. El servidor prepara un PDF con la predicción, gráfico y un QR; el navegador descargará o abrirá el fichero.

Si no funciona, revisa los logs del servidor para ver excepciones en `generar_reporte_pdf` o problemas con `shap`/`reportlab`.

## Exportar Excel
En la vista `/historial` hay un botón "Exportar" que descarga un `.xlsx` con los registros que cumplan los filtros aplicados.

## Dashboard
Visita `/dashboard`. Si las gráficas no muestran datos:
- Comprueba que `/api/dashboard_data` devuelva JSON (puedes abrirla directamente en el navegador).
- Asegúrate de que la base de datos contiene filas en la tabla usada para historial.

## Problemas comunes
- Error de conexión a SQL Server: revisa credenciales, firewall, y el driver ODBC.
- Dependencias faltantes: instala las librerías listadas anteriormente.
- `shap` puede fallar si no está instalado o si la versión no es compatible; es opcional para la explicación en PDFs.

## Mantener dependencias
Genera un `requirements.txt` con:

```powershell
pip freeze > requirements.txt
```

## Contribuir / Siguientes pasos sugeridos
- Añadir tests unitarios para rutas críticas y para la generación de PDFs.
- Asegurar manejo de errores más amigable en frontend (banners en vez de alertas).
- Añadir autenticación si piensas exponer la app a red pública.

---
Si quieres que deje un `requirements.txt` con las dependencias detectadas del proyecto, o que añada instrucciones más detalladas para la conexión a SQL Server (ejemplos de cadena de conexión), dímelo y lo agrego ahora.