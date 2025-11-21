"""
Aplicación Flask para clasificar fisuras de concreto.

Esta aplicación carga un modelo de machine learning previamente
entrenado, así como las columnas utilizadas para el one‑hot
encoding. Proporciona una interfaz web para introducir los
parámetros de la fisura y obtener una predicción sobre si la
fisura es plástica (helada temprana) o estructural (por compresión
simple). También expone una ruta de API para realizar
predicciones programáticas mediante JSON.

Requisitos:
  - El archivo modelo_fisuras.pkl debe contener un estimador
    compatible con scikit‑learn que implemente el método
    ``predict()`` y acepte un DataFrame.
  - El archivo columnas.pkl debe ser una lista o Índice de pandas
    con los nombres de columnas generadas durante el entrenamiento
    tras aplicar one‑hot encoding a los datos originales.
"""
import io
from datetime import datetime

import matplotlib.pyplot as plt
import shap
import qrcode
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

from pathlib import Path
import pickle
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import json

import matplotlib.pyplot as plt

from Prueba_Conexion import get_connection

fecha_str = datetime.now().strftime("%d%m%y_%H%M%S")

def cargar_modelo(ruta_modelo: str):
    """Carga el modelo entrenado desde un archivo pickle.

    Args:
        ruta_modelo: Ruta al archivo ``.pkl`` que contiene el modelo.

    Returns:
        Modelo deserializado.
    """
    modelo_path = Path(ruta_modelo)
    if not modelo_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de modelo en {ruta_modelo}")
    with modelo_path.open('rb') as f:
        modelo = pickle.load(f)
    return modelo


def cargar_columnas(ruta_columnas: str):
    """Carga la lista de columnas empleadas en el entrenamiento.

    Este archivo debe contener las columnas resultantes después de
    aplicar one‑hot encoding a los datos originales. Se utiliza para
    reindexar los datos de entrada de manera que coincidan con el
    esquema de entrenamiento.

    Args:
        ruta_columnas: Ruta al archivo ``.pkl`` que contiene las
            columnas.

    Returns:
        Una secuencia de nombres de columnas (list, pandas.Index, etc.).
    """
    columnas_path = Path(ruta_columnas)
    if not columnas_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de columnas en {ruta_columnas}")
    with columnas_path.open('rb') as f:
        columnas = pickle.load(f)
    return columnas


def construir_app():
    """Construye y configura la aplicación Flask.

    Carga el modelo y las columnas desde el disco y define las
    rutas necesarias para la interfaz web y la API de predicción.

    Returns:
        Objeto Flask configurado.
    """
    app = Flask(__name__)

    # Cargar modelo y columnas una sola vez al arrancar la aplicación
    ruta_base = Path(__file__).resolve().parent
    modelo = cargar_modelo(ruta_base / 'modelo_fisuras.pkl')
    columnas = cargar_columnas(ruta_base / 'columnas.pkl')
    
    explainer = shap.TreeExplainer(modelo)

    # Extraer las categorías de patron_fisura a partir de las columnas
    # Se identifican columnas que comienzan con el prefijo 'patron_fisura_'
    prefijo_categoria = 'patron_fisura_'
    categorias_patron = sorted({col[len(prefijo_categoria):] for col in columnas if col.startswith(prefijo_categoria)})

    @app.route('/', methods=['GET'])
    def index():
        """Muestra el formulario principal para introducir datos de la fisura."""
        return render_template('index.html', categorias=categorias_patron, valores={})

    def procesar_entrada(datos: dict) -> pd.DataFrame:
        """Convierte un diccionario de entrada en un DataFrame listo para el modelo.

        Realiza las siguientes operaciones:
          - Convierte los valores numéricos a floats/enteros apropiados.
          - Aplica one‑hot encoding a la columna ``patron_fisura``.
          - Reindexa el DataFrame para coincidir con las columnas del modelo,
            rellenando con cero las columnas que falten.

        Args:
            datos: Diccionario con claves correspondientes a los campos
                esperados por el modelo.

        Returns:
            DataFrame con una fila, listo para ser pasado al modelo.
        """
        # Convertir tipos numéricos
        entrada = {
            'profundidad_mm': float(datos.get('profundidad_mm', 0)),
            'longitud_cm': float(datos.get('longitud_cm', 0)),
            'temp_ambiente_C': float(datos.get('temp_ambiente_C', 0)),
            'humedad_relativa': float(datos.get('humedad_relativa', 0)),
            'columna_ensanchada': int(datos.get('columna_ensanchada', 0)),
            'edad_concreto_horas': float(datos.get('edad_concreto_horas', 0)),
            'exposicion_viento_kmh': float(datos.get('exposicion_viento_kmh', 0)),
            'patron_fisura': datos.get('patron_fisura', '')
        }

        # Crear DataFrame
        df = pd.DataFrame([entrada])

        # One‑hot encoding para el patrón de fisura
        df_encoded = pd.get_dummies(df, columns=['patron_fisura'])

        # Reindexar para incluir todas las columnas del entrenamiento
        df_encoded = df_encoded.reindex(columns=columnas, fill_value=0)

        return df_encoded
    
    def generar_grafica(datos):
        """Genera una gráfica de barras con los valores de la fisura y la devuelve como imagen."""
        fig, ax = plt.subplots(figsize=(5, 3))

        etiquetas = [
            "Profundidad (mm)",
            "Longitud (cm)",
            "Temp (°C)",
            "Humedad (%)",
            "Edad (hrs)",
            "Viento (km/h)"
        ]

        valores = [
            float(datos["profundidad_mm"]),
            float(datos["longitud_cm"]),
            float(datos["temp_ambiente_C"]),
            float(datos["humedad_relativa"]),
            float(datos["edad_concreto_horas"]),
            float(datos["exposicion_viento_kmh"])
        ]

        ax.bar(etiquetas, valores, color="#4A90E2")
        ax.set_ylabel("Valor")
        ax.set_title("Valores ingresados de la fisura")
        plt.xticks(rotation=30, ha="right")

        # Guardar en buffer
        grafica_buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(grafica_buf, format="PNG")
        grafica_buf.seek(0)
        plt.close()

        return grafica_buf

    
    def generar_reporte_pdf(datos: dict, df_modelo: pd.DataFrame, etiqueta: str):
        """Genera un PDF y devuelve un buffer BytesIO listo para enviar.

        El helper no es una ruta de Flask (no lleva decorador).
        Incluye datos de entrada, la predicción, explicación breve,
        fecha, logo (si existe), gráfica SHAP y código QR.
        """

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Fecha
        fecha_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # Logo (si existe static/logo.png)
        logo_path = ruta_base / "static" / "logo.png"
        y_actual = height - 60
        if logo_path.exists():
            try:
                c.drawImage(str(logo_path), 40, y_actual - 40, width=80, height=40, preserveAspectRatio=True, mask='auto')
            except Exception:
                pass

        # Título
        c.setFont("Helvetica-Bold", 16)
        c.drawString(140, y_actual - 10, "Reporte de clasificación de fisura")
        c.setFont("Helvetica", 10)
        c.drawString(140, y_actual - 25, f"Fecha y hora: {fecha_str}")

        # Datos de entrada
        y_actual -= 80
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y_actual, "Datos de entrada:")
        y_actual -= 15
        c.setFont("Helvetica", 10)

        for k, v in datos.items():
            c.drawString(50, y_actual, f"{k}: {v}")
            y_actual -= 12
            if y_actual < 200:  # evitar que se encime con las gráficas
                break

        # Predicción y explicación
        y_actual -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y_actual, "Predicción del modelo:")
        y_actual -= 15
        c.setFont("Helvetica", 11)
        c.drawString(50, y_actual, etiqueta)

        y_actual -= 25
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y_actual, "Explicación breve:")
        y_actual -= 15
        c.setFont("Helvetica", 9)

        if "HELADA" in etiqueta.upper():
            texto_exp = (
                "La fisura se clasifica como PLÁSTICA POR HELADA TEMPRANA. "
                "Los valores de temperatura baja, edad temprana del concreto y la ausencia de "
                "ensanchamiento en la columna indican un fenómeno de congelación."
            )
        else:
            texto_exp = (
                "La fisura se clasifica como ESTRUCTURAL POR COMPRESIÓN SIMPLE. "
                "La profundidad, longitud y la posible presencia de ensanchamiento en la columna "
                "sugieren un problema de carga y aplastamiento del concreto."
            )
        c.drawString(50, y_actual, texto_exp[:110])
        y_actual -= 12
        c.drawString(50, y_actual, texto_exp[110:220])
        
        # ===== Gráfica en el PDF =====
        y_actual -= 25
        grafica_buf = generar_grafica(datos)
        grafica_image = ImageReader(grafica_buf)
        
        c.drawString(40, y_actual, "Gráfica de datos ingresados:")
        y_actual -= 145
        c.drawImage(grafica_image, 40, y_actual, width=250, height=150, preserveAspectRatio=True, mask='auto')


        # Agregamos la confianza (porcentaje de probabilidad de la clase predicha)
        confianza = None
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(df_modelo)[0]
            pred_local = modelo.predict(df_modelo)[0]
            try:
                clases_local = list(getattr(modelo, "classes_", []))
                idx_local = clases_local.index(pred_local)
            except Exception:
                idx_local = int(pred_local) if isinstance(pred_local, (int, bool)) else 0
            confianza = round(proba[idx_local] * 100, 2)
        y_actual -= 18
        if confianza is not None:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y_actual, "Confianza del modelo:")
            y_actual -= 15
            c.setFont("Helvetica", 10)
            c.drawString(50, y_actual, f"{confianza}%")
        else:
            c.setFont("Helvetica", 10)
            c.drawString(40, y_actual, "Confianza del modelo no disponible (predict_proba ausente).")


        # ===== Gráfica SHAP =====
        # Calculamos SHAP para esta fila
        shap_values = explainer.shap_values(df_modelo)

        plt.figure(figsize=(4, 3))
        shap.summary_plot(shap_values, df_modelo, plot_type="bar", show=False)
        plt.tight_layout()

        shap_buf = io.BytesIO()
        plt.savefig(shap_buf, format="png")
        plt.close()
        shap_buf.seek(0)

        shap_img = ImageReader(shap_buf)
        c.drawString(40, 210, "Importancia de características (SHAP):")
        c.drawImage(shap_img, 40, 80, width=260, height=120, preserveAspectRatio=True, mask='auto')

        # ===== Código QR =====

        # Construir payload con datos ingresados y metadatos
        # Construir texto legible con los datos de la fisura
        campos_legibles = {
            "Profundidad (mm)": datos.get("profundidad_mm"),
            "Longitud (cm)": datos.get("longitud_cm"),
            "Temp ambiente (°C)": datos.get("temp_ambiente_C"),
            "Humedad relativa (%)": datos.get("humedad_relativa"),
            "Columna ensanchada (0/1)": datos.get("columna_ensanchada"),
            "Edad concreto (h)": datos.get("edad_concreto_horas"),
            "Exposición viento (km/h)": datos.get("exposicion_viento_kmh"),
            "Patrón fisura": datos.get("patron_fisura"),
        }

        lines = [
            "REPORTE FISURA",
            f"Fecha: {fecha_str}",
            f"Predicción: {etiqueta}",
            "----- Datos -----"
        ]
        for k, v in campos_legibles.items():
            lines.append(f"{k}: {v}")

        # También incluir versión JSON para posible parsing
        qr_payload = {
            "timestamp": fecha_str,
            "prediccion": etiqueta,
            "datos": campos_legibles
        }

        qr_texto = "\n".join(lines) + "\n----- JSON -----\n" + json.dumps(qr_payload, ensure_ascii=False)

        qr_img = qrcode.make(qr_texto)
        qr_buf = io.BytesIO()
        qr_img.save(qr_buf, format="PNG")
        qr_buf.seek(0)

        qr_image = ImageReader(qr_buf)
        c.drawString(340, 210, "QR de referencia:")
        c.drawImage(qr_image, 340, 90, width=140, height=140, preserveAspectRatio=True, mask='auto')

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    @app.route('/reporte_pdf', methods=['POST'])
    def reporte_pdf():
        """
        Genera y devuelve un PDF con los datos de la fisura y la predicción.
        Usa los mismos campos del formulario que /predecir.
        """
        datos = request.form.to_dict()

        try:
            df = procesar_entrada(datos)
            etiqueta = realizar_prediccion(df)
            pdf_buffer = generar_reporte_pdf(datos, df, etiqueta)
        except Exception as e:
            return f"Error al generar el reporte: {e}", 500

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"reporte_fisura_{fecha_str}.pdf",
            mimetype="application/pdf"
        )

    def realizar_prediccion(df: pd.DataFrame) -> str:
        """Realiza la predicción utilizando el modelo cargado y devuelve la etiqueta.

        La función asume que el modelo devuelve una etiqueta o clase. Si
        devuelve un valor numérico, se mapea a las etiquetas humanas.

        Args:
            df: DataFrame preprocesado.

        Returns:
            Etiqueta de predicción en lenguaje natural.
        """
        pred = modelo.predict(df)[0]

        if pred == 1:
            return "FÍSURA PLÁSTICA: HELADA TEMPRANA ❄️"
        elif pred == 0:
            return "FÍSURA ESTRUCTURAL: POR COMPRESIÓN SIMPLE 🧱"
        else:
            return f"Etiqueta desconocida: {pred}"

        # Mapear la predicción a etiquetas humanas
        # Si el modelo ya devuelve las etiquetas correctas, se usa tal cual
        # De lo contrario, se mapea 0 -> fisura plástica (helada temprana), 1 -> fisura estructural (compresión simple)
        """ if isinstance(pred, (int, float)):
            return 'FÍSURA PLÁSTICA: HELADA TEMPRANA' if pred == 0 else 'FÍSURA ESTRUCTURAL: POR COMPRESIÓN SIMPLE'
        elif isinstance(pred, str):
            # Convertir texto en las etiquetas solicitadas si es posible
            valor = pred.strip().lower()
            if 'plast' in valor or 'helada' in valor:
                return 'FÍSURA PLÁSTICA: HELADA TEMPRANA'
            elif 'estruct' in valor or 'compres' in valor:
                return 'FÍSURA ESTRUCTURAL: POR COMPRESIÓN SIMPLE'
            else:
                # En caso de etiqueta desconocida, se devuelve tal cual
                return pred
        else:
            return str(pred) """
    
    def obtener_confianza(df):
        """Devuelve el porcentaje de confianza de la clase predicha."""
        if not hasattr(modelo, "predict_proba"):
            return None
        proba = modelo.predict_proba(df)[0]
        pred = modelo.predict(df)[0]
        try:
            clases = list(getattr(modelo, "classes_", []))
            idx = clases.index(pred)
        except Exception:
            # Asumir orden estándar si no se puede mapear
            idx = int(pred) if isinstance(pred, (int, bool)) else 0
        confianza = proba[idx]
        return round(confianza * 100, 2)
    
    @app.route('/historial', methods=['GET'])
    def historial():
        """Muestra una tabla HTML con todos los registros guardados en SQL Server."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    id, fecha_generacion, 
                    profundidad_mm, longitud_cm, temp_ambiente_C,
                    humedad_relativa, patron_fisura, columna_ensanchada,
                    edad_concreto_horas, exposicion_viento_kmh,
                    resultado, confianza, ruta_pdf
                FROM Reportes
                ORDER BY id DESC
            """)

            registros = cursor.fetchall()
            columnas = [column[0] for column in cursor.description]

            cursor.close()
            conn.close()

        except Exception as e:
            return f"Error al obtener historial: {e}"

        return render_template("historial.html", columnas=columnas, registros=registros)

    @app.route('/predecir', methods=['POST'])
    def predecir():
        """Procesa el formulario, obtiene predicción, calcula confianza y guarda en SQL Server."""
        datos = request.form.to_dict()
        confianza = None

        try:
            # Procesar entrada y obtener predicción
            df = procesar_entrada(datos)
            resultado = realizar_prediccion(df)
            confianza = obtener_confianza(df)

            # === GUARDAR EN SQL SERVER ===
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO Reportes (
                    profundidad_mm, longitud_cm, temp_ambiente_C,
                    humedad_relativa, patron_fisura, columna_ensanchada,
                    edad_concreto_horas, exposicion_viento_kmh,
                    resultado, confianza
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                float(datos["profundidad_mm"]),
                float(datos["longitud_cm"]),
                float(datos["temp_ambiente_C"]),
                float(datos["humedad_relativa"]),
                str(datos["patron_fisura"]),
                int(datos["columna_ensanchada"]),
                float(datos["edad_concreto_horas"]),
                float(datos["exposicion_viento_kmh"]),
                str(resultado),
                float(confianza)
            ))


            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            return f"Error guardando en SQL Server o procesando predicción: {e}"

        # Render normal si todo salió bien
        return render_template(
            'index.html',
            categorias=categorias_patron,
            resultado=resultado,
            confianza=confianza,
            valores=datos
        )

    @app.route('/api/prediccion', methods=['POST'])
    def api_prediccion():
        """Endpoint de API que recibe un JSON con datos y devuelve la predicción en JSON."""
        if not request.is_json:
            return jsonify({'error': 'La solicitud debe ser JSON'}), 400
        datos = request.get_json()
        try:
            df = procesar_entrada(datos)
            resultado = realizar_prediccion(df)
            confianza = obtener_confianza(df)
            return
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        return jsonify({'prediccion': resultado})

    return app


if __name__ == '__main__':
    # Ejecutar la aplicación en modo local. No habilitar debug en producción.
    aplicacion = construir_app()
    aplicacion.run(host='0.0.0.0', port=5000, debug=False)
    