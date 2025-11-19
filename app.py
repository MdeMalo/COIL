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

from pathlib import Path
import pickle
from flask import Flask, render_template, request, jsonify
import pandas as pd


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

    # Extraer las categorías de patron_fisura a partir de las columnas
    # Se identifican columnas que comienzan con el prefijo 'patron_fisura_'
    prefijo_categoria = 'patron_fisura_'
    categorias_patron = sorted({col[len(prefijo_categoria):] for col in columnas if col.startswith(prefijo_categoria)})

    @app.route('/', methods=['GET'])
    def index():
        """Muestra el formulario principal para introducir datos de la fisura."""
        return render_template('index.html', categorias=categorias_patron)

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

    @app.route('/predecir', methods=['POST'])
    def predecir():
        """Procesa el formulario y muestra el resultado de la predicción."""
        datos = request.form.to_dict()
        try:
            df = procesar_entrada(datos)
            resultado = realizar_prediccion(df)
        except Exception as e:
            resultado = f"Error al procesar la predicción: {e}"
        return render_template('index.html', categorias=categorias_patron, resultado=resultado, valores=datos)

    @app.route('/api/prediccion', methods=['POST'])
    def api_prediccion():
        """Endpoint de API que recibe un JSON con datos y devuelve la predicción en JSON."""
        if not request.is_json:
            return jsonify({'error': 'La solicitud debe ser JSON'}), 400
        datos = request.get_json()
        try:
            df = procesar_entrada(datos)
            resultado = realizar_prediccion(df)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        return jsonify({'prediccion': resultado})

    return app


if __name__ == '__main__':
    # Ejecutar la aplicación en modo local. No habilitar debug en producción.
    aplicacion = construir_app()
    aplicacion.run(host='0.0.0.0', port=5000, debug=False)
    
print(cargar_columnas("columnas.pkl"))