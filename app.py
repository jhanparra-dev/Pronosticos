from flask import Flask, render_template, request, jsonify 
import pandas as pd
import numpy as np
import math
from io import StringIO #Importa StringIO para manejar archivos en memoria
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


def sanitize_for_json(value):
    """Convierte valores no JSON-serializables (NaN/Inf / tipos numpy) a tipos JSON válidos."""
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}  # Recursivamente sanitiza diccionarios
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]  # Recursivamente sanitiza listas

    # Numpy y pandas pueden generar tipos no nativos que JSON no acepta (NaN, Inf, np.float64, etc.)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    # Si llega un numpy array, conviértelo a lista
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())

    return value

def calcular_pronostico(serie, N, horizonte=0):
    """Calcula pronóstico y métricas de error con la metodología de media móvil.

    El pronóstico se calcula para los datos disponibles usando una media móvil simple
    con ventana N (desplazada 1 periodo). Luego se extiende de forma iterativa
    hasta el horizonte solicitado, usando los pronósticos previos como si fueran datos.
    """

    # Valores históricos (ventas reales)
    valores = [float(x) for x in serie]
    horizonte = max(0, int(horizonte))
    total_len = len(valores) + horizonte

    # Pronóstico (incluye proyecciones futuras)
    pronostico = [np.nan] * total_len # Inicializar con NaN para todo el rango (incluyendo futuros)

    for i in range(total_len):
        if i < N:
            continue
        # Construir la ventana de N valores (puede incluir pronósticos previos cuando i >= len(valores))
        window_values = [] # Lista para almacenar los valores de la ventana actual
        for j in range(i - N, i): # Desde i-N hasta i-1
            if j < len(valores):
                window_values.append(valores[j]) # Usar valor real si está dentro de los datos históricos
            else:
                window_values.append(pronostico[j]) # Usar pronóstico previo si estamos en la parte de proyección futura

        if len(window_values) == N and all(pd.notna(v) for v in window_values): #   Asegurarse de que la ventana esté completa y no tenga NaN
            pronostico[i] = sum(window_values) / N #  Calcular el pronóstico como el promedio de la ventana actual

    # Construir DataFrame con ventas (NaN para los periodos de pronóstico futuro)
    df = pd.DataFrame({
        'ventas': valores + [np.nan] * horizonte,
        'Pronostico': pronostico
    })

    # Calcular errores solo para los periodos en los que existen datos reales
    df['error'] = df['Pronostico'] - df['ventas']
    df['error_abs'] = df['error'].abs()
    df['ape'] = df['error_abs'] / df['ventas'].replace(0, np.nan)
    df['ape´'] = df['error_abs'] / df['Pronostico'].replace(0, np.nan)
    df['error_cuadrado'] = df['error'] * df['error']

    # Métricas de error (ignorando NaN)
    MAPE = df['ape'].mean()
    MAPE_prima = df['ape´'].mean()
    MSE = df['error_cuadrado'].mean()
    RMSE = np.sqrt(MSE)

    # Reemplazar NaN por 0 en métricas
    MAPE = 0 if pd.isna(MAPE) else MAPE
    MAPE_prima = 0 if pd.isna(MAPE_prima) else MAPE_prima
    MSE = 0 if pd.isna(MSE) else MSE
    RMSE = 0 if pd.isna(RMSE) else RMSE

    return df, {'MAPE': MAPE, 'MAPE_prima': MAPE_prima, 'MSE': MSE, 'RMSE': RMSE}

def calcular_exponential_smoothing(serie, horizonte=0):
    """Calcula pronóstico con Suavización Exponencial con Tendencia."""
    serie = pd.Series(serie)
    # ExponentialSmoothing con trend='add' para que el pronóstico tenga variación
    model = ExponentialSmoothing(serie, trend='add', seasonal=None).fit(optimized=True)
    pronostico = model.fittedvalues.tolist() + model.forecast(horizonte).tolist()
    
    df = pd.DataFrame({
        'ventas': serie.tolist() + [np.nan] * horizonte,
        'Pronostico': pronostico
    })
    
    df['error'] = df['Pronostico'] - df['ventas']
    df['error_abs'] = df['error'].abs()
    df['ape'] = df['error_abs'] / df['ventas'].replace(0, np.nan)
    df['ape´'] = df['error_abs'] / df['Pronostico'].replace(0, np.nan)
    df['error_cuadrado'] = df['error'] * df['error']
    
    MAPE = df['ape'].mean()
    MAPE_prima = df['ape´'].mean()
    MSE = df['error_cuadrado'].mean()
    RMSE = np.sqrt(MSE)
    
    MAPE = 0 if pd.isna(MAPE) else MAPE
    MAPE_prima = 0 if pd.isna(MAPE_prima) else MAPE_prima
    MSE = 0 if pd.isna(MSE) else MSE
    RMSE = 0 if pd.isna(RMSE) else RMSE
    
    return df, {'MAPE': MAPE, 'MAPE_prima': MAPE_prima, 'MSE': MSE, 'RMSE': RMSE}

def calcular_prophet(serie, horizonte=0):
    """Calcula pronóstico con Prophet (optimizado)."""
    # Asumir periodos mensuales desde 2020-01
    fechas = pd.date_range(start='2020-01-01', periods=len(serie), freq='ME')
    df_train = pd.DataFrame({'ds': fechas, 'y': serie})
    
    # Prophet optimizado: sin estacionalidad para datos pequeños
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
    model.fit(df_train)
    
    future = model.make_future_dataframe(periods=horizonte, freq='ME')
    forecast = model.predict(future)
    
    pronostico = forecast['yhat'].tolist()
    
    df = pd.DataFrame({
        'ventas': serie + [np.nan] * horizonte,
        'Pronostico': pronostico
    })
    
    df['error'] = df['Pronostico'] - df['ventas']
    df['error_abs'] = df['error'].abs()
    df['ape'] = df['error_abs'] / df['ventas'].replace(0, np.nan)
    df['ape´'] = df['error_abs'] / df['Pronostico'].replace(0, np.nan)
    df['error_cuadrado'] = df['error'] * df['error']
    
    MAPE = df['ape'].mean()
    MAPE_prima = df['ape´'].mean()
    MSE = df['error_cuadrado'].mean()
    RMSE = np.sqrt(MSE)
    
    MAPE = 0 if pd.isna(MAPE) else MAPE
    MAPE_prima = 0 if pd.isna(MAPE_prima) else MAPE_prima
    MSE = 0 if pd.isna(MSE) else MSE
    RMSE = 0 if pd.isna(RMSE) else RMSE
    
    return df, {'MAPE': MAPE, 'MAPE_prima': MAPE_prima, 'MSE': MSE, 'RMSE': RMSE}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calcular', methods=['POST'])
def calcular():
    try:
        file = request.files['archivo']
        ventana = int(request.form.get('ventana', 3))
        horizonte = int(request.form.get('horizonte', 0))
        horizonte = max(0, horizonte)

        # Leer CSV
        datos = pd.read_csv(file)
        
        if datos.empty:
            return jsonify({'error': 'Archivo vacío'}), 400
        
        resultados = {}
        
        # Procesar cada columna de producto
        for columna in datos.columns:
            # Ignorar la columna del índice/periodo si existe
            if str(columna).strip().lower() == 'periodo':
                continue

            if pd.api.types.is_numeric_dtype(datos[columna]):
                serie = datos[columna].dropna().tolist()
                if len(serie) > ventana:
                    # Media Móvil
                    df_mm, metricas_mm = calcular_pronostico(serie, ventana, horizonte)
                    
                    # Exponential Smoothing
                    df_es, metricas_es = calcular_exponential_smoothing(serie, horizonte)
                    
                    # Prophet
                    df_prophet, metricas_prophet = calcular_prophet(serie, horizonte)
                    
                    resultados[columna] = {
                        'media_movil': {
                            'metricas': {k: round(float(v) if pd.notna(v) and not np.isinf(v) else 0, 2) for k, v in metricas_mm.items()},
                            'datos': df_mm.where(pd.notna(df_mm), None).to_dict(orient='list'),
                            'grafico': {
                                'valores': [float(x) if pd.notna(x) else None for x in df_mm['ventas']],
                                'pronostico': [float(x) if pd.notna(x) else None for x in df_mm['Pronostico']],
                                'periodos': list(range(len(df_mm)))
                            }
                        },
                        'exponential_smoothing': {
                            'metricas': {k: round(float(v) if pd.notna(v) and not np.isinf(v) else 0, 2) for k, v in metricas_es.items()},
                            'datos': df_es.where(pd.notna(df_es), None).to_dict(orient='list'),
                            'grafico': {
                                'valores': [float(x) if pd.notna(x) else None for x in df_es['ventas']],
                                'pronostico': [float(x) if pd.notna(x) else None for x in df_es['Pronostico']],
                                'periodos': list(range(len(df_es)))
                            }
                        },
                        'prophet': {
                            'metricas': {k: round(float(v) if pd.notna(v) and not np.isinf(v) else 0, 2) for k, v in metricas_prophet.items()},
                            'datos': df_prophet.where(pd.notna(df_prophet), None).to_dict(orient='list'),
                            'grafico': {
                                'valores': [float(x) if pd.notna(x) else None for x in df_prophet['ventas']],
                                'pronostico': [float(x) if pd.notna(x) else None for x in df_prophet['Pronostico']],
                                'periodos': list(range(len(df_prophet)))
                            }
                        }
                    }
        
        if not resultados:
            return jsonify({'error': 'No hay columnas numéricas válidas'}), 400
        
        response = {
            'ventana': ventana,
            'productos': resultados
        }
        return jsonify(sanitize_for_json(response))
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
