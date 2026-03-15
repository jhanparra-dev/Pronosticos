from flask import Flask, render_template, request, jsonify 
import pandas as pd
import numpy as np
from io import StringIO #Importa StringIO para manejar archivos en memoria

app = Flask(__name__)

def calcular_pronostico(serie, N):
    """Calcula pronóstico y métricas de error con las fórmulas especificadas"""
    df = pd.DataFrame({'ventas': serie})
    
    # Pronóstico con media móvil
    df["Pronostico"] = df["ventas"].rolling(window=N).mean().shift(1).round(0)
    
    # Errores
    df["error"] = df["Pronostico"] - df["ventas"]
    df["error_abs"] = df["error"].abs()
    df["ape"] = df["error_abs"] / df["ventas"].replace(0, np.nan)
    df["ape´"] = df["error_abs"] / df["Pronostico"].replace(0, np.nan)
    df["error_cuadrado"] = df["error"] * df["error"]
    
    # Métricas de error (ignorando NaN)
    MAPE = df["ape"].mean()
    MAPE_prima = df["ape´"].mean()
    MSE = df["error_cuadrado"].mean()
    RMSE = np.sqrt(MSE)
    
    # Reemplazar NaN por 0 en métricas
    MAPE = 0 if pd.isna(MAPE) else MAPE
    MAPE_prima = 0 if pd.isna(MAPE_prima) else MAPE_prima
    MSE = 0 if pd.isna(MSE) else MSE
    RMSE = 0 if pd.isna(RMSE) else RMSE
    
    return df.fillna(0), {'MAPE': MAPE, 'MAPE_prima': MAPE_prima, 'MSE': MSE, 'RMSE': RMSE}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calcular', methods=['POST'])
def calcular():
    try:
        file = request.files['archivo']
        ventana = int(request.form.get('ventana', 3))
        
        # Leer CSV
        datos = pd.read_csv(file)
        
        if datos.empty:
            return jsonify({'error': 'Archivo vacío'}), 400
        
        resultados = {}
        
        # Procesar cada columna de producto
        for columna in datos.columns:
            if pd.api.types.is_numeric_dtype(datos[columna]):
                serie = datos[columna].dropna().tolist()
                if len(serie) > ventana:
                    df, metricas = calcular_pronostico(serie, ventana)
                    
                    resultados[columna] = {
                        'metricas': {k: round(float(v) if pd.notna(v) and not np.isinf(v) else 0, 2) for k, v in metricas.items()},
                        'datos': df.to_dict(orient='list'),
                        'grafico': {
                            'valores': [float(x) if pd.notna(x) else None for x in df['ventas']],
                            'pronostico': [float(x) if pd.notna(x) else None for x in df['Pronostico']],
                            'periodos': list(range(len(df)))
                        }
                    }
        
        if not resultados:
            return jsonify({'error': 'No hay columnas numéricas válidas'}), 400
        
        return jsonify({
            'ventana': ventana,
            'productos': resultados
        })
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
