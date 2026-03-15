import pandas as pd

#FUNCIÓN PARA EJECUTAR UN PRONÓSTICO

def pronosticar(datos, N):
  """
  Calcula pronóstico de ventas usando media móvil
  
  Parámetros:
    datos: DataFrame con columna 'ventas'
    N: ventana para la media móvil
    
  Retorna:
    datos: DataFrame con pronósticos y errores
    MAPE: Mean Absolute Percentage Error
    MAPE_prima: Media de errores absolutos sobre pronósticos
    MSE: Mean Squared Error
    RMSE: Root Mean Squared Error
  """
  # Crear copia para no modificar el DataFrame original
  datos = datos.copy()
  
  # Calcular pronóstico
  datos["Pronostico"] = datos["ventas"].rolling(window=N).mean().shift(1).round(0)
  
  # Calcular errores
  datos["error"] = datos["Pronostico"] - datos["ventas"]
  datos["error_abs"] = datos["error"].abs()
  datos["ape"] = datos["error_abs"] / datos["ventas"]
  datos["ape´"] = datos["error_abs"] / datos["Pronostico"]
  datos["error_cuadrado"] = datos["error"] * datos["error"]

  # MEDIDAS DE ERROR
  MAPE = datos["ape"].mean()
  MAPE_prima = datos["ape´"].mean()
  MSE = datos["error_cuadrado"].mean()
  RMSE = MSE**(1/2)
  
  return datos, MAPE, MAPE_prima, MSE, RMSE

# Ejecución solo si se corre directamente (no al importar)
if __name__ == "__main__":
  datos = pd.read_csv("venta_historicas.csv")
  df_pronostico, MAPE_result, MAPE_prima_result, MSE_result, RMSE_result = pronosticar(datos, 3)
  print(df_pronostico)
  print("Los resultados de las medidas son:", MAPE_result, MAPE_prima_result, MSE_result, RMSE_result)