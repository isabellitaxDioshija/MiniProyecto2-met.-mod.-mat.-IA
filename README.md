# Mini-Proyecto 2 — Métodos y Modelos Matemáticos para IA

Regresión Lineal, k-NN y PCA aplicados a variables climáticas (Berlín, 52.5°N / 13.5°E), con una interfaz web (Streamlit) para la demo en vivo.

## Integrantes
- _Juan Andrés Correa Arenas_
- _Isabella Garzón Salazar_
- _Juan José López Valencia_

## Objetivo
Estimar la **temperatura del aire a 2 m (`temperature_2m`)** a partir de variables meteorológicas horarias (humedad relativa, nubosidad, velocidad del viento y radiación solar), comparando **Regresión Lineal** vs. **k-NN**, y evaluando el efecto de **PCA** sobre el modelo con mejor desempeño.

## Estructura del repositorio
```
.
├── dataset/
│   └── dataset-MP2-04.xlsx       # Dataset asignado (variables climáticas horarias)
├── src/
│   ├── pipeline.py                # Lógica compartida: carga de datos, entrenamiento, PCA
│   ├── main.py                    # Script de consola: corre todo, guarda gráficas y tabla de resultados
│   └── app.py                     # Interfaz web (Streamlit) para la demo en vivo
├── outputs/
│   ├── graficos/                   # Gráficas generadas por main.py (.png)
│   └── results/                   # Tabla de métricas final (.csv)
├── requirements.txt
└── README.md
```

## Cómo ejecutar

1. Clona el repositorio y crea un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. **Análisis completo por consola** (genera gráficas y tabla de resultados en `outputs/`):
   ```bash
   python src/main.py
   ```
4. **Interfaz gráfica para la demo en vivo** (se abre en el navegador):
   ```bash
   streamlit run src/app.py
   ```
   La app permite:
   - Mover sliders de humedad, nubosidad, viento y radiación y ver la temperatura estimada en tiempo real, con los 3 modelos entrenados (Regresión Lineal, k-NN, k-NN+PCA).
   - Ver la comparación de métricas (R², RMSE, MAE) y las gráficas de varianza explicada de PCA.
   - Explorar el dataset limpio usado para entrenar.

## Metodología
1. **Carga y limpieza de datos** (`pipeline.load_and_clean_data`): el Excel trae metadatos de la estación en las primeras filas; se separan de la tabla horaria y se eliminan registros incompletos (últimas 96 h sin datos).
2. **Regresión Lineal** (`pipeline.train_linear_regression`): modelo base, variables estandarizadas (z-score).
3. **k-NN como regresor** (`pipeline.train_knn`): hiperparámetro `k` seleccionado por validación cruzada (5-fold).
4. **Comparación**: R², RMSE, MAE y gráficas de real vs. predicho / residuos para cada modelo.
5. **PCA** (`pipeline.train_knn_with_pca`): aplicado sobre el modelo ganador, reduciendo dimensionalidad de las predictoras (reteniendo ≥90% de varianza) y reentrenando.
6. **Comparación final**: desempeño con y sin PCA, tiempos de entrenamiento y varianza retenida.
7. **Interfaz de demo** (`app.py`): reutiliza exactamente los mismos modelos entrenados en `pipeline.py` (sin duplicar lógica) para predecir en tiempo real sobre valores ingresados por el usuario.

## Resultados obtenidos

| Modelo | R² | RMSE (°C) | MAE (°C) |
|---|---|---|---|
| Regresión Lineal | 0.6833 | 2.33 | 1.93 |
| k-NN (k=4) | **0.7523** | **2.06** | **1.69** |
| k-NN + PCA (3 comps, 92.6% var.) | 0.7252 | 2.17 | 1.71 |

El modelo **k-NN** superó a la regresión lineal; al aplicarle PCA (3 de 4 componentes, reteniendo 92.6% de la varianza) el desempeño se mantuvo muy cercano (ΔR² = −0.027), con la ventaja de trabajar con predictoras no correlacionadas. Dado que el dataset original solo tiene 4 predictoras, la ganancia de PCA aquí es más interpretativa (eliminar correlación entre variables) que computacional.

## Fuente del dataset
Datos meteorológicos horarios (tipo Open-Meteo) para la ubicación lat=52.5°, lon=13.5° (Berlín, Alemania).
