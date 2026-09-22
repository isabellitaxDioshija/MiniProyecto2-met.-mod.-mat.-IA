"""
Interfaz grafica (Streamlit) 

Uso:
    streamlit run src/app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="MP2 - Prediccion de Temperatura", page_icon="🌤️", layout="wide")


@st.cache_resource(show_spinner="Entrenando modelos...")
def get_trained_pipeline():
    return pl.run_full_pipeline(verbose=False)


results = get_trained_pipeline()
data = results["data"]
scaler = results["scaler"]

st.title("MP2 — Estimación de temperatura a partir de variables climáticas")
st.caption(
    "Regresión Lineal, k-NN y PCA aplicados a datos meteorológicos horarios "
    "(Berlín, 52.5°N / 13.5°E). Métodos y Modelos Matemáticos para IA."
)

tab_demo, tab_resultados, tab_datos = st.tabs(
    ["Predicción interactiva", "Comparación de modelos", "Dataset"]
)


with tab_demo:
    st.subheader("Ingresa condiciones climáticas para estimar la temperatura")

    col_inputs, col_outputs = st.columns([1, 1.3])

    with col_inputs:
        humidity = st.slider(
            "Humedad relativa (%)",
            float(data["relative_humidity_2m"].min()), float(data["relative_humidity_2m"].max()),
            float(data["relative_humidity_2m"].median())
        )
        cloud_cover = st.slider(
            "Nubosidad (%)",
            float(data["cloud_cover"].min()), float(data["cloud_cover"].max()),
            float(data["cloud_cover"].median())
        )
        wind_speed = st.slider(
            "Velocidad del viento (km/h)",
            float(data["wind_speed_10m"].min()), float(data["wind_speed_10m"].max()),
            float(data["wind_speed_10m"].median())
        )
        radiation = st.slider(
            "Radiación solar (W/m²)",
            float(data["shortwave_radiation"].min()), float(data["shortwave_radiation"].max()),
            float(data["shortwave_radiation"].median())
        )

        modelo_elegido = st.radio(
            "Modelo a usar como predicción principal",
            ["k-NN (ganador)", "Regresión Lineal", "k-NN + PCA"],
            index=0,
        )

  
    x_new = np.array([[humidity, cloud_cover, wind_speed, radiation]])
    x_new_scaled = scaler.transform(x_new)

    pred_lr = results["lr_model"].predict(x_new_scaled)[0]
    pred_knn = results["knn_model"].predict(x_new_scaled)[0]
    x_new_pca = results["pca"].transform(x_new_scaled)
    pred_knn_pca = results["knn_pca_model"].predict(x_new_pca)[0]

    preds = {
        "k-NN (ganador)": pred_knn,
        "Regresión Lineal": pred_lr,
        "k-NN + PCA": pred_knn_pca,
    }

    with col_outputs:
        st.metric("🌡️ Temperatura estimada", f"{preds[modelo_elegido]:.1f} °C", help=f"Modelo: {modelo_elegido}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Regresión Lineal", f"{pred_lr:.1f} °C")
        c2.metric(f"k-NN (k={results['knn_metrics']['k']})", f"{pred_knn:.1f} °C")
        c3.metric(f"k-NN + PCA ({results['knn_pca_metrics']['n_components']} comps)", f"{pred_knn_pca:.1f} °C")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(results["y_test"], results["knn_pred"], alpha=0.4, color="gray", label="Datos de prueba")
        ax.axhline(preds[modelo_elegido], color="crimson", linestyle="--", label="Predicción actual")
        ax.set_xlabel("Temperatura real (°C) — conjunto de prueba")
        ax.set_ylabel("Temperatura predicha (°C)")
        ax.set_title("Ubicación de la predicción actual vs. el conjunto de prueba")
        ax.legend()
        st.pyplot(fig)


with tab_resultados:
    st.subheader("Desempeño de los modelos sobre el conjunto de prueba")

    comparison = pd.DataFrame({
        "Modelo": [
            "Regresión Lineal",
            f"k-NN (k={results['knn_metrics']['k']})",
            f"k-NN + PCA ({results['knn_pca_metrics']['n_components']} comps)",
        ],
        "R²": [results["lr_metrics"]["r2"], results["knn_metrics"]["r2"], results["knn_pca_metrics"]["r2"]],
        "RMSE (°C)": [results["lr_metrics"]["rmse"], results["knn_metrics"]["rmse"], results["knn_pca_metrics"]["rmse"]],
        "MAE (°C)": [results["lr_metrics"]["mae"], results["knn_metrics"]["mae"], results["knn_pca_metrics"]["mae"]],
        "Varianza retenida (%)": [100.0, 100.0, results["knn_pca_metrics"]["variance_retained_pct"]],
    })
    st.dataframe(comparison, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        modelos = comparison["Modelo"]
        ax.bar(modelos, comparison["R²"], color=["steelblue", "seagreen", "purple"])
        ax.set_ylabel("R²")
        ax.set_title("Comparación de R² por modelo")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    with col2:
        var_explained, var_cumulative, _ = results["pca_info"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        pcs = [f"PC{i+1}" for i in range(len(pl.FEATURES))]
        ax.bar(pcs, var_explained * 100, color="royalblue", alpha=0.7, label="Individual")
        ax.step(pcs, var_cumulative * 100, where="mid", color="crimson", marker="o", label="Acumulada")
        ax.set_ylabel("% Varianza")
        ax.set_title("Varianza explicada por componente (PCA)")
        ax.legend()
        st.pyplot(fig)

    ganador = "k-NN" if results["knn_metrics"]["r2"] > results["lr_metrics"]["r2"] else "Regresión Lineal"
    st.markdown(
        f"**Modelo con mejor desempeño inicial:** {ganador}. "
        f"Al aplicarle PCA ({results['knn_pca_metrics']['n_components']} de {len(pl.FEATURES)} componentes, "
        f"reteniendo {results['knn_pca_metrics']['variance_retained_pct']:.1f}% de la varianza), "
        f"el R² pasó de {results['knn_metrics']['r2']:.3f} a {results['knn_pca_metrics']['r2']:.3f}."
    )


with tab_datos:
    st.subheader("Dataset utilizado")
    st.write(
        "Mediciones meteorológicas horarias para la estación en lat=52.5°, lon=13.5° (Berlín, Alemania). "
        f"Registros limpios utilizados: **{len(data)}**."
    )
    st.dataframe(data, width="stretch", hide_index=True)
