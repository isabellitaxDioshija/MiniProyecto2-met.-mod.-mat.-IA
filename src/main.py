"""
Uso:
    python src/main.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns

import pipeline as pl

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n[{name}]")
    print(f"  R2   = {metrics['r2']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f} °C")
    print(f"  MAE  = {metrics['mae']:.4f} °C")
    print(f"  Tiempo de entrenamiento = {metrics['fit_time_ms']:.3f} ms")


def plot_exploratory(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(data["time"], data[pl.TARGET], color="crimson")
    axes[0].set_title("Serie temporal de temperatura (variable objetivo)")
    axes[0].set_ylabel("°C")
    axes[0].tick_params(axis="x", rotation=30)

    sns.heatmap(data[[pl.TARGET] + pl.FEATURES].corr(), annot=True, cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title("Correlacion entre variables")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_exploratorio.png", dpi=150)
    plt.close(fig)


def plot_predictions(y_true, y_pred, title, filename, color) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].scatter(y_true, y_pred, alpha=0.7, edgecolor="k", color=color)
    axes[0].plot(lims, lims, "r--", label="Prediccion ideal")
    axes[0].set_xlabel("Temperatura real (°C)")
    axes[0].set_ylabel("Temperatura predicha (°C)")
    axes[0].set_title(f"{title}: Real vs. Predicho")
    axes[0].legend()

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.7, edgecolor="k", color=color)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Temperatura predicha (°C)")
    axes[1].set_ylabel("Residuo (real - predicho)")
    axes[1].set_title(f"{title}: Residuos")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def plot_scree_and_loadings(var_explained, var_cumulative, pca_full) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    pcs = [f"PC{i+1}" for i in range(len(pl.FEATURES))]
    axes[0].bar(pcs, var_explained * 100, color="royalblue", alpha=0.7, label="Individual")
    axes[0].step(pcs, var_cumulative * 100, where="mid", color="crimson", marker="o", label="Acumulada")
    axes[0].axhline(pl.VARIANCE_THRESHOLD * 100, color="gray", linestyle=":", label=f"Umbral {pl.VARIANCE_THRESHOLD*100:.0f}%")
    axes[0].set_title("Scree Plot")
    axes[0].set_ylabel("% Varianza")
    axes[0].legend()

    loadings = pd.DataFrame(pca_full.components_.T, index=pl.FEATURES, columns=pcs)
    sns.heatmap(loadings, annot=True, cmap="vlag", center=0, ax=axes[1])
    axes[1].set_title("Cargas (loadings) de cada variable")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_pca_scree_loadings.png", dpi=150)
    plt.close(fig)


def build_comparison_table(lr_metrics, knn_metrics, pca_metrics) -> pd.DataFrame:
    return pd.DataFrame({
        "Modelo": [
            "Regresion Lineal",
            f"k-NN (k={knn_metrics['k']})",
            f"k-NN + PCA ({pca_metrics['n_components']} comps, k={pca_metrics['k']})",
        ],
        "N_variables": [len(pl.FEATURES), len(pl.FEATURES), pca_metrics["n_components"]],
        "R2": [lr_metrics["r2"], knn_metrics["r2"], pca_metrics["r2"]],
        "RMSE_C": [lr_metrics["rmse"], knn_metrics["rmse"], pca_metrics["rmse"]],
        "MAE_C": [lr_metrics["mae"], knn_metrics["mae"], pca_metrics["mae"]],
        "Tiempo_entrenamiento_ms": [lr_metrics["fit_time_ms"], knn_metrics["fit_time_ms"], pca_metrics["fit_time_ms"]],
        "Varianza_retenida_pct": [100.0, 100.0, pca_metrics["variance_retained_pct"]],
    })


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MP2 - Regresion Lineal, k-NN y PCA sobre variables climaticas")
    print("=" * 70)

    results = pl.run_full_pipeline(verbose=True)

    print(f"\nEntrenamiento: {len(results['X_train'])} registros | Prueba: {len(results['X_test'])} registros")
    plot_exploratory(results["data"])

    print_metrics("Regresion Lineal", results["lr_metrics"])
    plot_predictions(results["y_test"], results["lr_pred"], "Regresion Lineal", "02_regresion_lineal.png", color="steelblue")

    print_metrics(f"k-NN (k={results['knn_metrics']['k']})", results["knn_metrics"])
    plot_predictions(results["y_test"], results["knn_pred"], f"k-NN (k={results['knn_metrics']['k']})", "03_knn.png", color="seagreen")

    winner = "k-NN" if results["knn_metrics"]["r2"] > results["lr_metrics"]["r2"] else "Regresion Lineal"
    print(f"\nModelo con mejor desempeño: {winner}")

    var_explained, var_cumulative, pca_full = results["pca_info"]
    plot_scree_and_loadings(var_explained, var_cumulative, pca_full)

    pca_metrics = results["knn_pca_metrics"]
    print_metrics(f"k-NN + PCA ({pca_metrics['n_components']} comps, k={pca_metrics['k']})", pca_metrics)
    plot_predictions(
        results["y_test"], results["knn_pca_pred"],
        f"k-NN + PCA ({pca_metrics['n_components']} comps)", "05_knn_pca.png", color="purple"
    )

    comparison = build_comparison_table(results["lr_metrics"], results["knn_metrics"], pca_metrics)
    print("\nComparacion final:")
    print(comparison.to_string(index=False))
    comparison.to_csv(RESULTS_DIR / "comparacion_final.csv", index=False)

    delta_r2 = pca_metrics["r2"] - results["knn_metrics"]["r2"]
    print(f"\nDelta R2 (k-NN+PCA vs. k-NN sin PCA): {delta_r2:+.4f}")
    print(f"Graficas guardadas en: {FIGURES_DIR}")
    print(f"Tabla de resultados guardada en: {RESULTS_DIR / 'comparacion_final.csv'}")


if __name__ == "__main__":
    main()
