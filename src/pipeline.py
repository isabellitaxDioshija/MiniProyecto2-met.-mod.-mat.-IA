"""

Logica compartida de datos y modelos para el Mini-Proyecto 2.
La usan tanto main.py como app.py,
para no duplicar el codigo de carga de datos, entrenamiento y evaluacion.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
TEST_SIZE = 0.2
VARIANCE_THRESHOLD = 0.90

TARGET = "temperature_2m"
FEATURES = ["relative_humidity_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "dataset-MP2-04.xlsx"

np.random.seed(RANDOM_STATE)


def load_and_clean_data(path: Path = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    """Lee el Excel del dataset (con metadatos en las primeras filas) y devuelve
    un dataframe limpio con columnas renombradas y filas incompletas eliminadas.
    """
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)

    metadata = dict(zip(raw.iloc[0], raw.iloc[1]))
    header = raw.iloc[3].tolist()
    data = raw.iloc[4:].reset_index(drop=True)
    data.columns = header

    data["time"] = pd.to_datetime(data["time"])
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col])

    n_nulls = data.isna().any(axis=1).sum()
    data = data.dropna().reset_index(drop=True)
    data = data.rename(columns={
        "temperature_2m (°C)": "temperature_2m",
        "relative_humidity_2m (%)": "relative_humidity_2m",
        "cloud_cover (%)": "cloud_cover",
        "wind_speed_10m (km/h)": "wind_speed_10m",
        "shortwave_radiation (W/m²)": "shortwave_radiation",
    })

    if verbose:
        print(f"Metadatos de la estacion: {metadata}")
        print(f"Registros con datos faltantes descartados: {n_nulls} | Registros limpios: {len(data)}")

    return data


def split_and_scale(data: pd.DataFrame):
    X = data[FEATURES].values
    y = data[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler


def evaluate(y_true, y_pred, fit_time) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "fit_time_ms": fit_time * 1000,
    }


def train_linear_regression(X_train, y_train, X_test, y_test):
    t0 = time.time()
    model = LinearRegression().fit(X_train, y_train)
    fit_time = time.time() - t0
    pred = model.predict(X_test)
    metrics = evaluate(y_test, pred, fit_time)
    return model, pred, metrics


def train_knn(X_train, y_train, X_test, y_test, k_range=range(2, 21)):
    grid = GridSearchCV(KNeighborsRegressor(), {"n_neighbors": list(k_range)}, cv=5, scoring="r2")
    grid.fit(X_train, y_train)
    best_k = grid.best_params_["n_neighbors"]

    t0 = time.time()
    model = KNeighborsRegressor(n_neighbors=best_k).fit(X_train, y_train)
    fit_time = time.time() - t0

    pred = model.predict(X_test)
    metrics = evaluate(y_test, pred, fit_time)
    metrics["k"] = best_k
    return model, pred, metrics


def fit_pca(X_train, variance_threshold: float = VARIANCE_THRESHOLD):
    pca_full = PCA(n_components=len(FEATURES)).fit(X_train)
    var_explained = pca_full.explained_variance_ratio_
    var_cumulative = np.cumsum(var_explained)
    n_components = int(np.argmax(var_cumulative >= variance_threshold) + 1)
    pca = PCA(n_components=n_components).fit(X_train)
    return pca, pca_full, var_explained, var_cumulative, n_components


def train_knn_with_pca(X_train, X_test, y_train, y_test, k_range=range(2, 21)):
    pca, pca_full, var_explained, var_cumulative, n_components = fit_pca(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    grid = GridSearchCV(KNeighborsRegressor(), {"n_neighbors": list(k_range)}, cv=5, scoring="r2")
    grid.fit(X_train_pca, y_train)
    best_k = grid.best_params_["n_neighbors"]

    t0 = time.time()
    model = KNeighborsRegressor(n_neighbors=best_k).fit(X_train_pca, y_train)
    fit_time = time.time() - t0

    pred = model.predict(X_test_pca)
    metrics = evaluate(y_test, pred, fit_time)
    metrics["k"] = best_k
    metrics["n_components"] = n_components
    metrics["variance_retained_pct"] = var_cumulative[n_components - 1] * 100

    return model, pca, pred, metrics, (var_explained, var_cumulative, pca_full)


def run_full_pipeline(path: Path = DATA_PATH, verbose: bool = True):
    """Corre todo el flujo (datos -> LR -> kNN -> PCA+kNN) y devuelve todos los
    objetos entrenados y metricas, para ser reutilizados por main.py o app.py.
    """
    data = load_and_clean_data(path, verbose=verbose)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(data)

    lr_model, lr_pred, lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test)
    knn_model, knn_pred, knn_metrics = train_knn(X_train, y_train, X_test, y_test)
    knn_pca_model, pca, knn_pca_pred, knn_pca_metrics, pca_info = train_knn_with_pca(
        X_train, X_test, y_train, y_test
    )

    return {
        "data": data,
        "scaler": scaler,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "lr_model": lr_model, "lr_pred": lr_pred, "lr_metrics": lr_metrics,
        "knn_model": knn_model, "knn_pred": knn_pred, "knn_metrics": knn_metrics,
        "knn_pca_model": knn_pca_model, "pca": pca, "knn_pca_pred": knn_pca_pred,
        "knn_pca_metrics": knn_pca_metrics, "pca_info": pca_info,
    }
