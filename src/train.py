"""
Modelos de pronostico para las transferencias mensuales (Etapa 2 y 3).

Etapa 2: modelo base estacional (seasonal naive).
Etapa 3: SARIMA y Prophet, comparados contra el base con MAE / MAPE,
         mas un grafico de real vs. pronosticado.

Métricas:
  - MAE  (Mean Absolute Error): error promedio en cantidad de transferencias.
  - MAPE (Mean Absolute Percentage Error): error promedio en %, comparable
    entre provincias de distinto tamano.

Uso:
    python -m src.train
    python -m src.train --provincia "Córdoba" --test-meses 12
    python -m src.train --buscar          # corre la grilla de SARIMA por AIC
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import pickle
import platform
import warnings
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: guardamos los graficos a archivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels

from src.data import get_series, load_clean, split_train_test
from src.model import SARIMA_ORDER, SARIMA_SEASONAL_ORDER, ModeloSARIMA
# Helpers de runtime compartidos con la API (en un modulo sin matplotlib/prophet).
from src.predict import (
    MODELS_DIR,
    cargar_metricas,
    cargar_modelo,
    pronostico_futuro,
    ruta_metricas,
    rutas_modelo,
    slug,
)

warnings.filterwarnings("ignore")  # statsmodels/prophet son charlatanes con los avisos
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

ESTACIONALIDAD = 12  # meses: el patron se repite cada anio

# Semilla global. SARIMA (MLE) es determinista, pero la fijamos igual para que
# cualquier parte con azar (p. ej. Prophet) sea reproducible.
SEED = 42

# Carpeta de salidas (tablas y graficos) para el README. MODELS_DIR viene de src.predict.
REPORTS_DIR = "reports"

# --- Parametros de SARIMA -----------------------------------------------------
# Las constantes SARIMA_ORDER y SARIMA_SEASONAL_ORDER viven en src/model.py
# (asi lo importa la API y se puede cargar el pickle). Eleccion explicada:
#   * s = 12      -> datos mensuales con ciclo anual.
#   * d = 1       -> el test ADF sobre el nivel (p=0.33) dice "no estacionaria".
#   * D = 1       -> la serie tiene estacionalidad fuerte; con la diferencia
#                    estacional (lag 12) el ADF da p<0.001.
#   * (p,q)=(1,2) -> elegidos por AIC sobre una grilla (ver buscar_sarima).
#   * (P,Q)=(1,1) -> el ACF/PACF de la serie diferenciada muestran un pico
#                    marcado en el lag 12 -> estacional AR y MA.
# Los mejores AIC quedaron empatados (dentro de 0.1); nos quedamos con
# (1,1,2)(1,1,1,12), que es ademas la opcion mas estandar y parsimoniosa.


# --- Metricas -----------------------------------------------------------------

def mae(y_real, y_pred) -> float:
    """Error absoluto medio: promedio de |real - predicho|."""
    y_real, y_pred = pd.Series(y_real), pd.Series(y_pred)
    return float((y_real - y_pred).abs().mean())


def mape(y_real, y_pred) -> float:
    """
    Error porcentual absoluto medio (en %).

    Ojo: el MAPE no esta definido cuando el valor real es 0 (dividimos por 0).
    Si hay ceros en el periodo real, los salteamos y avisamos cuántos quedaron afuera.
    """
    y_real, y_pred = pd.Series(y_real), pd.Series(y_pred)
    ceros = int((y_real == 0).sum())
    if ceros:
        print(f"Aviso: {ceros} mes(es) con valor real 0 se excluyen del MAPE.")
    mask = y_real != 0
    if not mask.any():
        return float("nan")
    error_pct = ((y_real[mask] - y_pred[mask]).abs() / y_real[mask].abs()) * 100
    return float(error_pct.mean())


# --- Modelo base: seasonal naive ---------------------------------------------

def pronostico_estacional(serie: pd.Series, fechas: pd.DatetimeIndex,
                          estacionalidad: int = ESTACIONALIDAD) -> pd.Series:
    """
    Seasonal naive: para cada fecha futura, toma el valor real de `estacionalidad`
    meses antes (por defecto 12 = mismo mes del anio pasado).
    """
    return serie.shift(estacionalidad).loc[fechas]


# --- SARIMA -------------------------------------------------------------------

def fit_sarima(serie: pd.Series, order: tuple = SARIMA_ORDER,
               seasonal_order: tuple = SARIMA_SEASONAL_ORDER):
    """Ajusta un SARIMA sobre `serie` y devuelve el resultado (modelo entrenado)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    return SARIMAX(
        serie,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,   # dejamos que el optimizador busque libremente
        enforce_invertibility=False,
    ).fit(disp=False)


def pronostico_sarima(resultado, fechas: pd.DatetimeIndex) -> pd.Series:
    """Pronostica `len(fechas)` pasos hacia adelante con un SARIMA ya ajustado."""
    pred = resultado.forecast(steps=len(fechas))
    pred.index = fechas
    pred.name = "SARIMA"
    return pred


def ajustar_sarima(train: pd.Series, fechas: pd.DatetimeIndex,
                   order: tuple = SARIMA_ORDER,
                   seasonal_order: tuple = SARIMA_SEASONAL_ORDER):
    """Ajusta SARIMA sobre train y pronostica el test (usado en la comparacion)."""
    resultado = fit_sarima(train, order, seasonal_order)
    return resultado, pronostico_sarima(resultado, fechas)


def buscar_sarima(train: pd.Series) -> tuple:
    """
    Grilla chica de (p,q) y (P,Q) con d=D=1 fijos; devuelve el de menor AIC.

    Es la parte "data-driven" de la eleccion de parametros. Tarda ~20 s.
    """
    mejores = []
    for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
        try:
            m = ajustar_sarima_para_aic(train, (p, 1, q), (P, 1, Q, 12))
            mejores.append((m, (p, 1, q), (P, 1, Q, 12)))
        except Exception:
            continue
    mejores.sort(key=lambda t: t[0])
    print("Grilla SARIMA (mejores 5 por AIC):")
    for aic, o, so in mejores[:5]:
        print(f"  AIC={aic:9.2f}  order={o}  seasonal={so}")
    _, mejor_order, mejor_seasonal = mejores[0]
    return mejor_order, mejor_seasonal


def ajustar_sarima_para_aic(train: pd.Series, order, seasonal_order) -> float:
    """Ajusta y devuelve el AIC (atajo para la grilla)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    res = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return float(res.aic)


# --- Prophet ------------------------------------------------------------------

def ajustar_prophet(train: pd.Series, fechas: pd.DatetimeIndex):
    """
    Ajusta Prophet (tendencia + estacionalidad anual) y pronostica el test.

    Por que Prophet rinde peor que SARIMA aca: la serie tiene dos saltos
    estructurales muy fuertes, no estacionales: la crisis de 2002 (el piso cae a
    ~3.300 en pleno 2001/2002) y el derrumbe de abril 2020 por la pandemia
    (9.165 -> 2.196 en un mes). Prophet modela tendencia suave + estacionalidad
    anual; esos quiebres puntuales le ensucian la estimacion de tendencia y
    estacionalidad, mientras que SARIMA los maneja mejor al diferenciar.
    """
    from prophet import Prophet

    # Prophet pide un DataFrame con columnas 'ds' (fecha) e 'y' (valor).
    datos = pd.DataFrame({"ds": train.index, "y": train.values})

    modelo = Prophet(
        yearly_seasonality=True,    # ciclo anual: lo que mas importa aca
        weekly_seasonality=False,   # datos mensuales: semanal no aplica
        daily_seasonality=False,    # idem diario
        seasonality_mode="additive",  # la estacionalidad no escala con el nivel
    )
    modelo.fit(datos)

    futuro = pd.DataFrame({"ds": fechas})
    fc = modelo.predict(futuro)
    pred = pd.Series(fc["yhat"].values, index=fechas, name="Prophet")
    return modelo, pred


# --- Comparacion --------------------------------------------------------------

def comparar(provincia: str = "Santa Fe", test_meses: int = 12) -> pd.DataFrame:
    """Ajusta los 3 modelos, imprime metricas, grafica real vs predicho y guarda todo."""
    datos = load_clean()
    serie = get_series(datos, provincia)
    train, test = split_train_test(serie, test_months=test_meses)

    # 1. Pronosticos de los tres modelos.
    preds = {
        "base estacional": pronostico_estacional(serie, test.index),
    }
    _, pred_sarima = ajustar_sarima(train, test.index)
    preds[f"SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER}"] = pred_sarima
    _, pred_prophet = ajustar_prophet(train, test.index)
    preds["Prophet"] = pred_prophet

    # 2. Tabla de metricas.
    filas = []
    for nombre, pred in preds.items():
        filas.append({"modelo": nombre, "MAE": mae(test, pred), "MAPE": mape(test, pred)})
    tabla = pd.DataFrame(filas).sort_values("MAE").reset_index(drop=True)

    print(f"\n=== Comparacion ({provincia}) | test: {test.index.min().date()} a {test.index.max().date()} ===")
    print(tabla.to_string(index=False, formatters={"MAE": "{:,.0f}".format, "MAPE": "{:.2f}%".format}))

    # 3. Grafico: real vs. cada pronostico dentro del test.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test.index, test.values, "o-", color="black", lw=2, label="real")
    for nombre, pred in preds.items():
        ax.plot(test.index, pred.values, "--", marker=".", label=nombre)
    ax.set_title(f"Transferencias mensuales - {provincia}: real vs. pronostico (test)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Transferencias")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # 4. Guardamos tabla y grafico en reports/ para el README.
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    nombre = slug(provincia)
    tabla.to_csv(f"{REPORTS_DIR}/comparacion_{nombre}.csv", index=False)
    fig.savefig(f"{REPORTS_DIR}/comparacion_{nombre}.png", dpi=120)
    plt.close(fig)

    # 5. Guardamos las metricas de los 3 modelos en models/ para el endpoint /metricas.
    ruta_met = ruta_metricas(provincia)
    metricas_json = {
        "provincia": provincia,
        "test_meses": test_meses,
        "test_rango": [str(test.index.min().date()), str(test.index.max().date())],
        "metricas": [
            {"modelo": r["modelo"], "MAE": round(r["MAE"], 2), "MAPE": round(r["MAPE"], 2)}
            for _, r in tabla.iterrows()
        ],
    }
    with open(ruta_met, "w", encoding="utf-8") as f:
        json.dump(metricas_json, f, ensure_ascii=False, indent=2)

    print(f"\nGuardado: {REPORTS_DIR}/comparacion_{nombre}.csv y .png")
    print(f"Guardado: {ruta_met}")

    return tabla


# --- Etapa 4: modelo final + persistencia -------------------------------------

def entrenar_final(provincia: str = "Santa Fe", test_meses: int = 12) -> dict:
    """
    Entrena el SARIMA de produccion y lo guarda. Flujo:

      1. Mide en el test honesto: ajusta sobre train y pronostica el test
         -> de ahi salen MAE/MAPE (las metricas que reportamos).
      2. Reentrena el modelo con TODA la serie (train + test), que es el que
         va a usar la API para pronosticar meses futuros.
      3. Guarda el modelo (.pkl) y sus metadatos (.json) en models/.
      4. Genera el grafico historico + pronostico para el README.
    """
    datos = load_clean()
    serie = get_series(datos, provincia)
    train, test = split_train_test(serie, test_months=test_meses)

    # 1. Metricas honestas: fit en train, pronostico del test.
    _, pred_test = ajustar_sarima(train, test.index)
    metricas = {"MAE": mae(test, pred_test), "MAPE": mape(test, pred_test)}
    print(f"\n=== Entrenamiento final ({provincia}) ===")
    print(f"Metricas en test ({test.index.min().date()} a {test.index.max().date()}):")
    print(f"  MAE  = {metricas['MAE']:,.0f}")
    print(f"  MAPE = {metricas['MAPE']:.2f} %")

    # 2. Reentrenar con toda la serie (train + test).
    resultado_final = fit_sarima(serie)
    modelo_final = ModeloSARIMA(
        valores=serie.values,
        fechas=serie.index,
        params=resultado_final.params.values,
    )
    print(f"Reentrenado con toda la serie: {len(serie)} meses "
          f"({serie.index.min().date()} a {serie.index.max().date()})")

    # 3. Guardar modelo (liviano) + metadatos.
    os.makedirs(MODELS_DIR, exist_ok=True)
    ruta_modelo, ruta_meta = rutas_modelo(provincia)
    with open(ruta_modelo, "wb") as f:
        pickle.dump(modelo_final, f)
    peso_kb = os.path.getsize(ruta_modelo) / 1024

    metadatos = {
        "provincia": provincia,
        "modelo": "SARIMA",
        "order": list(SARIMA_ORDER),
        "seasonal_order": list(SARIMA_SEASONAL_ORDER),
        "test_meses": test_meses,
        "datos_rango": [str(serie.index.min().date()), str(serie.index.max().date())],
        "test_rango": [str(test.index.min().date()), str(test.index.max().date())],
        "metricas_test": metricas,
        "entrenado_con": "serie completa (train + test)",
        "formato": "params + serie; se reconstruye con SARIMAX.smooth (liviano)",
        "peso_kb": round(peso_kb, 1),
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versiones": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "seed": SEED,
    }
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=2)

    print(f"Guardado: {ruta_modelo} ({peso_kb:,.0f} KB) y {ruta_meta}")

    # 4. Grafico historico reciente + pronostico futuro (para el README).
    graficar_pronostico(provincia, serie=serie)

    return metadatos


def graficar_pronostico(provincia: str = "Santa Fe", meses: int = 12,
                        serie: pd.Series | None = None) -> str:
    """
    Grafica los ultimos 3 anios reales + el pronostico de los proximos `meses`
    (usando el modelo guardado) y lo guarda en reports/.
    """
    if serie is None:
        serie = get_series(load_clean(), provincia)
    futuro = pronostico_futuro(provincia, meses)

    fig, ax = plt.subplots(figsize=(11, 5))
    historico = serie.iloc[-36:]  # ultimos 3 anios, para dar contexto
    ax.plot(historico.index, historico.values, "-", color="black", lw=2, label="historico")
    ax.plot(futuro.index, futuro.values, "--o", color="#d62728", label=f"pronostico ({meses} meses)")
    ax.axvline(historico.index[-1], color="gray", ls=":", lw=1)
    ax.set_title(f"Transferencias mensuales - {provincia}: historico y pronostico")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Transferencias")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    ruta = f"{REPORTS_DIR}/pronostico_{slug(provincia)}.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print(f"Guardado: {ruta}")
    return ruta


# --- Etapa 2: solo el base ----------------------------------------------------

def evaluar(provincia: str = "Santa Fe", test_meses: int = 12) -> dict:
    """Evaluacion del modelo base estacional (Etapa 2)."""
    datos = load_clean()
    serie = get_series(datos, provincia)
    train, test = split_train_test(serie, test_months=test_meses)

    pred = pronostico_estacional(serie, test.index)
    metricas = {"mae": mae(test, pred), "mape": mape(test, pred)}

    print(f"\n=== Modelo base estacional ({provincia}) ===")
    print(f"train: {len(train)} meses | test: {len(test)} meses "
          f"({test.index.min().date()} a {test.index.max().date()})")
    print(f"\nMAE : {metricas['mae']:,.0f} transferencias (promedio)")
    print(f"MAPE: {metricas['mape']:.2f} %")
    return {"provincia": provincia, **metricas}


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelos de transferencias mensuales.")
    parser.add_argument("--provincia", default="Santa Fe", help="Provincia a modelar.")
    parser.add_argument("--test-meses", type=int, default=12, help="Meses de test al final.")
    parser.add_argument("--base", action="store_true", help="Solo el modelo base (Etapa 2).")
    parser.add_argument("--buscar", action="store_true", help="Grilla SARIMA por AIC.")
    parser.add_argument("--sin-guardar", action="store_true",
                        help="No entrenar/guardar el modelo final (solo comparar).")
    args = parser.parse_args()

    np.random.seed(SEED)  # reproducibilidad (SARIMA ya es determinista)

    if args.buscar:
        serie = get_series(load_clean(), args.provincia)
        train, _ = split_train_test(serie, test_months=args.test_meses)
        order, seasonal = buscar_sarima(train)
        print(f"\nMejor por AIC: order={order} seasonal={seasonal}")
        return

    if args.base:
        evaluar(provincia=args.provincia, test_meses=args.test_meses)
        return

    # Pipeline por defecto (Etapas 3 y 4): comparar y entrenar/guardar el final.
    comparar(provincia=args.provincia, test_meses=args.test_meses)
    if not args.sin_guardar:
        entrenar_final(provincia=args.provincia, test_meses=args.test_meses)


if __name__ == "__main__":
    main()
