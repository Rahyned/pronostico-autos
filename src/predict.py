"""
Lectura del modelo entrenado y generacion de pronosticos (runtime de la API).

Este modulo NO importa matplotlib ni prophet a proposito: la API lo importa y
asi el contenedor de produccion queda liviano. Solo necesita pandas/numpy y
statsmodels (para reconstruir el SARIMA al pronosticar).
"""

from __future__ import annotations

import json
import pickle

import pandas as pd

# Carpeta donde se guardan el modelo y sus metadatos/metricas.
MODELS_DIR = "models"


def slug(texto: str) -> str:
    """Nombre de archivo simple a partir de la provincia (ej. 'Santa Fe'->'santa_fe')."""
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    s = texto.lower().replace(" ", "_")
    for k, v in reemplazos.items():
        s = s.replace(k, v)
    return s


def rutas_modelo(provincia: str) -> tuple[str, str]:
    """Devuelve (ruta .pkl, ruta .json) del modelo de una provincia."""
    nombre = slug(provincia)
    return (f"{MODELS_DIR}/sarima_{nombre}.pkl", f"{MODELS_DIR}/sarima_{nombre}.json")


def ruta_metricas(provincia: str) -> str:
    """Ruta del JSON con las metricas de los 3 modelos (lo usa la API)."""
    return f"{MODELS_DIR}/metricas_{slug(provincia)}.json"


def cargar_modelo(provincia: str = "Santa Fe"):
    """Carga el modelo guardado de una provincia y sus metadatos."""
    ruta_modelo, ruta_meta = rutas_modelo(provincia)
    with open(ruta_modelo, "rb") as f:
        modelo = pickle.load(f)
    with open(ruta_meta, encoding="utf-8") as f:
        metadatos = json.load(f)
    return modelo, metadatos


def cargar_metricas(provincia: str = "Santa Fe") -> dict:
    """Carga las metricas de los 3 modelos (base, SARIMA, Prophet) para la API."""
    with open(ruta_metricas(provincia), encoding="utf-8") as f:
        return json.load(f)


def pronostico_futuro(provincia: str = "Santa Fe", meses: int = 6) -> pd.Series:
    """Pronostica los proximos `meses` meses usando el modelo guardado."""
    modelo, _ = cargar_modelo(provincia)
    pred = modelo.forecast(steps=meses)
    pred.index.name = "fecha"
    return pred.round(0).astype(int)
