"""
Configuracion de pytest para la API.

Los tests necesitan el modelo y las metricas ya generados. Si no existen, los
creamos una sola vez por sesion corriendo el pipeline de entrenamiento.
"""

from __future__ import annotations

import os

import pytest

ARTEFACTOS = ["models/sarima_santa_fe.pkl", "models/metricas_santa_fe.json"]


@pytest.fixture(scope="session", autouse=True)
def asegurar_artefactos():
    """Garantiza que existan el modelo y las metricas antes de testear la API."""
    if not all(os.path.exists(p) for p in ARTEFACTOS):
        from src.train import comparar, entrenar_final

        comparar("Santa Fe", test_meses=12)
        entrenar_final("Santa Fe", test_meses=12)
