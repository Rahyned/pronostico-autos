"""
Tests basicos de la API (Etapa 5).

Correr con:
    pytest
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from api.main import app

cliente = TestClient(app)

FORMATO_MES = re.compile(r"^\d{4}-\d{2}$")


def test_health():
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_raiz_lista_endpoints():
    r = cliente.get("/")
    assert r.status_code == 200
    assert "/metricas" in r.json()["endpoints"]


def test_pronostico_devuelve_meses_y_valores():
    r = cliente.get("/pronostico", params={"provincia": "Santa Fe", "meses": 6})
    assert r.status_code == 200
    cuerpo = r.json()

    assert cuerpo["provincia"] == "Santa Fe"
    assert cuerpo["modelo"] == "SARIMA"
    assert len(cuerpo["pronostico"]) == 6

    for item in cuerpo["pronostico"]:
        assert set(item) == {"mes", "valor"}
        assert FORMATO_MES.match(item["mes"])
        assert isinstance(item["valor"], int)
        assert item["valor"] > 0


def test_pronostico_arranca_al_mes_siguiente():
    # Los datos llegan hasta 2026-09, asi que el primer pronostico es 2026-10.
    r = cliente.get("/pronostico", params={"meses": 1})
    assert r.status_code == 200
    assert r.json()["pronostico"][0]["mes"] == "2026-10"


def test_pronostico_respeta_cantidad_de_meses():
    r = cliente.get("/pronostico", params={"meses": 3})
    assert r.status_code == 200
    assert len(r.json()["pronostico"]) == 3


def test_pronostico_meses_invalidos():
    # 0 y 999 estan fuera del rango permitido (1..36) -> error de validacion.
    assert cliente.get("/pronostico", params={"meses": 0}).status_code == 422
    assert cliente.get("/pronostico", params={"meses": 999}).status_code == 422


def test_pronostico_provincia_desconocida():
    r = cliente.get("/pronostico", params={"provincia": "Narnia"})
    assert r.status_code == 404


def test_metricas_devuelve_los_tres_modelos():
    r = cliente.get("/metricas", params={"provincia": "Santa Fe"})
    assert r.status_code == 200
    cuerpo = r.json()

    assert cuerpo["provincia"] == "Santa Fe"
    modelos = {m["modelo"] for m in cuerpo["metricas"]}
    assert len(cuerpo["metricas"]) == 3
    assert any("SARIMA" in m for m in modelos)
    assert "base estacional" in modelos
    assert "Prophet" in modelos

    for m in cuerpo["metricas"]:
        assert {"modelo", "MAE", "MAPE"} <= set(m)
        assert m["MAE"] >= 0
        assert m["MAPE"] >= 0


def test_metricas_provincia_desconocida():
    r = cliente.get("/metricas", params={"provincia": "Narnia"})
    assert r.status_code == 404
