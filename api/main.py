"""
API del proyecto pronostico-autos (Etapa 5).

Expone el modelo SARIMA entrenado como servicio HTTP con FastAPI:

  GET /health       -> chequeo de vida.
  GET /pronostico   -> pronostico de N meses para una provincia.
  GET /metricas     -> MAE y MAPE de base, SARIMA y Prophet (para el panel).

El modelo y las metricas se leen de la carpeta models/ (los genera
`python -m src.train`).

Levantar en desarrollo:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.predict import cargar_metricas, cargar_modelo, pronostico_futuro

app = FastAPI(
    title="pronostico-autos",
    description="Pronostico de transferencias mensuales de automotores en Argentina (DNRPA).",
    version="1.0.0",
)

# Sin CORS a proposito: el Panel consume esta API server-to-server (no desde el
# navegador), asi que no hace falta habilitar origenes cruzados.


@app.get("/")
def raiz() -> dict:
    """Indice simple de endpoints."""
    return {
        "endpoints": ["/health", "/pronostico?provincia=Santa%20Fe&meses=6", "/metricas"],
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    """Chequeo de vida (para monitoreo / deploy)."""
    return {"status": "ok"}


@app.get("/pronostico")
def pronostico(
    provincia: str = Query("Santa Fe", description="Provincia a pronosticar."),
    meses: int = Query(6, ge=1, le=36, description="Cantidad de meses a futuro (1 a 36)."),
) -> dict:
    """
    Devuelve el pronostico de las transferencias para los proximos `meses`.

    Cada elemento trae el mes (YYYY-MM) y el valor estimado (cantidad de
    transferencias, redondeado).
    """
    try:
        serie = pronostico_futuro(provincia, meses)
        _, meta = cargar_modelo(provincia)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No hay modelo entrenado para '{provincia}'. "
                   "Corré: python -m src.train --provincia "
                   f'"{provincia}"',
        )

    return {
        "provincia": provincia,
        "modelo": meta["modelo"],
        "detalle_modelo": f"SARIMA{tuple(meta['order'])}x{tuple(meta['seasonal_order'])}",
        "entrenado_hasta": meta["datos_rango"][1],
        "pronostico": [
            {"mes": fecha.strftime("%Y-%m"), "valor": int(valor)}
            for fecha, valor in serie.items()
        ],
    }


@app.get("/metricas")
def metricas(provincia: str = Query("Santa Fe", description="Provincia del modelo.")) -> dict:
    """
    Devuelve MAE y MAPE de los tres modelos (base estacional, SARIMA y Prophet)
    medidos sobre el test, para compararlos en el panel.
    """
    try:
        return cargar_metricas(provincia)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No hay metricas para '{provincia}'. "
                   "Corré: python -m src.train",
        )
