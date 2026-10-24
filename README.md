# Pronóstico de transferencias de autos

**Modelo de series de tiempo que pronostica las transferencias mensuales de automotores en Santa Fe, publicado como API.**

🔗 **API:** https://pronostico-autos.onrender.com · **Documentación interactiva:** https://pronostico-autos.onrender.com/docs

## El problema

Estimar cuántos autos se van a transferir por mes a partir de los datos oficiales de la DNRPA (2000-01 a 2026-09).

## Resultados

Test de 12 meses (2025-10 a 2026-09), con el modelo ajustado solo con datos hasta 2025-09:

| Modelo | MAE | MAPE |
|---|---|---|
| **SARIMA(1,1,2)(1,1,1,12)** | **687,74** | **5,29 %** |
| Base estacional (seasonal naive) | 868,25 | 6,62 % |
| Prophet | 907,12 | 6,70 % |

![Comparación de modelos](reports/comparacion_santa_fe.png)
![Histórico y pronóstico](reports/pronostico_santa_fe.png)

Para producción, el SARIMA se reentrena con toda la serie. En lugar del objeto completo de statsmodels (~39 MB), se guardan solo los parámetros y la serie, y el modelo se reconstruye con `SARIMAX.smooth`: pronósticos idénticos en **~9 KB**.

## API

| Endpoint | Descripción |
|---|---|
| `GET /health` | Chequeo de vida |
| `GET /pronostico?provincia=Santa%20Fe&meses=6` | Pronóstico de los próximos meses |
| `GET /metricas?provincia=Santa%20Fe` | MAE y MAPE de los tres modelos |

La consume el Panel personal del autor (server-to-server). En el plan gratuito de Render el servicio se duerme tras ~15 minutos sin uso, así que la primera respuesta puede tardar 30–60 s.

## Stack

Python · pandas · statsmodels · Prophet · FastAPI · pytest · Docker · Render.

## Cómo correrlo

```bash
pip install -r requirements.txt
python src/download_data.py        # descarga el CSV oficial
python -m src.train                # compara modelos y guarda el final
uvicorn api.main:app --reload      # API en http://127.0.0.1:8000/docs
pytest
```

Con Docker: `docker build -t pronostico-autos . && docker run -p 8000:8000 pronostico-autos`.

## Estado

Desplegado. Pendiente: extender el pipeline a todas las provincias (hoy solo Santa Fe).

## Autor

Lautaro Rodríguez · [GitHub](https://github.com/Rahyned) · Datos: [DNRPA – datos.jus.gob.ar](https://datos.jus.gob.ar/dataset/1ab0cc03-ab9b-4520-975c-5757d87d1061)
