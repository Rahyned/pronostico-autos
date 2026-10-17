# Imagen de produccion para la API (pronostico-autos).
# Liviana: solo dependencias de runtime (sin matplotlib/prophet/jupyterlab).
FROM python:3.12-slim

# Evita que Python escriba .pyc y fuerza logs sin buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalamos dependencias primero (mejor cache de capas).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Codigo y modelo entrenado (el .pkl esta versionado en git).
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Render asigna el puerto en $PORT (por defecto 8000 en local).
ENV PORT=8000
EXPOSE 8000

# Forma shell para expandir ${PORT}.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
