"""
Descarga el CSV oficial de transferencias de automotores de la DNRPA.

La idea es simple: si el archivo ya existe en data/, no lo baja de nuevo
(así evitamos descargar 375 KB cada vez). Con --force forzamos la descarga.

Uso:
    python src/download_data.py
    python src/download_data.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

# --- Configuración -----------------------------------------------------------

# URL oficial: dataset de la DNRPA en datos.jus.gob.ar (transferencias 2000-01 a 2026-09).
URL = (
    "https://datos.jus.gob.ar/dataset/1ab0cc03-ab9b-4520-975c-5757d87d1061/"
    "resource/7912aa2b-7e9a-49dd-8c14-0570ca043d27/download/"
    "estadistica-transferencias-automotores-2000-01-2026-09.csv"
)

# Carpeta data/ relativa a la raíz del repo (este archivo está en src/).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DESTINO = DATA_DIR / "transferencias-automotores.csv"


def descargar(destino: Path = DESTINO, forzar: bool = False) -> Path:
    """Descarga el CSV a `destino`. Si ya existe y no se fuerza, no hace nada."""
    if destino.exists() and not forzar:
        print(f"Ya existe: {destino}  (usá --force para volver a bajarlo)")
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL}")
    print(f"  -> {destino}")

    # Streaming para no cargar todo el archivo en memoria de golpe.
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(destino, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    size_kb = destino.stat().st_size / 1024
    print(f"Listo: {size_kb:.0f} KB")
    return destino


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga el CSV de transferencias de la DNRPA.")
    parser.add_argument("--force", action="store_true", help="Vuelve a descargar aunque ya exista.")
    args = parser.parse_args()

    try:
        destino = descargar(forzar=args.force)
    except requests.RequestException as e:
        print(f"Error al descargar: {e}", file=sys.stderr)
        return 1

    # Chequeo rápido: contamos las filas del CSV bajado.
    with open(destino, encoding="utf-8-sig") as f:
        filas = sum(1 for _ in f) - 1  # -1 por el encabezado
    print(f"Filas (sin encabezado): {filas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
