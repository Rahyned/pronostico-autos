"""
Carga y limpieza de los datos de transferencias de la DNRPA.

Dejamos acá las funciones reutilizables para que las usen el notebook, el
entrenamiento (src/train.py) y cualquier otro script, sin repetir código.

Idea central: pasar de un CSV "largo" (una fila por provincia/mes/tipo) a una
serie temporal mensual por provincia, con un índice de fechas continuo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Ruta del CSV descargado por src/download_data.py.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "transferencias-automotores.csv"

# Nos quedamos solo con las transferencias de automotores (no motos, etc.).
TIPO_VEHICULO = "Automotores"


def load_raw(path: Path = CSV_PATH) -> pd.DataFrame:
    """Lee el CSV crudo. Usamos utf-8-sig porque el archivo tiene BOM."""
    if not path.exists():
        raise FileNotFoundError(
            f"No encontré {path}. Corré primero:  python src/download_data.py"
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame crudo y devuelve uno tidy con columnas:
    fecha (datetime), provincia (str) y cantidad (int).

    Pasos:
      1. Renombrar columnas a nombres cortos.
      2. Filtrar solo automotores.
      3. Construir la fecha (año-mes-01).
      4. Tipar y agrupar (por si hay filas repetidas).
    """
    df = df.rename(
        columns={
            "anio_transferencia": "anio",
            "mes_transferencia": "mes",
            "provincia_transferencia": "provincia",
            "cantidad_transferencias": "cantidad",
        }
    )

    # 2. Solo el tipo de vehículo que nos interesa.
    df = df[df["tipo_vehiculo"] == TIPO_VEHICULO].copy()

    # 3. Fecha: primer día del mes. Es lo que convierte "2025"/"6" en 2025-06-01.
    df["fecha"] = pd.to_datetime(
        dict(year=df["anio"].astype(int), month=df["mes"].astype(int), day=1)
    )

    # 4. Tipos y orden. Agrupamos por si hubiera más de una fila por provincia/mes.
    df["cantidad"] = df["cantidad"].astype(int)
    df = (
        df.groupby(["provincia", "fecha"], as_index=False)["cantidad"].sum()
        .sort_values(["provincia", "fecha"])
        .reset_index(drop=True)
    )
    return df


def load_clean(path: Path = CSV_PATH) -> pd.DataFrame:
    """Atajo: leer + limpiar en un solo paso."""
    return clean(load_raw(path))


def get_series(df: pd.DataFrame, provincia: str = "Santa Fe") -> pd.Series:
    """
    Devuelve la serie mensual de una provincia, con índice de fechas continuo.

    No rellenamos meses faltantes: si la serie tiene huecos, tiramos un error
    claro con las fechas que faltan. Preferimos que falle a modelar sobre datos
    incompletos en silencio.
    """
    serie = (
        df[df["provincia"] == provincia]
        .set_index("fecha")["cantidad"]
        .sort_index()
    )
    if serie.empty:
        raise ValueError(f"La provincia '{provincia}' no aparece en los datos.")

    # Chequeamos que no falte ningún mes entre el primero y el último.
    rango = pd.date_range(serie.index.min(), serie.index.max(), freq="MS")
    faltantes = rango.difference(serie.index)
    if len(faltantes):
        lista = ", ".join(f.strftime("%Y-%m") for f in faltantes)
        raise ValueError(
            f"A '{provincia}' le faltan {len(faltantes)} mes(es): {lista}. "
            "Revisá los datos antes de modelar."
        )

    serie = serie.astype(int)
    serie.name = provincia
    return serie


def split_train_test(serie: pd.Series, test_months: int = 12) -> tuple[pd.Series, pd.Series]:
    """
    Divide la serie en train/test de forma cronológica.

    Los `test_months` últimos meses van a test; el resto a train.
    Con datos hasta 2026-09 y test_months=12, el test es 2025-10 .. 2026-09.
    """
    if test_months <= 0 or test_months >= len(serie):
        raise ValueError("test_months debe ser mayor que 0 y menor que la serie.")
    train = serie.iloc[:-test_months]
    test = serie.iloc[-test_months:]
    return train, test


if __name__ == "__main__":
    # Pequeña demo por consola: útil para chequear que todo carga bien.
    datos = load_clean()
    print(f"Filas limpias: {len(datos)}")
    print(f"Provincias: {datos['provincia'].nunique()}")
    print(f"Rango de fechas: {datos['fecha'].min().date()} a {datos['fecha'].max().date()}")

    serie = get_series(datos, "Santa Fe")
    train, test = split_train_test(serie, test_months=12)
    print(f"\nSanta Fe: {len(serie)} meses en total")
    print(f"  train: {len(train)}  ({train.index.min().date()} a {train.index.max().date()})")
    print(f"  test:  {len(test)}   ({test.index.min().date()} a {test.index.max().date()})")
    print("\nÚltimos 6 valores reales:")
    print(serie.tail(6).to_string())
