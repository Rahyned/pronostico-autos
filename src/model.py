"""
Definicion del modelo y su version liviana serializable.

`ModeloSARIMA` vive en su propio modulo (y no dentro de train.py) para que el
pickle guarde la ruta estable `src.model.ModeloSARIMA`. Si la clase se definiera
en train.py y lo corriéramos con `python -m src.train`, el modulo seria '__main__'
y el pickle no se podria cargar desde la API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Hiperparametros del SARIMA elegido (ver explicacion en src/train.py).
SARIMA_ORDER = (1, 1, 2)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)


class ModeloSARIMA:
    """
    Envoltorio liviano de un SARIMA ajustado.

    Un SARIMAXResults completo pesa ~39 MB (guarda todo el estado interno) y con
    `remove_data()` baja a ~6 MB pero se rompe el `forecast`. Guardando solo los
    parametros + la serie (unos pocos KB) y reconstruyendo con `.smooth(params)`
    obtenemos pronosticos identicos y un archivo ~2500 veces mas chico.
    """

    def __init__(self, valores, fechas, params, order=SARIMA_ORDER,
                 seasonal_order=SARIMA_SEASONAL_ORDER):
        self.serie = pd.Series(np.asarray(valores, dtype=float),
                               index=pd.DatetimeIndex(fechas))
        self.params = np.asarray(params, dtype=float)
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self._resultado = None  # cache: se arma recien al primer pronostico

    def _result(self):
        """Reconstruye el resultado filtrado con los parametros guardados."""
        if self._resultado is None:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            modelo = SARIMAX(self.serie, order=self.order,
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False)
            self._resultado = modelo.smooth(self.params)
        return self._resultado

    def forecast(self, steps: int) -> pd.Series:
        """Pronostica `steps` meses hacia adelante."""
        pred = self._result().forecast(steps=steps)
        # Reconstruimos el indice futuro (mes a mes) por las dudas de que el
        # indice guardado haya perdido la frecuencia al serializar.
        inicio = self.serie.index[-1] + pd.DateOffset(months=1)
        pred.index = pd.date_range(inicio, periods=steps, freq="MS")
        pred.name = "SARIMA"
        return pred
