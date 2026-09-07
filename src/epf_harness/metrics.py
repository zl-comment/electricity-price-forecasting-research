"""Metrics used by the EPF-DE acceptance test."""

import numpy as np
import pandas as pd
from epftoolbox.evaluation import DM, MAE, RMSE, rMAE, sMAPE


def forecast_metrics(
    actual: pd.Series, forecast: pd.Series, rmae_seasonality: str
) -> dict:
    return {
        "mae": float(MAE(actual, forecast)),
        "rmse": float(RMSE(actual, forecast)),
        "smape": float(sMAPE(actual, forecast)),
        "rmae": float(rMAE(actual, forecast, m=rmae_seasonality)),
    }


def dm_p_value(
    actual: pd.Series,
    forecast_1: pd.Series,
    forecast_2: pd.Series,
    hours_per_day: int,
    norm: int,
    version: str,
):
    shape = (-1, hours_per_day)
    value = DM(
        actual.to_numpy().reshape(shape),
        forecast_1.to_numpy().reshape(shape),
        forecast_2.to_numpy().reshape(shape),
        norm=norm,
        version=version,
    )
    values = np.asarray(value, dtype=float)
    if not np.isfinite(values).all():
        return None
    return values.tolist() if values.ndim else float(values)


def absolute_bias_percent(value: float, reference: float) -> float:
    return float(abs(value - reference) / abs(reference) * 100.0)


def point_difference_statistics(local: pd.Series, author: pd.Series) -> dict:
    absolute_difference = (local - author).abs()
    max_timestamp = absolute_difference.idxmax()
    return {
        "median": float(absolute_difference.median()),
        "percentile_99": float(absolute_difference.quantile(0.99)),
        "maximum": float(absolute_difference.loc[max_timestamp]),
        "maximum_timestamp": max_timestamp.isoformat(),
    }
