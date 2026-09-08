"""Daily F08 LightGBM rolling estimation on DK1."""

import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from f01_lear_dk1.backtest import forecast_timestamp
from f01_lear_dk1.data import available_at_gate

from .model import build_matrices, fit_predict_hourly


_WORKER_CONTEXT = {}


def _forecast_single_day(delivery_day: str) -> np.ndarray:
    context = _WORKER_CONTEXT
    available = available_at_gate(context["frame"], delivery_day, context["day_map"])
    timestamp = forecast_timestamp(context["day_map"], delivery_day)
    matrices = build_matrices(context["lear_class"], available, timestamp,
                              context["calibration_window"])
    return fit_predict_hourly(*matrices, context["params"], context["seed"])


def run_target(lear_class, frame: pd.DataFrame, day_map: pd.DataFrame, test_days: list,
               config: dict) -> pd.DataFrame:
    """Forecast every requested day using the fixed raw LEAR matrix and LightGBM parameters."""
    periods = config["protocol"]["periods_per_day"]
    if config["protocol"]["recalibration_days"] != 1:
        raise ValueError("F08 requires daily recalibration")
    _WORKER_CONTEXT.update(
        lear_class=lear_class, frame=frame, day_map=day_map,
        calibration_window=config["protocol"]["calibration_window_days"],
        params=config["model"]["params"], seed=config["protocol"]["random_seed"],
    )
    rows, started = [], time.monotonic()
    workers = config["model"]["worker_processes"]
    with get_context("fork").Pool(processes=workers) as pool:
        predictions = pool.imap(_forecast_single_day, test_days, chunksize=1)
        for position, (day, values) in enumerate(zip(test_days, predictions), 1):
            actual = frame.loc[day_map.index[day_map["delivery_day"] == day], "Price"].to_numpy()
            rows.extend({"delivery_day": day, "hour": hour, "forecast": float(values[hour]),
                         "actual": float(actual[hour])} for hour in range(periods))
            elapsed = (time.monotonic() - started) / 60
            print(f"F08 {position}/{len(test_days)} {day} elapsed={elapsed:.1f}min", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != len(test_days) * periods or not np.isfinite(result[["forecast", "actual"]]).all().all():
        raise ValueError("F08 rolling forecast output is incomplete")
    return result
