"""Daily F01 LEAR rolling recalibration on synthetic DK1 civil days."""

import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from .data import available_at_gate


_WORKER_CONTEXT = {}


def forecast_timestamp(day_map: pd.DataFrame, delivery_day: str) -> pd.Timestamp:
    stamps = day_map.index[day_map["delivery_day"] == delivery_day]
    if len(stamps) != 24 or stamps[0].hour != 0:
        raise ValueError(f"{delivery_day}: synthetic mapping does not contain one 24-hour day")
    return stamps[0]


def split_lear_frames(frame: pd.DataFrame, forecast_day: pd.Timestamp,
                      calibration_window: int) -> tuple:
    """Mirror epftoolbox's rolling slices exactly."""
    df_train = frame.loc[:forecast_day - pd.Timedelta(hours=1)]
    df_train = df_train.iloc[-calibration_window * 24:]
    df_test = frame.loc[forecast_day - pd.Timedelta(weeks=2):]
    if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
        raise ValueError("LEAR train and test slices must start at hour zero")
    return df_train, df_test


def raw_design_matrices(lear_class, frame: pd.DataFrame, day_map: pd.DataFrame,
                        forecast_day: str, calibration_window: int) -> tuple:
    """Return the unscaled matrices built by the pinned epftoolbox LEAR source."""
    available = available_at_gate(frame, forecast_day, day_map)
    timestamp = forecast_timestamp(day_map, forecast_day)
    df_train, df_test = split_lear_frames(available, timestamp, calibration_window)
    model = lear_class(calibration_window=calibration_window)
    return model._build_and_split_XYs(df_train=df_train, df_test=df_test, date_test=timestamp)


def design_shape(lear_class, frame: pd.DataFrame, day_map: pd.DataFrame, config: dict) -> dict:
    """Fit the first test day once and assert the price-only LEAR design dimensions."""
    window = config["protocol"]["calibration_window_days"]
    day = config["split"]["first_test_delivery_day"]
    available = available_at_gate(frame, day, day_map)
    timestamp = forecast_timestamp(day_map, day)
    model = lear_class(calibration_window=window)
    model.recalibrate_and_forecast_next_day(
        df=available, next_day_date=timestamp, calibration_window=window
    )
    widths = {fitted.coef_.size for fitted in model.models.values()}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent LEAR feature widths: {sorted(widths)}")
    feature_count = widths.pop()
    samples = window - config["runtime"]["lear_max_lag_days"]
    acceptance = config["acceptance"]
    if feature_count != acceptance["design_feature_count"] or samples != acceptance["effective_training_samples"]:
        raise ValueError(f"Unexpected LEAR design shape: n={samples}, p={feature_count}")
    return {"design_feature_count": feature_count, "effective_training_samples": samples,
            "n_over_p": float(samples / feature_count)}


def _forecast_single_day(delivery_day: str) -> np.ndarray:
    context = _WORKER_CONTEXT
    available = available_at_gate(context["frame"], delivery_day, context["day_map"])
    timestamp = forecast_timestamp(context["day_map"], delivery_day)
    values = context["model"].recalibrate_and_forecast_next_day(
        df=available, next_day_date=timestamp, calibration_window=context["calibration_window"]
    )
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != context["periods_per_day"] or not np.isfinite(values).all():
        raise ValueError(f"Invalid F01 forecast for {delivery_day}")
    return values


def run_target(lear_class, frame: pd.DataFrame, day_map: pd.DataFrame, test_days: list,
               config: dict, runtime: dict) -> pd.DataFrame:
    """Forecast every requested delivery day with one fixed rolling window."""
    periods = config["protocol"]["periods_per_day"]
    window = config["protocol"]["calibration_window_days"]
    if config["protocol"]["recalibration_days"] != 1:
        raise ValueError("F01 requires daily recalibration")
    _WORKER_CONTEXT.update(frame=frame, day_map=day_map, calibration_window=window,
                           periods_per_day=periods, model=lear_class(calibration_window=window))
    rows, started = [], time.monotonic()
    with get_context("fork").Pool(processes=runtime["worker_processes"]) as pool:
        predictions = pool.imap(_forecast_single_day, test_days, chunksize=1)
        for position, (day, values) in enumerate(zip(test_days, predictions), 1):
            actual = frame.loc[day_map.index[day_map["delivery_day"] == day], "Price"].to_numpy()
            rows.extend({"delivery_day": day, "hour": hour, "forecast": float(values[hour]),
                         "actual": float(actual[hour])} for hour in range(periods))
            elapsed = (time.monotonic() - started) / 60
            print(f"F01 {position}/{len(test_days)} {day} elapsed={elapsed:.1f}min", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != len(test_days) * periods or not np.isfinite(result[["forecast", "actual"]]).all().all():
        raise ValueError("F01 rolling forecast output is incomplete")
    return result
