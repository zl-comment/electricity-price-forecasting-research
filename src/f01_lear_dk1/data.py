"""Build the 24-period synthetic civil-day frames required by paper-era LEAR."""

from pathlib import Path

import numpy as np
import pandas as pd

from epf_harness.dk1_audit import load_table
from epf_harness.dk1_coupling import add_delivery_day


def normalise_day_length(values: np.ndarray, target_length: int) -> np.ndarray:
    """Use S2's frozen DST policy: truncate, or repeat the last value."""
    if len(values) == target_length:
        return values
    if len(values) > target_length:
        return values[:target_length]
    return np.concatenate([values, np.repeat(values[-1], target_length - len(values))])


def build_synthetic_index(days: list) -> pd.DatetimeIndex:
    """Return a continuous naive index whose dates denote civil delivery days, not UTC."""
    start = pd.Timestamp(days[0])
    expected = pd.date_range(start, pd.Timestamp(days[-1]), freq="D").strftime("%Y-%m-%d").tolist()
    if days != expected:
        raise ValueError("Synthetic civil-day index requires every calendar delivery day")
    return pd.date_range(start, periods=len(days) * 24, freq="h")


def _target_values(frame: pd.DataFrame, target: dict) -> dict:
    values = {}
    for day, group in frame.groupby("delivery_day", sort=True):
        raw = group[target["column"]].to_numpy(dtype=float)
        if target["aggregation"] == "mean":
            if len(raw) % 4:
                raise ValueError(f"{day}: quarter-hour target length {len(raw)} is not divisible by four")
            raw = pd.Series(raw).groupby(np.arange(len(raw)) // 4).mean().to_numpy()
        elif target["aggregation"] != "none":
            raise ValueError(f"Unknown target aggregation: {target['aggregation']}")
        if not np.isfinite(raw).all():
            raise ValueError(f"{day}: target contains missing or non-finite values")
        values[day] = raw
    return values


def _window_days(config: dict) -> list:
    return pd.date_range(
        config["window"]["first_delivery_day"], config["window"]["last_delivery_day"], freq="D"
    ).strftime("%Y-%m-%d").tolist()


def _day_map(index: pd.DatetimeIndex, days: list, source_lengths: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "delivery_day": np.repeat(days, 24),
            "hour": np.tile(np.arange(24), len(days)),
            "source_hours": np.repeat(source_lengths, 24),
            "dst_length_adjusted": np.repeat(np.asarray(source_lengths) != 24, 24),
            "index_semantics": "synthetic_civil_day_not_utc",
        },
        index=index,
    )


def load_target_frame(root: Path, config: dict, target: str) -> tuple:
    """Load one target, retain all window days, normalise DST, and return its trace map."""
    if target not in config["targets"]:
        raise ValueError(f"Unknown target: {target}")
    target_config = config["targets"][target]
    settings = config["datasets"][target_config["source"]]
    raw = load_table(root, settings)
    area = config["data"]["primary_price_area"]
    raw = raw[raw["PriceArea"] == area].sort_values(settings["time_column"])
    raw = add_delivery_day(raw, settings["time_column"], config["protocol"]["civil_timezone"])
    raw = raw[raw["delivery_day"].isin(_window_days(config))]
    by_day = _target_values(raw, target_config)
    days = _window_days(config)
    if sorted(by_day) != days:
        missing = sorted(set(days) - set(by_day))
        raise ValueError(f"Target {target} is missing delivery days: {missing}")
    source_lengths = [len(by_day[day]) for day in days]
    periods = config["protocol"]["periods_per_day"]
    values = np.concatenate([normalise_day_length(by_day[day], periods) for day in days])
    index = build_synthetic_index(days)
    frame = pd.DataFrame({"Price": values}, index=index)
    mapping = _day_map(index, days, source_lengths)
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Target {target} frame contains missing or non-finite values")
    return frame, mapping


def forecast_timestamp(day_map: pd.DataFrame, forecast_day: str) -> pd.Timestamp:
    rows = day_map[day_map["delivery_day"] == forecast_day]
    if len(rows) != 24 or rows["hour"].tolist() != list(range(24)):
        raise ValueError(f"Synthetic day map is invalid for {forecast_day}")
    return rows.index[0]


def available_at_gate(frame: pd.DataFrame, forecast_day: str, day_map: pd.DataFrame) -> pd.DataFrame:
    """Expose historical realised prices and a blank 24-hour delivery day."""
    start = forecast_timestamp(day_map, forecast_day)
    end = start + pd.Timedelta(hours=23)
    available = frame.loc[:end].copy()
    available.loc[start:end, "Price"] = np.nan
    if not available.loc[start:end, "Price"].isna().all():
        raise ValueError(f"Forecast day {forecast_day} contains realised prices")
    if available.loc[: start - pd.Timedelta(hours=1), "Price"].isna().any():
        raise ValueError(f"History before {forecast_day} contains missing prices")
    mapped = day_map.loc[available.index, "delivery_day"]
    realised_future = (mapped >= forecast_day) & available["Price"].notna()
    if realised_future.any():
        raise ValueError(f"Frame exposes realised prices on or after {forecast_day}")
    return available
