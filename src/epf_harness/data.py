"""Load and validate the canonical EPF-DE inputs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EPFData:
    market: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    author: pd.DataFrame


def _read_hourly_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "timestamp"
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps in {path}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"Non-monotonic timestamps in {path}")
    return frame


def _validate_index(frame: pd.DataFrame, settings: dict, prefix: str) -> None:
    expected = pd.date_range(
        settings[f"{prefix}_start"], settings[f"{prefix}_end"], freq="h"
    )
    if len(frame) != settings[f"expected_{prefix}_rows"]:
        raise ValueError(f"Unexpected {prefix} row count: {len(frame)}")
    if not frame.index.equals(expected):
        raise ValueError(f"Unexpected {prefix} time index")


def load_epf_data(repository_root: Path, settings: dict) -> EPFData:
    market = _read_hourly_csv(repository_root / settings["market_csv"])
    author = _read_hourly_csv(repository_root / settings["author_csv"])
    _validate_index(market, settings, "market")
    _validate_index(author, settings, "author")
    if list(market.columns) != settings["market_columns"]:
        raise ValueError(f"Unexpected market columns: {list(market.columns)}")
    if list(author.columns) != settings["author_columns_all"]:
        raise ValueError(f"Unexpected author columns: {list(author.columns)}")
    if market.isna().any().any() or author.isna().any().any():
        raise ValueError("Input data contain missing values")

    market = market.rename(columns=settings["epftoolbox_column_mapping"])
    test_start = pd.Timestamp(settings["test_start"])
    test_end = pd.Timestamp(settings["test_end"])
    train = market.loc[: test_start - pd.Timedelta(hours=1)].copy()
    test = market.loc[test_start:test_end].copy()
    if not np.array_equal(test["Price"].to_numpy(), author["Real price"].to_numpy()):
        raise ValueError("Author real prices do not match DE.csv exactly")
    return EPFData(market=market, train=train, test=test, author=author)


def build_available_data(data: EPFData, forecast_date: pd.Timestamp, hours: int) -> pd.DataFrame:
    day_end = forecast_date + pd.Timedelta(hours=hours - 1)
    available = data.market.loc[:day_end].copy()
    available.loc[forecast_date:day_end, "Price"] = np.nan
    if available.index[-1] != day_end:
        raise ValueError(f"Missing exogenous inputs for {forecast_date}")
    if available.loc[forecast_date:day_end, "Price"].notna().any():
        raise ValueError(f"Future prices visible for {forecast_date}")
    if available.loc[forecast_date:day_end].drop(columns="Price").isna().any().any():
        raise ValueError(f"Missing day-ahead exogenous inputs for {forecast_date}")
    return available
