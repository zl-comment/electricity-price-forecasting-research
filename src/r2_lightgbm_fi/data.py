"""Load and split the authors' published hourly Finnish aFRR dataset."""

import hashlib
from pathlib import Path

import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hourly_grid(path: Path, config: dict) -> pd.DataFrame:
    if file_sha256(path) != config["sha256"]:
        raise ValueError(f"Unexpected Finnish aFRR CSV checksum: {path}")
    frame = pd.read_csv(path)
    time_column = config["time_column"]
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    if frame[time_column].duplicated().any():
        raise ValueError("Finnish aFRR CSV contains duplicate timestamps")
    frame = frame.set_index(time_column).sort_index()
    if frame.isna().sum().sum() != 0:
        raise ValueError("Published v2 CSV must not contain missing values before reindexing")
    grid = pd.date_range(config["grid_start_utc"], config["grid_end_utc"], freq="h")
    if len(frame) != 6456 or len(grid) != 6457:
        raise ValueError(f"Unexpected raw/grid row counts: {len(frame)}/{len(grid)}")
    if not frame.index.isin(grid).all():
        raise ValueError("Finnish aFRR timestamps fall outside the declared hourly grid")
    result = frame.reindex(grid)
    result.index.name = time_column
    missing = result.index[result.isna().all(axis=1)]
    expected = pd.Timestamp(config["missing_grid_timestamp_utc"])
    if len(missing) != 1 or missing[0] != expected:
        raise ValueError(f"Unexpected missing grid timestamp: {missing.tolist()}")
    return result


def chronological_splits(frame: pd.DataFrame, protocol: dict) -> dict:
    modeled = frame.iloc[protocol["head_rows"]:]
    train_end = protocol["train_rows"]
    validation_end = train_end + protocol["validation_rows"]
    train = modeled.iloc[:train_end]
    validation = modeled.iloc[train_end:validation_end]
    test = modeled.iloc[validation_end:]
    expected = (protocol["train_rows"], protocol["validation_rows"], protocol["test_rows"])
    actual = (len(train), len(validation), len(test))
    if actual != expected or len(modeled) != sum(expected):
        raise ValueError(f"Unexpected chronological split sizes: {actual}")
    if len(test) % protocol["forecast_horizon_hours"] != 0:
        raise ValueError("Test block is not divisible by the 48-hour horizon")
    return {"modeled": modeled, "train": train, "validation": validation, "test": test}


def split_metadata(frame: pd.DataFrame, splits: dict) -> dict:
    def describe(part: pd.DataFrame) -> dict:
        return {"rows": len(part), "start_utc": part.index[0].isoformat(),
                "end_utc": part.index[-1].isoformat()}

    missing = frame.index[frame.isna().all(axis=1)]
    return {
        "raw_rows": len(frame) - len(missing),
        "complete_grid_rows": len(frame),
        "missing_grid_timestamp_utc": missing[0].isoformat(),
        "missing_grid_position_zero_based": int(frame.index.get_loc(missing[0])),
        "modeled": describe(splits["modeled"]),
        "train": describe(splits["train"]),
        "validation": describe(splits["validation"]),
        "test": describe(splits["test"]),
    }
