"""S0 structural checks on the Energinet DK1 multi-market tables."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_table(root: Path, settings: dict, price_area: str) -> pd.DataFrame:
    frame = pd.read_csv(root / settings["csv"], parse_dates=[settings["time_column"]])
    if list(frame[settings["key_columns"][1]].unique()) != [price_area]:
        raise ValueError(f"Unexpected price areas in {settings['csv']}")
    return frame.sort_values(settings["key_columns"]).reset_index(drop=True)


def check_primary_key(frame: pd.DataFrame, settings: dict) -> dict:
    duplicated = frame.duplicated(subset=settings["key_columns"])
    time_column = frame[settings["time_column"]]
    return {
        "rows": int(len(frame)),
        "duplicate_keys": int(duplicated.sum()),
        "timestamps_naive": bool(time_column.dt.tz is None),
        "monotonic_after_sort": bool(time_column.is_monotonic_increasing),
    }


def check_resolution(frame: pd.DataFrame, settings: dict, seconds: dict) -> dict:
    time_column = settings["time_column"]
    stamps = pd.Index(frame[time_column].unique()).sort_values()
    steps = pd.Series(stamps).diff().dropna().dt.total_seconds()
    declared = seconds[settings["declared_resolution"]]
    expected = pd.date_range(stamps.min(), stamps.max(), freq=pd.Timedelta(seconds=declared))
    return {
        "declared_resolution_seconds": declared,
        "modal_step_seconds": int(steps.mode().iloc[0]),
        "distinct_step_seconds": sorted({int(v) for v in steps.unique()}),
        "unique_timestamps": int(len(stamps)),
        "expected_timestamps": int(len(expected)),
        "missing_timestamps": int(len(expected) - len(stamps)),
        "first_utc": stamps.min().isoformat(),
        "last_utc": stamps.max().isoformat(),
    }


def check_daylight_saving(frame: pd.DataFrame, settings: dict, daylight: dict) -> dict:
    time_column = settings["time_column"]
    grid = frame[[time_column, settings["local_time_column"]]].drop_duplicates()
    stamps = pd.Index(grid[time_column]).sort_values()
    switches = {}
    for instant in daylight["switch_instants_utc"]:
        moment = pd.Timestamp(instant)
        window = stamps[(stamps >= moment - pd.Timedelta(hours=3)) & (stamps < moment + pd.Timedelta(hours=3))]
        switches[instant] = {
            "utc_stamps_in_window": int(len(window)),
            "utc_duplicates_in_window": int(len(window) - window.nunique()),
        }
    return {
        "civil_timezone": daylight["civil_timezone"],
        "utc_duplicates": int(len(stamps) - stamps.nunique()),
        "local_clock_duplicates": int(grid[settings["local_time_column"]].duplicated().sum()),
        "switch_windows": switches,
    }


def check_missing_values(frame: pd.DataFrame, settings: dict) -> dict:
    counts = frame[settings["value_columns"]].isna().sum()
    return {
        "missing_by_column": {k: int(v) for k, v in counts.items() if v},
        "rows_with_any_missing": int(frame[settings["value_columns"]].isna().any(axis=1).sum()),
    }


def audit_dataset(root: Path, settings: dict, config: dict) -> dict:
    frame = load_table(root, settings, config["data"]["price_area"])
    return {
        "primary_key": check_primary_key(frame, settings),
        "resolution": check_resolution(frame, settings, config["resolution_seconds"]),
        "daylight_saving": check_daylight_saving(frame, settings, config["daylight_saving"]),
        "missing_values": check_missing_values(frame, settings),
    }


def common_window(audits: dict) -> dict:
    starts = [pd.Timestamp(a["resolution"]["first_utc"]) for a in audits.values()]
    ends = [pd.Timestamp(a["resolution"]["last_utc"]) for a in audits.values()]
    start, end = max(starts), min(ends)
    return {
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "hours": float((end - start).total_seconds() / 3600),
        "limited_by_start": [n for n, a in audits.items() if pd.Timestamp(a["resolution"]["first_utc"]) == start],
        "limited_by_end": [n for n, a in audits.items() if pd.Timestamp(a["resolution"]["last_utc"]) == end],
        "end_shortfall_hours": {
            name: float((max(ends) - pd.Timestamp(a["resolution"]["last_utc"])).total_seconds() / 3600)
            for name, a in audits.items()
        },
    }


def collect_findings(audits: dict) -> list:
    findings = []
    for name, audit in audits.items():
        key, resolution, daylight = audit["primary_key"], audit["resolution"], audit["daylight_saving"]
        if key["duplicate_keys"]:
            findings.append((name, "A", "duplicate_primary_key", str(key["duplicate_keys"])))
        if resolution["missing_timestamps"]:
            findings.append((name, "A", "missing_timestamps", str(resolution["missing_timestamps"])))
        if len(resolution["distinct_step_seconds"]) > 1:
            findings.append((name, "B", "irregular_step_seconds", str(resolution["distinct_step_seconds"])))
        if resolution["modal_step_seconds"] != resolution["declared_resolution_seconds"]:
            findings.append((name, "B", "resolution_mismatch", str(resolution["modal_step_seconds"])))
        if daylight["utc_duplicates"]:
            findings.append((name, "A", "utc_duplicates", str(daylight["utc_duplicates"])))
        if daylight["local_clock_duplicates"]:
            findings.append((name, "A", "local_clock_not_a_key", str(daylight["local_clock_duplicates"])))
        for column, count in audit["missing_values"]["missing_by_column"].items():
            findings.append((name, "D", f"missing_values:{column}", str(count)))
    return findings
