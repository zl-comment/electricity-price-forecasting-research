#!/usr/bin/env python3
"""Profile the DK1 S3 targets using only the frozen pre-test lookback."""

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
import statsmodels
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def environment_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
    }


def protocol_dates(config: dict) -> tuple:
    protocol = config["protocol"]
    first = date.fromisoformat(protocol["first_delivery_day"])
    last = date.fromisoformat(protocol["last_delivery_day"])
    first_test = date.fromisoformat(protocol["first_test_delivery_day"])
    excluded = {date.fromisoformat(value) for value in protocol["excluded_delivery_days"]}
    included = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    included = [value for value in included if value not in excluded]
    assert last < first_test
    assert len(included) == protocol["expected_delivery_days"]
    assert all(value < first_test for value in included)
    return first, last, first_test, excluded, included


def read_safe_file_parts(path: Path, settings: dict) -> tuple:
    assert path.stat().st_size == settings["expected_file_bytes"]
    with path.open("rb", buffering=0) as stream:
        header_bytes = stream.read(settings["header_bytes"])
    assert header_bytes.endswith(b"\n")
    assert b"\n" not in header_bytes[:-1]
    assert settings["safe_suffix_offset"] >= settings["header_bytes"]
    with path.open("rb", buffering=0) as stream:
        stream.seek(settings["safe_suffix_offset"])
        safe_suffix = stream.read()
    assert hashlib.sha256(safe_suffix).hexdigest() == settings["safe_suffix_sha256"]
    assert safe_suffix and not safe_suffix.startswith((b"\n", b"\r"))
    return header_bytes.decode("utf-8-sig").rstrip("\r\n"), safe_suffix.splitlines()


def read_lookback_csv(path: Path, settings: dict, config: dict) -> pd.DataFrame:
    first, last, first_test, excluded, _ = protocol_dates(config)
    civil_zone = ZoneInfo(config["data"]["civil_timezone"])
    selected, screened_days = [], []
    header, safe_lines = read_safe_file_parts(path, settings)
    columns = next(csv.reader([header]))
    indices = {name: columns.index(name) for name in columns}
    for raw_line in safe_lines:
        utc_stamp = datetime.fromisoformat(raw_line[:19].decode("ascii")).replace(tzinfo=timezone.utc)
        delivery_day = utc_stamp.astimezone(civil_zone).date()
        screened_days.append(delivery_day)
        assert delivery_day < first_test
        if delivery_day < first or delivery_day in excluded:
            continue
        assert delivery_day <= last
        row = next(csv.reader(io.StringIO(raw_line.decode("utf-8"))))
        if row[indices[settings["area_column"]]] == config["data"]["primary_price_area"]:
            selected.append(row)
            if len(selected) == settings["expected_lookback_area_rows"]:
                break
    assert len(selected) == settings["expected_lookback_area_rows"]
    frame = pd.DataFrame(selected, columns=columns)
    frame[settings["time_column"]] = pd.to_datetime(frame[settings["time_column"]], utc=True)
    frame.attrs["read_audit"] = {"direction": "safe_suffix_to_file_end", "selected_rows": len(selected),
                                 "screened_rows": len(screened_days), "minimum_screened_delivery_day": min(screened_days).isoformat(),
                                 "maximum_screened_delivery_day": max(screened_days).isoformat(), "parsed_test_rows": 0,
                                 "bytes_read_from_test_period": 0, "header_bytes": settings["header_bytes"],
                                 "safe_suffix_offset": settings["safe_suffix_offset"],
                                 "safe_suffix_sha256": settings["safe_suffix_sha256"]}
    return frame


def load_targets(config: dict) -> dict:
    frames = {}
    for dataset, settings in config["data"]["datasets"].items():
        path = REPOSITORY_ROOT / config["data"]["root"] / settings["csv"]
        frames[dataset] = read_lookback_csv(path, settings, config)
    targets = {}
    civil_zone = config["data"]["civil_timezone"]
    for name, settings in config["targets"].items():
        source = frames[settings["dataset"]]
        timestamp = source[config["data"]["datasets"][settings["dataset"]]["time_column"]]
        local = timestamp.dt.tz_convert(civil_zone)
        targets[name] = pd.DataFrame({
            "timestamp_utc": timestamp,
            "delivery_day": local.dt.date,
            "local_hour": local.dt.hour,
            "value": pd.to_numeric(source[settings["column"]], errors="coerce"),
        }).sort_values("timestamp_utc").reset_index(drop=True)
        targets[name].attrs["read_audit"] = source.attrs["read_audit"]
    return targets


def assert_target_scope(targets: dict, config: dict) -> list:
    _, _, first_test, _, included = protocol_dates(config)
    expected = set(included)
    for frame in targets.values():
        assert set(frame["delivery_day"].unique()) == expected
        assert frame["delivery_day"].map(lambda value: value < first_test).all()
    return included


def hourly_target(frame: pd.DataFrame, settings: dict, quarters: int) -> pd.DataFrame:
    indexed = frame.set_index("timestamp_utc")["value"]
    aggregation = settings["hourly_aggregation"]
    if aggregation == "identity":
        hourly = indexed.copy()
    elif aggregation == "mean":
        hourly = indexed.resample("1H").mean()
        counts = indexed.resample("1H").count()
        hourly[counts < quarters] = np.nan
    elif aggregation == "sum":
        hourly = indexed.resample("1H").sum(min_count=quarters)
    else:
        raise ValueError(aggregation)
    local = hourly.index.tz_convert(frame.attrs["civil_timezone"])
    return pd.DataFrame({"timestamp_utc": hourly.index, "delivery_day": local.date,
                         "local_hour": local.hour, "value": hourly.to_numpy()})


def make_hourly_targets(targets: dict, config: dict) -> dict:
    hourly = {}
    quarters = config["protocol"]["quarters_per_hour"]
    for name, frame in targets.items():
        frame.attrs["civil_timezone"] = config["data"]["civil_timezone"]
        hourly[name] = hourly_target(frame, config["targets"][name], quarters)
    return hourly


def spike_threshold(values: pd.Series, config: dict) -> float:
    clean = values.dropna()
    lower = config["analysis"]["spike_lower_quantile"]
    upper = config["analysis"]["spike_upper_quantile"]
    q1, q3 = clean.quantile([lower, upper])
    return float(q3 + config["analysis"]["spike_iqr_multiplier"] * (q3 - q1))


def analyse_d1(targets: dict, config: dict) -> pd.DataFrame:
    quantiles = config["analysis"]["distribution_quantiles"]
    tolerance = config["analysis"]["zero_tolerance"]
    rows = []
    for name, frame in targets.items():
        values = frame["value"].dropna()
        threshold = spike_threshold(values, config)
        row = {"target": name, "unit": config["targets"][name]["unit"],
               "n_total": len(frame), "n_valid": len(values), "n_missing": frame["value"].isna().sum(),
               "mean": values.mean(), "std": values.std(), "skewness": stats.skew(values, bias=False),
               "spike_threshold": threshold, "n_spike": (values > threshold).sum(),
               "spike_share": (values > threshold).mean(), "n_negative": (values < -tolerance).sum(),
               "negative_share": (values < -tolerance).mean(), "n_zero": (values.abs() <= tolerance).sum(),
               "zero_share": (values.abs() <= tolerance).mean()}
        row.update({f"q{int(q * 100):02d}": values.quantile(q) for q in quantiles})
        rows.append(row)
    return pd.DataFrame(rows)


def aligned_pair(first: pd.DataFrame, second: pd.DataFrame, lag: int) -> pd.DataFrame:
    left = first[["timestamp_utc", "value"]].rename(columns={"value": "x"})
    right = second[["timestamp_utc", "value"]].rename(columns={"value": "y"}).copy()
    right["timestamp_utc"] = right["timestamp_utc"] - pd.Timedelta(hours=lag)
    return left.merge(right, on="timestamp_utc", how="inner").dropna()


def analyse_d2(hourly: dict, config: dict) -> pd.DataFrame:
    names = config["analysis"]["dependency_targets"]
    lower_q = config["analysis"]["lower_tail_quantile"]
    upper_q = config["analysis"]["upper_tail_quantile"]
    rows = []
    for i, first_name in enumerate(names):
        for second_name in names[i + 1:]:
            for lag in config["analysis"]["dependency_lags_hours"]:
                pair = aligned_pair(hourly[first_name], hourly[second_name], lag)
                assert len(pair) >= config["analysis"]["minimum_pair_samples"]
                x_low, y_low = pair["x"].quantile(lower_q), pair["y"].quantile(lower_q)
                x_high, y_high = pair["x"].quantile(upper_q), pair["y"].quantile(upper_q)
                lower_base = (pair["x"] <= x_low).sum()
                upper_base = (pair["x"] >= x_high).sum()
                rows.append({"target_x": first_name, "target_y": second_name, "lag_hours_y_after_x": lag,
                             "n_valid_pairs": len(pair), "spearman_rho": pair["x"].corr(pair["y"], method="spearman"),
                             "lower_tail_quantile": lower_q, "n_x_lower_tail": lower_base,
                             "lower_tail_dependence": ((pair["x"] <= x_low) & (pair["y"] <= y_low)).sum() / lower_base,
                             "upper_tail_quantile": upper_q, "n_x_upper_tail": upper_base,
                             "upper_tail_dependence": ((pair["x"] >= x_high) & (pair["y"] >= y_high)).sum() / upper_base})
    return pd.DataFrame(rows)


def correlation_row(name: str, label: str, lag_days: int, frame: pd.DataFrame) -> dict:
    shifted = frame[["delivery_day", "local_hour", "value"]].copy()
    shifted["delivery_day"] = shifted["delivery_day"].map(lambda value: value + timedelta(days=lag_days))
    pair = frame.merge(shifted, on=["delivery_day", "local_hour"], suffixes=("_y", "_x")).dropna()
    return {"target": name, "statistic": label, "group": "all", "lag": lag_days,
            "n_valid": len(pair), "value": pair["value_x"].corr(pair["value_y"], method="spearman")}


def analyse_d3(hourly: dict, config: dict) -> pd.DataFrame:
    rows = []
    for name, frame in hourly.items():
        ordered = frame.sort_values(["delivery_day", "timestamp_utc"]).copy()
        ordered["previous"] = ordered.groupby("delivery_day")["value"].shift(1)
        for hour, group in ordered.groupby("local_hour"):
            valid = group[["previous", "value"]].dropna()
            rows.append({"target": name, "statistic": "intraday_lag_1_hour_spearman", "group": f"hour_{hour:02d}",
                         "lag": 1, "n_valid": len(valid), "value": valid["previous"].corr(valid["value"], method="spearman")})
        for lag_days in config["analysis"]["autocorrelation_day_lags"]:
            rows.append(correlation_row(name, "same_civil_hour_spearman", lag_days, frame))
        weekend = set(config["analysis"]["weekend_weekdays"])
        ordered["day_type"] = ordered["delivery_day"].map(lambda value: "weekend" if value.weekday() in weekend else "weekday")
        for day_type, group in ordered.groupby("day_type"):
            valid = group["value"].dropna()
            rows.append({"target": name, "statistic": "level_mean", "group": day_type, "lag": np.nan,
                         "n_valid": len(valid), "value": valid.mean()})
    return pd.DataFrame(rows)


def availability_timestamp(row: pd.Series, settings: dict, civil_zone: ZoneInfo) -> datetime:
    if settings["availability_kind"] == "hour_end":
        return row["timestamp_utc"].to_pydatetime() + timedelta(hours=1)
    publish_day = row["delivery_day"] + timedelta(days=settings["publication_day_offset"])
    publish_time = time.fromisoformat(settings["publication_civil_time"])
    return datetime.combine(publish_day, publish_time, tzinfo=civil_zone).astimezone(timezone.utc)


def d4_design(frame: pd.DataFrame, settings: dict, config: dict) -> pd.DataFrame:
    zone = ZoneInfo(config["data"]["civil_timezone"])
    gate_time = time.fromisoformat(config["protocol"]["decision_gate_civil_time"])
    history_count = config["protocol"]["history_mean_observations"]
    collapsed = frame.groupby(["delivery_day", "local_hour"], as_index=False)["value"].mean()
    collapsed["timestamp_utc"] = collapsed.apply(
        lambda row: pd.Timestamp(datetime.combine(row["delivery_day"], time(int(row["local_hour"])), tzinfo=zone).astimezone(timezone.utc)), axis=1)
    collapsed["available_at"] = collapsed.apply(lambda row: availability_timestamp(row, settings, zone), axis=1)
    rows = []
    for target in collapsed.itertuples(index=False):
        gate = datetime.combine(target.delivery_day - timedelta(days=1), gate_time, tzinfo=zone).astimezone(timezone.utc)
        history = collapsed[(collapsed["local_hour"] == target.local_hour) & (collapsed["available_at"] <= gate)].dropna(subset=["value"])
        history = history.sort_values("available_at")
        recent = history.tail(history_count)
        rows.append({"target": target.value, "previous_same_slot": recent["value"].iloc[-1] if len(recent) else np.nan,
                     "previous_7_mean": recent["value"].mean() if len(recent) == history_count else np.nan,
                     "gate_utc": gate, "latest_input_available_at": recent["available_at"].max() if len(recent) else pd.NaT})
    design = pd.DataFrame(rows)
    assert (design["latest_input_available_at"].dropna() <= design.loc[design["latest_input_available_at"].notna(), "gate_utc"]).all()
    return design


def regression_r2(design: pd.DataFrame, columns: list) -> tuple:
    valid = design[["target"] + columns].dropna()
    matrix = np.column_stack([np.ones(len(valid)), valid[columns].to_numpy()])
    coefficients = np.linalg.lstsq(matrix, valid["target"].to_numpy(), rcond=None)[0]
    residual = valid["target"].to_numpy() - matrix @ coefficients
    denominator = ((valid["target"] - valid["target"].mean()) ** 2).sum()
    return len(valid), 1.0 - float(residual @ residual) / denominator


def analyse_d4(hourly: dict, config: dict) -> pd.DataFrame:
    specs = {"previous_same_slot": ["previous_same_slot"], "previous_7_mean": ["previous_7_mean"],
             "combined": ["previous_same_slot", "previous_7_mean"]}
    assert list(specs) == config["analysis"]["d4_model_specs"]
    rows = []
    for name, frame in hourly.items():
        design = d4_design(frame, config["targets"][name], config)
        for model, columns in specs.items():
            n_valid, r2 = regression_r2(design, columns)
            rows.append({"target": name, "model": model, "n_valid": n_valid, "n_predictors": len(columns), "r2": r2})
    return pd.DataFrame(rows)


def analyse_d5(targets: dict, hourly: dict, included_days: list, d1: pd.DataFrame, config: dict) -> pd.DataFrame:
    window_days = config["protocol"]["training_window_days"]
    joint_names = config["analysis"]["dependency_targets"]
    hours = config["protocol"]["hours_per_standard_day"]
    component_dimensions = {name: hours * 60 // config["targets"][name]["native_minutes"] for name in joint_names}
    dimension = sum(component_dimensions.values())
    dimension_text = ";".join(f"{name}:{value}" for name, value in component_dimensions.items())
    thresholds = d1.set_index("target")["spike_threshold"].to_dict()
    rows = []
    for end_index in range(window_days - 1, len(included_days)):
        days = included_days[end_index - window_days + 1:end_index + 1]
        for name, frame in targets.items():
            values = frame.loc[frame["delivery_day"].isin(days), "value"]
            rows.append({"window_start": days[0], "window_end": days[-1], "target": name, "n_days": len(days),
                         "n_total": len(values), "n_valid": values.notna().sum(), "n_extreme": (values.dropna() > thresholds[name]).sum(),
                         "joint_dimension": dimension if name in joint_names else np.nan,
                         "joint_dimension_components": dimension_text if name in joint_names else ""})
        wide = None
        for name in joint_names:
            part = hourly[name].loc[hourly[name]["delivery_day"].isin(days), ["timestamp_utc", "value"]].rename(columns={"value": name})
            wide = part if wide is None else wide.merge(part, on="timestamp_utc", how="outer")
        rows.append({"window_start": days[0], "window_end": days[-1], "target": "joint_complete_hours", "n_days": len(days),
                     "n_total": len(wide), "n_valid": wide[joint_names].notna().all(axis=1).sum(), "n_extreme": np.nan,
                     "joint_dimension": dimension, "joint_dimension_components": dimension_text})
    return pd.DataFrame(rows)


def analyse_d6(targets: dict, included_days: list, config: dict) -> pd.DataFrame:
    assert config["analysis"]["stability_halves"] == 2
    midpoint = len(included_days) // config["analysis"]["stability_halves"]
    halves = {"first": set(included_days[:midpoint]), "second": set(included_days[midpoint:])}
    quantiles = config["analysis"]["stability_quantiles"]
    rows = []
    for name, frame in targets.items():
        values = {half: frame.loc[frame["delivery_day"].isin(days), "value"].dropna() for half, days in halves.items()}
        pooled_std = pd.concat(list(values.values())).std()
        standardized = (values["second"].mean() - values["first"].mean()) / pooled_std
        ks = stats.ks_2samp(values["first"], values["second"])
        row = {"target": name, "first_n_valid": len(values["first"]), "second_n_valid": len(values["second"]),
               "first_mean": values["first"].mean(), "second_mean": values["second"].mean(),
               "standardized_mean_change": standardized, "ks_statistic": ks.statistic, "ks_pvalue": ks.pvalue,
               "drift_flag": abs(standardized) >= config["analysis"]["stability_standardized_mean_threshold"] or ks.pvalue < config["analysis"]["stability_ks_alpha"]}
        for q in quantiles:
            row[f"first_q{int(q * 100):02d}"] = values["first"].quantile(q)
            row[f"second_q{int(q * 100):02d}"] = values["second"].quantile(q)
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", line_terminator="\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    arguments = parse_arguments()
    config = yaml.safe_load(arguments.config.open(encoding="utf-8"))
    np.random.seed(config["protocol"]["random_seed"])
    targets = load_targets(config)
    included_days = assert_target_scope(targets, config)
    hourly = make_hourly_targets(targets, config)
    results = {}
    results["d1"] = analyse_d1(targets, config)
    results["d2"] = analyse_d2(hourly, config)
    results["d3"] = analyse_d3(hourly, config)
    results["d4"] = analyse_d4(hourly, config)
    results["d5"] = analyse_d5(targets, hourly, included_days, results["d1"], config)
    results["d6"] = analyse_d6(targets, included_days, config)
    output = config["output"]
    directory = REPOSITORY_ROOT / output["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for number in range(1, 7):
        key = f"d{number}"
        path = directory / output[f"{key}_csv"]
        write_csv(results[key], path)
        output_paths[key] = path
    summary = {
        "random_seed": config["protocol"]["random_seed"],
        "environment_fingerprint": environment_fingerprint(),
        "window": {"first_delivery_day": config["protocol"]["first_delivery_day"],
                   "last_delivery_day": config["protocol"]["last_delivery_day"],
                   "first_test_delivery_day": config["protocol"]["first_test_delivery_day"],
                   "excluded_delivery_days": config["protocol"]["excluded_delivery_days"],
                   "delivery_days": len(included_days), "all_delivery_days_before_test": True},
        "missing_value_policy": "preserve",
        "activation_hourly_aggregation": "sum four 15-minute MWh values; no 0.25 multiplier",
        "utc_primary_key": True,
        "civil_delivery_day_timezone": config["data"]["civil_timezone"],
        "input_read_audit": {name: frame.attrs["read_audit"] for name, frame in targets.items()},
        "result_rows": {key: len(frame) for key, frame in results.items()},
        "outputs": {key: {"file": path.name, "sha256": file_sha256(path)} for key, path in output_paths.items()},
    }
    with (directory / output["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"delivery_days={len(included_days)} all_before_test=True")
    print(f"wrote {directory}")


if __name__ == "__main__":
    main()
