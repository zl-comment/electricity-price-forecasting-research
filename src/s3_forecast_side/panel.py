"""Build the frozen S3-A hourly target panel and availability-safe features."""

from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest import TestCase
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _utc_timestamp(day: date, clock: str, timezone: str) -> pd.Timestamp:
    local = datetime.combine(day, time.fromisoformat(clock), tzinfo=ZoneInfo(timezone))
    return pd.Timestamp(local).tz_convert("UTC")


def decision_time_utc(delivery_day: str, snapshot: str, config: dict) -> pd.Timestamp:
    """Return D-1's configured civil gate in UTC; assert that the snapshot exists."""
    snapshots = config["visibility"]["snapshots"]
    _require(snapshot in snapshots, f"Unknown snapshot: {snapshot}")
    civil_day = date.fromisoformat(delivery_day) - timedelta(days=1)
    clock = snapshots[snapshot]["decision_civil_time"]
    return _utc_timestamp(civil_day, clock, config["protocol"]["civil_timezone"])


def _publication_time(delivery_day: str, settings: dict, timezone: str) -> pd.Timestamp:
    publish_day = date.fromisoformat(delivery_day) + timedelta(days=settings["publication_day_offset"])
    return _utc_timestamp(publish_day, settings["publication_civil_time"], timezone)


def _read_target_source(root: Path, config: dict, target: str) -> pd.DataFrame:
    target_settings = config["targets"][target]
    source_settings = config["data"]["datasets"][target_settings["dataset"]]
    path = root / config["data"]["root"] / source_settings["csv"]
    raw = pd.read_csv(path)
    area_column = source_settings["area_column"]
    raw = raw.loc[raw[area_column].eq(config["data"]["primary_price_area"])].copy()
    time_column = source_settings["time_column"]
    raw["timestamp_utc"] = pd.to_datetime(raw[time_column], utc=True)
    _require(not raw["timestamp_utc"].duplicated().any(), f"{target}: duplicate UTC timestamps")
    raw = raw.sort_values("timestamp_utc").reset_index(drop=True)
    local = raw["timestamp_utc"].dt.tz_convert(config["protocol"]["civil_timezone"])
    raw["delivery_day"] = local.dt.strftime("%Y-%m-%d")
    raw["hour_start_utc"] = raw["timestamp_utc"].dt.floor("H")
    return raw


def _complete_group(group: pd.DataFrame, columns: list, expected: int) -> bool:
    return len(group) == expected and not group[columns].isna().any().any()


def _aggregate_group(group: pd.DataFrame, target: str, settings: dict, expected: int) -> float:
    column = settings["column"]
    aggregation = settings["hourly_aggregation"]
    required = [column]
    if aggregation == "activation_weighted_mean":
        required.append(settings["weight_column"])
    if not _complete_group(group, required, expected):
        return np.nan
    if aggregation == "mean":
        return float(group[column].mean())
    if aggregation == "sum":
        _require(settings["aggregation_multiplier"] == 1.0, f"{target}: invalid sum multiplier")
        return float(group[column].sum())
    if aggregation == "identity":
        return float(group[column].iloc[0])
    _require(aggregation == "activation_weighted_mean", f"{target}: unknown aggregation")
    weights = group[settings["weight_column"]].to_numpy(dtype=float)
    total = float(weights.sum())
    if total == 0.0:
        _require(settings["zero_activation_value"] is None, f"{target}: zero value must be missing")
        return np.nan
    return float(np.dot(group[column].to_numpy(dtype=float), weights) / total)


def _hourly_target(raw: pd.DataFrame, target: str, config: dict) -> pd.DataFrame:
    settings = config["targets"][target]
    source = config["data"]["datasets"][settings["dataset"]]
    expected = int(pd.Timedelta(hours=1) / pd.Timedelta(minutes=source["native_minutes"]))
    rows = []
    for stamp, group in raw.groupby("hour_start_utc", sort=True):
        day_values = group["delivery_day"].unique().tolist()
        _require(len(day_values) == 1, f"{target}: UTC hour crosses delivery days")
        rows.append({"delivery_day": day_values[0], "timestamp_utc": stamp,
                     "value": _aggregate_group(group, target, settings, expected),
                     "source_periods": len(group)})
    hourly = pd.DataFrame(rows)
    kind = config["visibility"]["targets"][target]["availability_kind"]
    if kind == "hour_end":
        hourly["available_at"] = hourly["timestamp_utc"] + pd.Timedelta(hours=1)
    else:
        _require(kind == "daily_publication", f"{target}: unknown availability kind")
        visibility = config["visibility"]["targets"][target]
        timezone = config["protocol"]["civil_timezone"]
        hourly["available_at"] = hourly["delivery_day"].map(
            lambda day: _publication_time(day, visibility, timezone))
    return hourly


def _normalise_day(group: pd.DataFrame, target: str, periods: int, policy: str) -> pd.DataFrame:
    _require(policy == "truncate_or_repeat_last", f"Unknown DST policy: {policy}")
    ordered = group.sort_values("timestamp_utc").reset_index(drop=True)
    source_hours = len(ordered)
    _require(source_hours > 0, f"{target}: empty delivery day")
    if source_hours > periods:
        ordered = ordered.iloc[:periods].copy()
    elif source_hours < periods:
        repeated = pd.concat([ordered.iloc[[-1]]] * (periods - source_hours), ignore_index=True)
        ordered = pd.concat([ordered, repeated], ignore_index=True)
    _require(len(ordered) == periods, f"{target}: DST normalisation failed")
    ordered["hour"] = np.arange(periods)
    ordered["target"] = target
    ordered["source_hours"] = source_hours
    ordered["dst_adjusted"] = source_hours != periods
    return ordered


def _validate_availability_config(config: dict, repository_root: Path) -> None:
    profile_path = repository_root / config["data"]["target_profile_config"]
    with profile_path.open(encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    for target, settings in config["visibility"]["targets"].items():
        observed = profile["targets"][target]["availability_kind"]
        _require(settings["availability_kind"] == observed,
                 f"{target}: availability_kind differs from target profile")


def build_hourly_panel(config: dict, repository_root: Path) -> pd.DataFrame:
    """Return a long 24-period panel with UTC source and availability timestamps."""
    _validate_availability_config(config, repository_root)
    first = config["window"]["first_delivery_day"]
    last = config["window"]["last_delivery_day"]
    excluded = set(config["window"]["excluded_delivery_days"])
    periods = config["protocol"]["periods_per_day"]
    policy = config["protocol"]["dst_policy"]
    parts = []
    for target in config["targets"]:
        raw = _read_target_source(repository_root, config, target)
        hourly = _hourly_target(raw, target, config)
        hourly = hourly[hourly["delivery_day"].between(first, last)]
        for day, group in hourly.groupby("delivery_day", sort=True):
            if day not in excluded:
                parts.append(_normalise_day(group, target, periods, policy))
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["delivery_day", "target", "hour"]).reset_index(drop=True)
    expected_days = _usable_days(config)
    _require(sorted(panel["delivery_day"].unique()) == expected_days, "Panel delivery days differ")
    counts = panel.groupby(["delivery_day", "target"]).size()
    _require(counts.eq(periods).all(), "Panel does not contain 24 rows per target-day")
    return panel[["delivery_day", "hour", "target", "value", "available_at",
                  "timestamp_utc", "source_periods", "source_hours", "dst_adjusted"]]


def _usable_days(config: dict) -> list:
    days = pd.date_range(config["window"]["first_delivery_day"],
                         config["window"]["last_delivery_day"], freq="D")
    excluded = set(config["window"]["excluded_delivery_days"])
    return [stamp.strftime("%Y-%m-%d") for stamp in days if stamp.strftime("%Y-%m-%d") not in excluded]


def build_snapshot(panel: pd.DataFrame, delivery_day: str, snapshot: str,
                   config: dict) -> pd.DataFrame:
    """Return only rows visible at the configured gate and verify D-day exposure."""
    decision = decision_time_utc(delivery_day, snapshot, config)
    visible = panel.loc[panel["available_at"].le(decision)].copy()
    _require(visible["available_at"].le(decision).all(), "Snapshot contains a late input")
    current = visible.loc[visible["delivery_day"].eq(delivery_day)]
    settings = config["visibility"]["snapshots"][snapshot]
    if "added_realized_target" in settings:
        added = settings["added_realized_target"]
        _require(set(current["target"]) == {added}, f"{snapshot}: wrong D-day target exposure")
        _require(len(current) == config["protocol"]["periods_per_day"],
                 f"{snapshot}: incomplete realized target")
    else:
        _require(current.empty, f"{snapshot}: D-day target is visible")
    return visible


def window_days(delivery_day: str, config: dict) -> dict:
    """Return calendar and naturally available F, C, and clearing-gap delivery days."""
    anchor = pd.Timestamp(delivery_day)
    excluded = set(config["window"]["excluded_delivery_days"])
    result = {}
    names = {"fit": "fit", "calibration": "probability_calibration", "gap": "clearing_gap"}
    for output_name, config_name in names.items():
        settings = config["window"][config_name]
        first = anchor + pd.DateOffset(days=settings["start_offset_days"])
        last = anchor + pd.DateOffset(days=settings["end_offset_days"])
        calendar = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d").tolist()
        _require(len(calendar) == settings["calendar_days"], f"{output_name}: wrong calendar length")
        result[f"{output_name}_calendar"] = calendar
        result[output_name] = [day for day in calendar if day not in excluded]
    _require(not (set(result["fit"]) & set(result["calibration"])), "F and C overlap")
    _require(not (set(result["calibration"]) & set(result["gap"])), "C and gap overlap")
    return result


def test_delivery_days(config: dict, repository_root: Path) -> list:
    """Return and assert the frozen S2-aligned 211-day evaluation set."""
    usable = set(_usable_days(config))
    first = config["split"]["first_test_delivery_day"]
    candidates = []
    for day in sorted(usable):
        previous = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        if day >= first and previous in usable:
            candidates.append(day)
    reference_path = repository_root / config["data"]["s2_daily_results_csv"]
    reference = pd.read_csv(reference_path, usecols=["delivery_day"])
    expected = sorted(reference["delivery_day"].astype(str).unique().tolist())
    _require(candidates == expected, "Test delivery days differ from S2 day by day")
    _require(len(expected) == config["split"]["expected_test_days"], "Wrong test-day count")
    return expected


def _activation_daily(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    volume = panel[panel["target"].eq("activation_volume")]
    price = panel[panel["target"].eq("activation_price")]
    joined = volume.merge(price, on=["delivery_day", "hour"], suffixes=("_v", "_p"))
    lower = config["targets"]["activation_volume"]["physical_lower_bound"]
    joined["positive"] = joined["value_v"].gt(lower)
    _require(joined.loc[joined["positive"], "value_p"].notna().all(),
             "Positive activation has missing price")
    joined["weighted_price"] = joined["value_v"] * joined["value_p"]
    rows = []
    periods = config["protocol"]["periods_per_day"]
    for day, group in joined.groupby("delivery_day", sort=True):
        _require(len(group) == periods, f"{day}: incomplete activation join")
        total = group["value_v"].sum(min_count=periods)
        numerator = group.loc[group["positive"], "weighted_price"].sum()
        rows.append({"delivery_day": day, "total_volume": total, "numerator": numerator,
                     "available_at": group[["available_at_v", "available_at_p"]].max().max()})
    return pd.DataFrame(rows).set_index("delivery_day")


def _panel_index(panel: pd.DataFrame, config: dict) -> dict:
    by_key = {}
    for (target, hour), group in panel.groupby(["target", "hour"], sort=False):
        indexed = group.set_index("delivery_day").sort_index()
        _require(not indexed.index.duplicated().any(), f"{target} hour {hour}: duplicate day")
        by_key[(target, int(hour))] = indexed
    return {"by_key": by_key, "activation_daily": _activation_daily(panel, config)}


def _exact_history(panel_index: dict, target: str, hour: int, source_day: str,
                   decision: pd.Timestamp) -> tuple:
    rows = panel_index["by_key"][(target, hour)]
    if source_day not in rows.index:
        return np.nan, pd.NaT, None
    row = rows.loc[source_day]
    if row["available_at"] > decision:
        return np.nan, pd.NaT, None
    return row["value"], row["available_at"], source_day


def _recent_history(panel_index: dict, target: str, hour: int, delivery_day: str,
                    decision: pd.Timestamp, config: dict) -> tuple:
    rows = panel_index["by_key"][(target, hour)]
    rows = rows.loc[rows.index < delivery_day]
    rows = rows.loc[rows["available_at"].le(decision)].dropna(subset=["value"])
    count = config["features"]["common"]["recent_visible_days"]
    recent = rows.sort_values(["delivery_day", "available_at"]).tail(count)
    minimum = config["features"]["common"]["minimum_recent_valid"]
    if len(recent) < minimum:
        return np.nan, np.nan, recent["available_at"].max(), None
    latest = recent.index.max()
    return float(recent["value"].mean()), float(recent["value"].eq(0.0).mean()), \
        recent["available_at"].max(), latest


def _daily_activation_price(panel_index: dict, delivery_day: str,
                            decision: pd.Timestamp, config: dict) -> tuple:
    daily = panel_index["activation_daily"]
    eligible = daily.loc[(daily.index < delivery_day) & daily["available_at"].le(decision)]
    count = config["features"]["common"]["recent_visible_days"]
    recent = eligible.tail(count)
    minimum = config["features"]["common"]["minimum_recent_valid"]
    if len(recent) < minimum:
        return np.nan, pd.NaT, None
    total = float(recent["total_volume"].sum())
    if total == 0.0:
        return np.nan, recent["available_at"].max(), recent.index.max()
    value = float(recent["numerator"].sum() / total)
    return value, recent["available_at"].max(), recent.index.max()


def _period_features(hour: int, decision: pd.Timestamp, config: dict) -> dict:
    boundaries = config["features"]["activation_period_boundaries"]
    _require(boundaries[0] == 0 and boundaries[-1] == config["protocol"]["periods_per_day"],
             "Activation period boundaries do not span the day")
    values = {}
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        name = f"period_{lower:02d}_{upper:02d}"
        values[name] = int(lower <= hour < upper)
        values[f"{name}_available_at"] = decision
    _require(sum(values[name] for name in values if not name.endswith("_available_at")) == 1,
             "Activation periods are not one-hot")
    return values


def _feature_snapshot(snapshot: str, target: str, config: dict) -> str:
    settings = config["visibility"]["snapshots"][snapshot]
    if target.startswith("activation_"):
        return settings["activation_feature_snapshot"]
    return snapshot


def _expanded_features(target: str, config: dict) -> list:
    output = []
    supported = {"previous_visible_same_hour", "lag_7_same_hour",
                 "recent_7_visible_same_hour_mean", "recent_7_visible_same_hour_zero_share",
                 "recent_7_visible_days_volume_weighted_price", "is_weekend",
                 "four_hour_period_indicators"}
    for feature in config["features"][target]:
        _require(feature in supported, f"Unknown feature: {feature}")
        if feature == "four_hour_period_indicators":
            bounds = config["features"]["activation_period_boundaries"]
            output.extend([f"period_{lower:02d}_{upper:02d}"
                           for lower, upper in zip(bounds[:-1], bounds[1:])])
        else:
            output.append(feature)
    return output


def _base_feature_row(panel_index: dict, target: str, hour: int, delivery_day: str,
                      snapshot: str, config: dict) -> dict:
    feature_snapshot = _feature_snapshot(snapshot, target, config)
    decision = decision_time_utc(delivery_day, feature_snapshot, config)
    labels = panel_index["by_key"][(target, hour)]
    _require(delivery_day in labels.index, f"{target} {delivery_day} hour {hour}: missing label")
    label = labels.loc[delivery_day]
    row = {"delivery_day": delivery_day, "target": target, "hour": hour,
           "value": label["value"], "label_available_at": label["available_at"],
           "decision_time_utc": decision, "feature_snapshot": feature_snapshot}
    weekend = date.fromisoformat(delivery_day).weekday() in config["features"]["common"]["weekend_weekdays"]
    row["is_weekend"] = int(weekend)
    row["is_weekend_available_at"] = decision
    if "four_hour_period_indicators" in config["features"][target]:
        row.update(_period_features(hour, decision, config))
    return row


def _same_hour_features(row: dict, panel_index: dict, config: dict) -> None:
    target, hour, delivery_day = row["target"], row["hour"], row["delivery_day"]
    decision = row["decision_time_utc"]
    previous = (date.fromisoformat(delivery_day) - timedelta(days=1)).isoformat()
    lag = config["features"]["common"]["recent_visible_days"]
    lag_day = (date.fromisoformat(delivery_day) - timedelta(days=lag)).isoformat()
    for name, source_day in [("previous_visible_same_hour", previous), ("lag_7_same_hour", lag_day)]:
        if name in config["features"][target]:
            value, available, actual_day = _exact_history(panel_index, target, hour, source_day, decision)
            row[name], row[f"{name}_available_at"], row[f"{name}_source_day"] = value, available, actual_day
    requested = config["features"][target]
    if any(name.startswith("recent_7_visible_same_hour") for name in requested):
        mean, zero_share, available, latest = _recent_history(
            panel_index, target, hour, delivery_day, decision, config)
        if "recent_7_visible_same_hour_mean" in requested:
            row["recent_7_visible_same_hour_mean"] = mean
            row["recent_7_visible_same_hour_mean_available_at"] = available
            row["recent_7_visible_same_hour_mean_source_day"] = latest
        if "recent_7_visible_same_hour_zero_share" in requested:
            row["recent_7_visible_same_hour_zero_share"] = zero_share
            row["recent_7_visible_same_hour_zero_share_available_at"] = available
            row["recent_7_visible_same_hour_zero_share_source_day"] = latest


def _activation_price_feature(row: dict, values: tuple, config: dict) -> None:
    name = "recent_7_visible_days_volume_weighted_price"
    if name not in config["features"][row["target"]]:
        return
    value, available, latest = values
    row[name] = value
    row[f"{name}_available_at"] = available
    row[f"{name}_source_day"] = latest


def assert_input_visibility(frame: pd.DataFrame) -> None:
    """Assert every recorded input timestamp is at its row decision and sources predate D."""
    available_columns = [column for column in frame if column.endswith("_available_at")
                         and column != "label_available_at"]
    _require(bool(available_columns), "No input availability columns recorded")
    for column in available_columns:
        valid = frame[column].notna()
        if valid.any():
            available = pd.to_datetime(frame.loc[valid, column], utc=True)
            decision = pd.to_datetime(frame.loc[valid, "decision_time_utc"], utc=True)
            _require(available.le(decision).all(), f"Late feature input in {column}")
    source_columns = [column for column in frame if column.endswith("_source_day")]
    for column in source_columns:
        valid = frame[column].notna()
        _require(frame.loc[valid, column].lt(frame.loc[valid, "delivery_day"]).all(),
                 f"D-day or future feature input in {column}")


def _build_feature_frame(panel_index: dict, delivery_day: str, snapshot: str,
                         config: dict, validate: bool) -> pd.DataFrame:
    periods = config["protocol"]["periods_per_day"]
    rows = []
    for target in config["targets"]:
        target_decision = decision_time_utc(delivery_day, _feature_snapshot(snapshot, target, config), config)
        activation_values = _daily_activation_price(panel_index, delivery_day, target_decision, config) \
            if target == "activation_price" else (np.nan, pd.NaT, None)
        for hour in range(periods):
            row = _base_feature_row(panel_index, target, hour, delivery_day, snapshot, config)
            _same_hour_features(row, panel_index, config)
            _activation_price_feature(row, activation_values, config)
            rows.append(row)
    design = pd.DataFrame(rows)
    if validate:
        assert_input_visibility(design)
    return design.sort_values(["target", "hour"]).reset_index(drop=True)


def build_feature_frame(panel: pd.DataFrame, delivery_day: str, snapshot: str,
                        config: dict) -> pd.DataFrame:
    """Build one delivery day's per-target design and assert input visibility."""
    return _build_feature_frame(_panel_index(panel, config), delivery_day, snapshot, config, True)


def build_all_features(panel: pd.DataFrame, snapshot: str, config: dict) -> pd.DataFrame:
    """Build availability-safe designs for every naturally available delivery day."""
    panel_index = _panel_index(panel, config)
    frames = [_build_feature_frame(panel_index, day, snapshot, config, False)
              for day in _usable_days(config)]
    design = pd.concat(frames, ignore_index=True)
    assert_input_visibility(design)
    return design


def _complete_mask(frame: pd.DataFrame, target: str, config: dict) -> pd.Series:
    required = _expanded_features(target, config)
    return frame[required].notna().all(axis=1)


def _observed_max(frame: pd.DataFrame, target: str, config: dict) -> pd.Timestamp:
    static = {"is_weekend"} | {name for name in _expanded_features(target, config)
                               if name.startswith("period_")}
    columns = [f"{name}_available_at" for name in _expanded_features(target, config)
               if name not in static]
    if not columns:
        return pd.NaT
    values = [pd.to_datetime(frame[column].dropna(), utc=True).max()
              for column in columns if frame[column].notna().any()]
    return max(values) if values else pd.NaT


def _block_counts(design: pd.DataFrame, target: str, days: list,
                  current_decision: pd.Timestamp, config: dict) -> dict:
    block = design[design["target"].eq(target) & design["delivery_day"].isin(days)]
    features_valid = _complete_mask(block, target, config)
    labels_valid = block["value"].notna()
    label_visible = block["label_available_at"].le(current_decision)
    _require(label_visible.all(), f"{target}: block label is unavailable at current decision")
    usable = features_valid & labels_valid
    return {"candidate_rows": len(block), "feature_complete_rows": int(features_valid.sum()),
            "feature_missing_rows": int((~features_valid).sum()),
            "label_valid_rows": int(labels_valid.sum()),
            "label_missing_rows": int((~labels_valid).sum()), "usable_rows": int(usable.sum()),
            "removed_rows": int((~usable).sum())}


def _audit_row(design: pd.DataFrame, panel: pd.DataFrame, day: str, snapshot: str,
               target: str, blocks: dict, config: dict) -> dict:
    decision = decision_time_utc(day, snapshot, config)
    forecast = design[design["target"].eq(target) & design["delivery_day"].eq(day)]
    _require(len(forecast) == config["protocol"]["periods_per_day"], "Incomplete forecast design")
    _require(_complete_mask(forecast, target, config).all(), f"{day} {target}: missing test feature")
    fit = _block_counts(design, target, blocks["fit"], decision, config)
    calibration = _block_counts(design, target, blocks["calibration"], decision, config)
    maximum = _observed_max(forecast, target, config)
    if "added_realized_target" in config["visibility"]["snapshots"][snapshot]:
        added = config["visibility"]["snapshots"][snapshot]["added_realized_target"]
        current = panel[panel["delivery_day"].eq(day) & panel["target"].eq(added)]
        maximum = max(maximum, current["available_at"].max())
    row = {"delivery_day": day, "snapshot": snapshot, "target": target,
           "decision_time_utc": decision, "max_input_available_at": maximum,
           "visibility_assertion": True,
           "fit_first_day": blocks["fit_calendar"][0], "fit_last_day": blocks["fit_calendar"][-1],
           "fit_calendar_days": len(blocks["fit_calendar"]), "fit_available_days": len(blocks["fit"]),
           "fit_excluded_days": len(blocks["fit_calendar"]) - len(blocks["fit"]),
           "calibration_first_day": blocks["calibration_calendar"][0],
           "calibration_last_day": blocks["calibration_calendar"][-1],
           "calibration_calendar_days": len(blocks["calibration_calendar"]),
           "calibration_available_days": len(blocks["calibration"]),
           "calibration_excluded_days": len(blocks["calibration_calendar"]) - len(blocks["calibration"]),
           "gap_first_day": blocks["gap_calendar"][0], "gap_last_day": blocks["gap_calendar"][-1],
           "gap_calendar_days": len(blocks["gap_calendar"]),
           "gap_available_days": len(blocks["gap"]),
           "gap_excluded_days": len(blocks["gap_calendar"]) - len(blocks["gap"])}
    row.update({f"fit_{key}": value for key, value in fit.items()})
    row.update({f"calibration_{key}": value for key, value in calibration.items()})
    return row


def build_panel_audit(panel: pd.DataFrame, config: dict, repository_root: Path) -> pd.DataFrame:
    """Return per-test-day, snapshot, and target window/missingness audit rows."""
    days = test_delivery_days(config, repository_root)
    rows = []
    for snapshot in config["visibility"]["snapshots"]:
        design = build_all_features(panel, snapshot, config)
        for day in days:
            blocks = window_days(day, config)
            for target in config["targets"]:
                row = _audit_row(design, panel, day, snapshot, target, blocks, config)
                row["audit_scope"] = "target_window"
                rows.append(row)
    design = build_all_features(panel, "gate_0730", config)
    rows.extend(_activation_zero_rows(design, days, config))
    audit = pd.DataFrame(rows).sort_values(["delivery_day", "audit_scope", "snapshot", "target", "hour"])
    _require(audit["visibility_assertion"].all(), "Panel audit visibility failed")
    return audit.reset_index(drop=True)


def _activation_zero_rows(design: pd.DataFrame, days: list, config: dict) -> list:
    rows = []
    for day in days:
        blocks = window_days(day, config)
        fit = design.loc[design["delivery_day"].isin(blocks["fit"])
                         & design["target"].eq("activation_volume")]
        actual = design.loc[design["delivery_day"].eq(day)
                            & design["target"].eq("activation_volume")].set_index("hour")
        for hour, group in fit.groupby("hour", sort=True):
            usable = _complete_mask(group, "activation_volume", config) & group["value"].notna()
            values = group.loc[usable, "value"]
            zero_count = int(values.eq(0.0).sum())
            positive_count = int(values.gt(0.0).sum())
            assert zero_count + positive_count == len(values), "negative activation volume"
            assert positive_count > 0, f"{day} hour {hour}: only zero activation events"
            degenerate = zero_count == 0
            actual_positive = bool(actual.loc[hour, "value"] > 0.0)
            if degenerate:
                assert actual_positive, f"{day} hour {hour}: confirmed boundary fact changed"
            rows.append({"delivery_day": day, "snapshot": "gate_0730", "target": "activation_volume",
                         "hour": int(hour), "audit_scope": "activation_zero_cell",
                         "fit_zero_events": zero_count, "fit_positive_events": positive_count,
                         "degenerate_zero_cell": degenerate,
                         "delivery_actual_positive": actual_positive,
                         "visibility_assertion": True})
    return rows


def write_panel_audit(panel: pd.DataFrame, config: dict, repository_root: Path) -> pd.DataFrame:
    """Write the deterministic panel audit CSV configured for S3-A and return it."""
    audit = build_panel_audit(panel, config, repository_root)
    output = repository_root / config["data"]["panel_audit_csv"]
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    return audit


def _reference_checks(config: dict, repository_root: Path) -> None:
    day = config["split"]["reference_delivery_day"]
    blocks = window_days(day, config)
    split = config["split"]
    _require(blocks["fit_calendar"][0] == split["reference_fit_first_day"], "Wrong F start")
    _require(blocks["fit_calendar"][-1] == split["reference_fit_last_day"], "Wrong F end")
    _require(len(blocks["fit"]) == split["reference_fit_available_days"], "Wrong F available days")
    _require(blocks["calibration_calendar"][0] == split["reference_calibration_first_day"],
             "Wrong C start")
    _require(blocks["calibration_calendar"][-1] == split["reference_calibration_last_day"],
             "Wrong C end")
    _require(len(blocks["calibration"]) == config["window"]["probability_calibration"]["calendar_days"],
             "Wrong C available days")
    _require(len(blocks["gap"]) == config["window"]["clearing_gap"]["calendar_days"], "Wrong gap")
    test_delivery_days(config, repository_root)


def _injection_checks(design: pd.DataFrame) -> None:
    check = TestCase()
    source_columns = [column for column in design if column.endswith("_source_day")]
    _require(bool(source_columns), "No source-day column available for leakage check")
    injected_day = design.loc[design["target"].eq("day_ahead_price")].iloc[[0]].copy()
    source_column = "previous_visible_same_hour_source_day"
    injected_day[source_column] = injected_day["delivery_day"]
    check.assertRaises(AssertionError, assert_input_visibility, injected_day)
    injected_late = design.loc[design["target"].eq("activation_volume")].iloc[[0]].copy()
    available_column = "recent_7_visible_same_hour_mean_available_at"
    injected_late[available_column] = injected_late["decision_time_utc"] + pd.Timedelta(minutes=1)
    check.assertRaises(AssertionError, assert_input_visibility, injected_late)


def panel_checks(config: dict, repository_root: Path) -> dict:
    """Run panel, visibility-injection, reference-window, and S2 day-set checks."""
    _reference_checks(config, repository_root)
    panel = build_hourly_panel(config, repository_root)
    reference_day = config["split"]["reference_delivery_day"]
    first_snapshot = next(iter(config["visibility"]["snapshots"]))
    design = build_feature_frame(panel, reference_day, first_snapshot, config)
    _injection_checks(design)
    for snapshot in config["visibility"]["snapshots"]:
        build_snapshot(panel, reference_day, snapshot, config)
    audit = write_panel_audit(panel, config, repository_root)
    return {"panel_rows": len(panel), "audit_rows": len(audit),
            "test_days": config["split"]["expected_test_days"],
            "visibility_assertion": bool(audit["visibility_assertion"].all())}
