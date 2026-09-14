"""Cross-market predictors, hour effects and specification configs for the S3 cross-market study."""

import copy

import numpy as np
import pandas as pd

from src.s3_forecast_side import panel


MODIFIED_TARGETS = ("day_ahead_price", "capacity_price")
GRANULARITIES = {"target_hour": "target_hour", "pooled_hours_with_hour_effects": "pooled_hours"}


def _source_days(delivery_days: pd.Series, offset: int) -> np.ndarray:
    shifted = pd.to_datetime(delivery_days) + pd.to_timedelta(int(offset), unit="D")
    return shifted.dt.strftime("%Y-%m-%d").to_numpy()


def _daily_table(hourly_panel: pd.DataFrame, target: str, statistic: str,
                 spread_hours: int) -> pd.DataFrame:
    rows = []
    for day, group in hourly_panel.loc[hourly_panel["target"].eq(target)].groupby("delivery_day"):
        values = np.sort(group["value"].to_numpy(dtype=float))
        assert values.size == 24 and np.isfinite(values).all(), f"{target} {day}: incomplete source day"
        if statistic == "daily_mean":
            value = float(values.mean())
        elif statistic == "top_minus_bottom_mean":
            value = float(values[-spread_hours:].mean() - values[:spread_hours].mean())
        else:
            raise AssertionError(f"unknown daily statistic: {statistic}")
        rows.append({"delivery_day": day, "value": value, "available_at": group["available_at"].max()})
    return pd.DataFrame(rows).set_index("delivery_day")


def _feature_values(rows: pd.DataFrame, hourly_panel: pd.DataFrame, settings: dict,
                    config: dict) -> tuple:
    """Look up one predictor for the given design rows from the realized hourly panel."""
    sources = _source_days(rows["delivery_day"], settings["day_offset"])
    target = settings["source_target"]
    if settings["statistic"] == "same_hour":
        table = hourly_panel.loc[hourly_panel["target"].eq(target)].set_index(["delivery_day", "hour"])
        index = pd.MultiIndex.from_arrays([sources, rows["hour"].to_numpy()])
    else:
        table = _daily_table(hourly_panel, target, settings["statistic"],
                             int(config["cross_market"]["spread_hours"]))
        index = pd.Index(sources)
    selected = table.reindex(index)
    values = selected["value"].to_numpy(dtype=float)
    available = pd.Series(pd.DatetimeIndex(selected["available_at"]), index=rows.index)
    return values, available, sources


def _add_features(design: pd.DataFrame, hourly_panel: pd.DataFrame, target: str,
                  definitions: dict, config: dict, record_source: bool) -> None:
    rows = design.loc[design["target"].eq(target)]
    for name, settings in definitions.items():
        assert name not in design.columns, f"{name} already exists in the design"
        values, available, sources = _feature_values(rows, hourly_panel, settings, config)
        design[name] = pd.Series(values, index=rows.index).reindex(design.index)
        design[f"{name}_available_at"] = available.reindex(design.index)
        if record_source:
            recorded = np.where(np.isfinite(values), sources, None)
            design[f"{name}_source_day"] = pd.Series(recorded, index=rows.index).reindex(design.index)


def hour_indicator_columns(config: dict) -> list:
    prefix = config["cross_market"]["hour_indicator_prefix"]
    return [f"{prefix}{hour:02d}" for hour in range(1, int(config["protocol"]["periods_per_day"]))]


def _add_hour_indicators(design: pd.DataFrame, config: dict) -> None:
    for hour, name in enumerate(hour_indicator_columns(config), start=1):
        assert name not in design.columns, f"{name} already exists in the design"
        design[name] = design["hour"].eq(hour).astype(int)
        design[f"{name}_available_at"] = design["decision_time_utc"]


def build_cross_design(design: pd.DataFrame, hourly_panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach 07:30-visible cross-market predictors and hour indicators to the S3-A design."""
    output = design.copy()
    for target, definitions in config["cross_market"]["cross_features"].items():
        assert target in MODIFIED_TARGETS, f"{target} is not a modified target"
        assert all(int(settings["day_offset"]) < 0 for settings in definitions.values()), \
            "07:30 cross-market predictors must come from days before delivery"
        _add_features(output, hourly_panel, target, definitions, config, True)
    _add_hour_indicators(output, config)
    keys = ["delivery_day", "target", "hour"]
    assert output[keys].equals(design[keys]), "cross design reordered the S3-A rows"
    panel.assert_input_visibility(output)
    return output


def build_conditional_design(cross_design: pd.DataFrame, hourly_panel: pd.DataFrame,
                             config: dict) -> pd.DataFrame:
    """Attach realized delivery-day capacity prices to day-ahead rows at the 12:00 gate."""
    settings = config["cross_market"]["conditional_1200"]
    assert all(int(item["day_offset"]) == 0 for item in settings["features"].values())
    output = cross_design.copy()
    selected = output["target"].eq(settings["target"])
    early = output.loc[selected, "decision_time_utc"].copy()
    days = output.loc[selected, "delivery_day"]
    late = {day: panel.decision_time_utc(day, settings["snapshot"], config) for day in days.unique()}
    output.loc[selected, "decision_time_utc"] = days.map(late)
    _add_features(output, hourly_panel, settings["target"], settings["features"], config, False)
    rows = output.loc[selected]
    panel.assert_input_visibility(rows)
    for name in settings["features"]:
        finite = rows[name].notna()
        stamps = pd.to_datetime(rows.loc[finite, f"{name}_available_at"], utc=True)
        assert stamps.gt(early[finite]).all(), f"{name} is already visible at 07:30"
    return output


def spec_config(config: dict, spec: str, conditional: bool = False) -> dict:
    """Return a config whose modified-target features and granularity follow one specification."""
    settings = config["cross_market"]["specs"][spec]
    output = copy.deepcopy(config)
    conditional_settings = config["cross_market"]["conditional_1200"]
    for target in MODIFIED_TARGETS:
        extra = list(config["cross_market"]["cross_features"][target]) if settings["cross"] else []
        if conditional and target == conditional_settings["target"]:
            extra += list(conditional_settings["features"])
        granularity = GRANULARITIES[settings["granularity"]]
        if granularity == "pooled_hours":
            extra += hour_indicator_columns(config)
        output["targets"][target]["model_granularity"] = granularity
        output["features"][target] = list(config["features"][target]) + extra
    return output
