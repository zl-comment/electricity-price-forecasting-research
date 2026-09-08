#!/usr/bin/env python3
"""Run the F01 LEAR same-protocol transfer evaluation on DK1."""

import argparse
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from epf_harness.backtest import load_paper_lear
from epf_harness.dk1_coupling import split_boundaries, usable_delivery_days
from epf_harness.metrics import dm_p_value, forecast_metrics
from f01_lear_dk1.backtest import design_shape, run_target
from f01_lear_dk1.data import load_target_frame
from f01_lear_dk1.settlement import aggregation_cost, build_panels, forecast_driven_arm, settlement_protocol


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def environment_fingerprint() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__, "lightgbm": lightgbm.__version__}


def storage_settings(config: dict) -> dict:
    return dict(config["storage"], slot_hours=config["protocol"]["slot_hours"],
                soc_tolerance_mwh=config["solver"]["soc_tolerance_mwh"],
                recovery_constraint_tolerance_mwh=config["solver"]["recovery_constraint_tolerance_mwh"])


def persistence_forecast(frame: pd.DataFrame, day_map: pd.DataFrame, days: list) -> pd.DataFrame:
    rows = []
    for day in days:
        previous = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        source = frame.loc[day_map.index[day_map["delivery_day"] == previous], "Price"].to_numpy()
        actual = frame.loc[day_map.index[day_map["delivery_day"] == day], "Price"].to_numpy()
        if len(source) != 24 or len(actual) != 24:
            raise ValueError(f"{day}: persistence input does not contain 24 periods")
        rows.extend({"delivery_day": day, "hour": hour, "forecast": float(source[hour]),
                     "actual": float(actual[hour])} for hour in range(24))
    return pd.DataFrame(rows)


def target_metrics(forecast: pd.DataFrame, persistence: pd.DataFrame, config: dict) -> dict:
    index = pd.date_range("2000-01-01", periods=len(forecast), freq="h")
    actual = pd.Series(forecast["actual"].to_numpy(), index=index)
    model = pd.Series(forecast["forecast"].to_numpy(), index=index)
    baseline = pd.Series(persistence["forecast"].to_numpy(), index=index)
    metrics = config["metrics"]
    model_metrics = forecast_metrics(actual, model, metrics["rmae_seasonality"])
    persistence_metrics = forecast_metrics(actual, baseline, metrics["rmae_seasonality"])
    undefined = {"model": [name for name, value in model_metrics.items() if not np.isfinite(value)],
                 "persistence": [name for name, value in persistence_metrics.items() if not np.isfinite(value)]}
    model_metrics = {name: value if np.isfinite(value) else None for name, value in model_metrics.items()}
    persistence_metrics = {name: value if np.isfinite(value) else None
                           for name, value in persistence_metrics.items()}
    return {
        "model": model_metrics,
        "persistence": persistence_metrics,
        "undefined_metrics": undefined,
        "rmae_index_semantics": "continuous_24-period_evaluation_index_required_by_epftoolbox",
        "dm_semantics": ("epftoolbox DM is one-sided: a small p-value means the forecast named in the "
                         "key is significantly more accurate than the other one"),
        "dm_p_value_model_more_accurate": dm_p_value(
            actual, baseline, model, config["protocol"]["periods_per_day"],
            metrics["dm_norm"], metrics["dm_version"]),
        "dm_p_value_persistence_more_accurate": dm_p_value(
            actual, model, baseline, config["protocol"]["periods_per_day"],
            metrics["dm_norm"], metrics["dm_version"]),
    }


def prediction_summary(forecasts: pd.DataFrame, persistence: pd.DataFrame,
                       days: list, config: dict) -> dict:
    result = {}
    for target in config["targets"]:
        model = forecasts[(forecasts["target"] == target) & forecasts["delivery_day"].isin(days)]
        base = persistence[(persistence["target"] == target) & persistence["delivery_day"].isin(days)]
        model = model.sort_values(["delivery_day", "hour"]).reset_index(drop=True)
        base = base.sort_values(["delivery_day", "hour"]).reset_index(drop=True)
        if not model[["delivery_day", "hour", "actual"]].equals(base[["delivery_day", "hour", "actual"]]):
            raise ValueError(f"{target}: model and persistence evaluation rows differ")
        result[target] = target_metrics(model, base, config)
    return result


def describe(series: pd.Series) -> dict:
    return {"n": int(series.count()), "mean": float(series.mean()), "std": float(series.std(ddof=0)),
            "min": float(series.min()), "p50": float(series.quantile(0.5)), "max": float(series.max()),
            "total": float(series.sum())}


def settlement_summary(daily: pd.DataFrame) -> dict:
    columns = ["total_eur", "energy_eur", "capacity_eur", "activation_eur",
               "shortfall_cost_eur", "recovery_cost_eur", "recovery_grid_mwh",
               "required_activation_mwh", "delivered_activation_mwh", "reserve_energy_gap_mwh",
               "reserve_power_gap_mwh", "committed_mw_hours", "cycles",
               "base_terminal_soc_deviation_mwh", "terminal_soc_deviation_mwh"]
    return {"days": int(len(daily)),
            "exclusivity_violations": int(daily["exclusivity_violations"].sum()),
            "days_with_shortfall": int((daily["slots_with_shortfall"] > 0).sum()),
            **{column: describe(daily[column]) for column in columns}}


def build_summary(config: dict, runtime: dict, shapes: dict, forecasts: pd.DataFrame,
                  persistence: pd.DataFrame, test_days: list, settled_days: list,
                  threshold: float, daily: pd.DataFrame, aggregation: dict, adjusted: int) -> dict:
    unique_shapes = {(item["design_feature_count"], item["effective_training_samples"])
                     for item in shapes.values()}
    if len(unique_shapes) != 1:
        raise ValueError(f"Target design shapes differ: {shapes}")
    shape = next(iter(shapes.values()))
    comparison = load_s2_comparison(config, float(daily["total_eur"].sum()))
    return {
        "method_role": "same-protocol transfer evaluation and external baseline",
        "index_semantics": "synthetic_civil_day_not_utc",
        "feature_scaling": "LEAR invariant scaling inside recalibrate_predict",
        "exogenous_feature_count": 0,
        "design_limit": "price lags plus weekday dummies only; cross-market exogenous inputs give n < p",
        "environment_fingerprint": environment_fingerprint(),
        "random_seed": config["protocol"]["random_seed"],
        "calibration_window_days": config["protocol"]["calibration_window_days"],
        "design_feature_count": shape["design_feature_count"],
        "effective_training_samples": shape["effective_training_samples"],
        "n_over_p": shape["n_over_p"],
        "design_by_target": shapes,
        "runtime": runtime,
        "days_with_dst_length_adjustment": adjusted,
        "reserve_threshold_eur_per_mw": threshold,
        "prediction_test_days": len(test_days),
        "settled_test_days": len(settled_days),
        "prediction_metrics_212_days": prediction_summary(forecasts, persistence, test_days, config),
        "prediction_metrics_211_common_days": prediction_summary(forecasts, persistence, settled_days, config),
        "settlement": settlement_summary(daily),
        "aggregation_cost": aggregation,
        "headroom_decomposition": headroom_decomposition(comparison["total_eur"], aggregation,
                                                       "F01_LEAR"),
        "profit_comparison": comparison,
    }


def headroom_decomposition(totals: dict, aggregation: dict, model_name: str) -> dict:
    """Split the frozen B1 -> Oracle headroom into forecast error, resolution and decision layer.

    Every arm here except Oracle plans with the same frozen threshold reserve rule, so the residual
    named `decision_layer_eur` is what perfect prices cannot buy under that rule.
    """
    b1, oracle = totals["B1"], totals["Oracle"]
    hourly, quarter = aggregation["hourly_total_eur"], aggregation["quarter_total_eur"]
    gap = oracle - b1
    parts = {
        "forecast_error_eur": hourly - b1,
        "time_resolution_eur": quarter - hourly,
        "decision_layer_eur": oracle - quarter,
    }
    return {
        "b1_to_oracle_gap_eur": gap,
        "perfect_hourly_price_total_eur": hourly,
        "perfect_quarter_price_total_eur": quarter,
        **parts,
        **{name.replace("_eur", "_share"): value / gap for name, value in parts.items()},
        "forecast_headroom_captured": (totals[model_name] - b1) / (hourly - b1),
        "definition": ("perfect-price arms use realised day-ahead and capacity prices under the same "
                       "frozen threshold rule; only Oracle co-optimises with perfect activation"),
    }


def load_s2_comparison(config: dict, model_total: float) -> dict:
    path = REPOSITORY_ROOT / config["comparison"]["s2_summary_json"]
    s2 = json.loads(path.read_text(encoding="utf-8"))
    totals = {name: s2["strategies"][name]["total_eur"]["total"] for name in ("B0", "B1", "Oracle")}
    totals["F01_LEAR"] = model_total
    return {"total_eur": totals, "over_oracle": {name: float(value / totals["Oracle"])
                                                   for name, value in totals.items()}}


def write_outputs(summary: dict, forecasts: pd.DataFrame, daily: pd.DataFrame,
                  day_map: pd.DataFrame, output: dict) -> Path:
    directory = REPOSITORY_ROOT / output["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(directory / output["forecasts_csv"], index=False)
    daily.to_csv(directory / output["daily_csv"], index=False)
    day_map.to_csv(directory / output["day_index_map_csv"], index_label="synthetic_timestamp")
    with (directory / output["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    return directory


def main() -> None:
    config = yaml.safe_load(parse_arguments().config.read_text(encoding="utf-8"))
    np.random.seed(config["protocol"]["random_seed"])
    root = REPOSITORY_ROOT / config["data"]["root"]
    lear_class, runtime = load_paper_lear(config["runtime"])
    usable = usable_delivery_days(config["window"])
    split = split_boundaries(config, usable)
    test_days = [day for day in usable if day >= split["first_test_delivery_day"]]
    frames, maps, shapes, forecasts, persistence = {}, {}, {}, [], []
    for target in config["targets"]:
        frames[target], maps[target] = load_target_frame(root, config, target)
        shapes[target] = design_shape(lear_class, frames[target], maps[target], config)
        result = run_target(lear_class, frames[target], maps[target], test_days, config, runtime)
        forecasts.append(result.assign(target=target))
        persistence.append(persistence_forecast(frames[target], maps[target], test_days).assign(target=target))
    if not maps["day_ahead_price"].equals(maps["capacity_price"]):
        raise ValueError("Target day-index maps differ")
    columns = ["delivery_day", "hour", "target", "forecast", "actual"]
    forecast_frame = pd.concat(forecasts, ignore_index=True)[columns]
    persistence_frame = pd.concat(persistence, ignore_index=True)[columns]
    panels = build_panels(root, config)
    settled_days, threshold = settlement_protocol(panels, config)
    settled_forecasts = forecast_frame[forecast_frame["delivery_day"].isin(settled_days)]
    storage, arm = storage_settings(config), config["storage"]["arms"]["main"]
    daily = pd.DataFrame(forecast_driven_arm(panels, settled_forecasts, threshold, arm,
                                             storage, config["solver"]))
    daily.insert(1, "strategy", "F01_LEAR")
    aggregation = aggregation_cost({day: panels[day] for day in settled_days}, threshold, arm,
                                   storage, config["solver"])
    daily = daily.merge(aggregation.pop("daily"), on="delivery_day", validate="one_to_one")
    terminal = daily["terminal_soc_deviation_mwh"].abs().max()
    if terminal >= config["solver"]["soc_tolerance_mwh"]:
        raise ValueError(f"Terminal SOC tolerance failed: {terminal}")
    adjusted = maps["day_ahead_price"].loc[maps["day_ahead_price"]["dst_length_adjusted"],
                                           "delivery_day"].nunique()
    if adjusted != config["acceptance"]["days_with_dst_length_adjustment"]:
        raise ValueError(f"Unexpected DST adjustment count: {adjusted}")
    summary = build_summary(config, runtime, shapes, forecast_frame, persistence_frame,
                            test_days, settled_days, threshold, daily, aggregation, adjusted)
    directory = write_outputs(summary, forecast_frame, daily, maps["day_ahead_price"], config["output"])
    print(json.dumps(summary["profit_comparison"], indent=2, sort_keys=True), flush=True)
    print(f"F01 results written to {directory}", flush=True)


if __name__ == "__main__":
    main()
