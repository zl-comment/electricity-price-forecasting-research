#!/usr/bin/env python3
"""Run the F08 LightGBM external baseline on DK1."""

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
from f01_lear_dk1.backtest import forecast_timestamp, raw_design_matrices
from f01_lear_dk1.data import available_at_gate, load_target_frame
from f01_lear_dk1.settlement import aggregation_cost, build_panels, forecast_driven_arm, settlement_protocol
from f08_lightgbm_dk1.backtest import run_target
from f08_lightgbm_dk1.model import build_matrices


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


def metric_report(actual: pd.Series, forecast: pd.Series, config: dict) -> tuple:
    values = forecast_metrics(actual, forecast, config["metrics"]["rmae_seasonality"])
    undefined = [name for name, value in values.items() if not np.isfinite(value)]
    return {name: value if np.isfinite(value) else None for name, value in values.items()}, undefined


def target_metrics(forecast: pd.DataFrame, persistence: pd.DataFrame, config: dict) -> dict:
    index = pd.date_range("2000-01-01", periods=len(forecast), freq="h")
    actual = pd.Series(forecast["actual"].to_numpy(), index=index)
    model = pd.Series(forecast["forecast"].to_numpy(), index=index)
    baseline = pd.Series(persistence["forecast"].to_numpy(), index=index)
    model_metrics, model_undefined = metric_report(actual, model, config)
    baseline_metrics, baseline_undefined = metric_report(actual, baseline, config)
    metrics = config["metrics"]
    return {"model": model_metrics, "persistence": baseline_metrics,
            "undefined_metrics": {"model": model_undefined, "persistence": baseline_undefined},
            "rmae_index_semantics": "continuous_24-period_evaluation_index_required_by_epftoolbox",
            "dm_semantics": ("epftoolbox DM is one-sided: a small p-value means the forecast named in "
                             "the key is significantly more accurate than the other one"),
            "dm_p_value_model_more_accurate": dm_p_value(
                actual, baseline, model, config["protocol"]["periods_per_day"],
                metrics["dm_norm"], metrics["dm_version"]),
            "dm_p_value_persistence_more_accurate": dm_p_value(
                actual, model, baseline, config["protocol"]["periods_per_day"],
                metrics["dm_norm"], metrics["dm_version"])}


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


def matrix_identity(lear_class, frame: pd.DataFrame, day_map: pd.DataFrame, config: dict) -> dict:
    day = config["split"]["first_test_delivery_day"]
    window = config["protocol"]["calibration_window_days"]
    available = available_at_gate(frame, day, day_map)
    timestamp = forecast_timestamp(day_map, day)
    candidate = build_matrices(lear_class, available, timestamp, window)
    reference = raw_design_matrices(lear_class, frame, day_map, day, window)
    labels = ("Xtrain", "Ytrain", "Xtest")
    allclose = {label: bool(np.allclose(left, right))
                for label, left, right in zip(labels, candidate, reference)}
    maximum = {label: float(np.max(np.abs(left - right)))
               for label, left, right in zip(labels, candidate, reference)}
    if not all(allclose.values()):
        raise ValueError(f"F08 matrices differ from F01: {maximum}")
    shapes = {label: list(left.shape) for label, left in zip(labels, candidate)}
    expected = config["acceptance"]
    if shapes["Xtrain"] != [expected["effective_training_samples"], expected["design_feature_count"]]:
        raise ValueError(f"Unexpected F08 design shape: {shapes}")
    return {"allclose": allclose, "maximum_absolute_difference": maximum, "shapes": shapes}


def dm_pair(first: pd.DataFrame, second: pd.DataFrame, days: list, target: str, config: dict):
    keys = ["delivery_day", "hour"]
    left = first[(first["target"] == target) & first["delivery_day"].isin(days)].sort_values(keys)
    right = second[(second["target"] == target) & second["delivery_day"].isin(days)].sort_values(keys)
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if (not left[keys].equals(right[keys])
            or not np.allclose(left["actual"].to_numpy(), right["actual"].to_numpy())):
        raise ValueError(f"{target}: DM inputs are not aligned")
    metrics = config["metrics"]
    return dm_p_value(pd.Series(left["actual"].to_numpy()), pd.Series(left["forecast"].to_numpy()),
                      pd.Series(right["forecast"].to_numpy()), config["protocol"]["periods_per_day"],
                      metrics["dm_norm"], metrics["dm_version"])


def joint_dm_tests(f01: pd.DataFrame, f08: pd.DataFrame, persistence: pd.DataFrame,
                   days: list, config: dict) -> dict:
    """Report both directions. epftoolbox DM is one-sided on its second forecast argument."""
    named = {"persistence": persistence, "F01_LEAR": f01, "F08_LightGBM": f08}
    pairs = [("persistence", "F01_LEAR"), ("persistence", "F08_LightGBM"),
             ("F08_LightGBM", "F01_LEAR")]
    result = {}
    for target in config["targets"]:
        tests = {"semantics": "a small p-value means the forecast named first in the key is more accurate"}
        for first, second in pairs:
            tests[f"{second}_more_accurate_than_{first}"] = dm_pair(
                named[first], named[second], days, target, config)
            tests[f"{first}_more_accurate_than_{second}"] = dm_pair(
                named[second], named[first], days, target, config)
        result[target] = tests
    return result


def settlement_summary(daily: pd.DataFrame) -> dict:
    return {"days": int(len(daily)), "total_eur": float(daily["total_eur"].sum()),
            "energy_eur": float(daily["energy_eur"].sum()),
            "capacity_eur": float(daily["capacity_eur"].sum()),
            "activation_eur": float(daily["activation_eur"].sum()),
            "recovery_cost_eur": float(daily["recovery_cost_eur"].sum()),
            "exclusivity_violations": int(daily["exclusivity_violations"].sum()),
            "terminal_soc_deviation_mwh_max_abs": float(daily["terminal_soc_deviation_mwh"].abs().max())}


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


def profit_comparison(config: dict, model_total: float) -> dict:
    s2 = json.loads((REPOSITORY_ROOT / config["comparison"]["s2_summary_json"]).read_text())
    f01 = json.loads((REPOSITORY_ROOT / config["comparison"]["f01_summary_json"]).read_text())
    totals = {name: s2["strategies"][name]["total_eur"]["total"] for name in ("B0", "B1", "Oracle")}
    totals["F01_LEAR"] = f01["settlement"]["total_eur"]["total"]
    totals["F08_LightGBM"] = model_total
    return {"total_eur": totals, "over_oracle": {name: float(value / totals["Oracle"])
                                                   for name, value in totals.items()}}


def arm_comparison_table(summary: dict) -> pd.DataFrame:
    """Every arm settled on the same 211 days, plus the B1 to Oracle headroom split."""
    totals = summary["profit_comparison"]["total_eur"]
    split = summary["headroom_decomposition"]
    oracle = totals["Oracle"]
    ladder = [
        ("B0", "persistence", "none", "none", totals["B0"]),
        ("B1", "persistence", "frozen_threshold", "none", totals["B1"]),
        ("F08_LightGBM", "F08 forecast", "frozen_threshold", "none", totals["F08_LightGBM"]),
        ("F01_LEAR", "F01 forecast", "frozen_threshold", "none", totals["F01_LEAR"]),
        ("perfect_hourly_price", "realised hourly mean", "frozen_threshold", "none",
         split["perfect_hourly_price_total_eur"]),
        ("perfect_quarter_price", "realised quarter", "frozen_threshold", "none",
         split["perfect_quarter_price_total_eur"]),
        ("Oracle", "realised quarter", "free", "perfect", oracle),
    ]
    arms = pd.DataFrame(
        [{"row": "arm", "name": name, "price_information": price, "reserve_commitment": reserve,
          "activation_information": activation, "eur": value, "share_of_oracle": value / oracle}
         for name, price, reserve, activation, value in ladder])
    components = pd.DataFrame(
        [{"row": "headroom_component", "name": name, "eur": split[f"{name}_eur"],
          "share_of_b1_to_oracle_gap": split[f"{name}_share"]}
         for name in ("forecast_error", "time_resolution", "decision_layer")])
    return pd.concat([arms, components], ignore_index=True)


def write_outputs(summary: dict, forecasts: pd.DataFrame, daily: pd.DataFrame,
                  day_map: pd.DataFrame, output: dict) -> Path:
    directory = REPOSITORY_ROOT / output["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(directory / output["forecasts_csv"], index=False)
    daily.to_csv(directory / output["daily_csv"], index=False)
    arm_comparison_table(summary).to_csv(directory / output["arm_comparison_csv"], index=False)
    day_map.to_csv(directory / output["day_index_map_csv"], index_label="synthetic_timestamp")
    with (directory / output["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    return directory


def build_summary(config: dict, runtime: dict, identities: dict, forecasts: pd.DataFrame,
                  persistence: pd.DataFrame, f01: pd.DataFrame, test_days: list,
                  settled_days: list, threshold: float, daily: pd.DataFrame,
                  aggregation: dict, adjusted: int) -> dict:
    shape = next(iter(identities.values()))["shapes"]
    comparison = profit_comparison(config, float(daily["total_eur"].sum()))
    return {
        "method_role": "external baseline; not an original-paper reproduction",
        "index_semantics": "synthetic_civil_day_not_utc",
        "matrix_feature_source": "unscaled_pinned_epftoolbox_LEAR__build_and_split_XYs",
        "feature_scaling": "none",
        "exogenous_feature_count": 0,
        "design_limit": "price lags plus weekday dummies only; cross-market exogenous inputs give n < p",
        "environment_fingerprint": environment_fingerprint(),
        "random_seed": config["protocol"]["random_seed"],
        "calibration_window_days": config["protocol"]["calibration_window_days"],
        "design_feature_count": shape["Xtrain"][1],
        "effective_training_samples": shape["Xtrain"][0],
        "n_over_p": float(shape["Xtrain"][0] / shape["Xtrain"][1]),
        "matrix_identity_with_F01": identities,
        "runtime": runtime,
        "model": config["model"],
        "days_with_dst_length_adjustment": adjusted,
        "reserve_threshold_eur_per_mw": threshold,
        "prediction_test_days": len(test_days),
        "settled_test_days": len(settled_days),
        "prediction_metrics_212_days": prediction_summary(forecasts, persistence, test_days, config),
        "prediction_metrics_211_common_days": prediction_summary(forecasts, persistence, settled_days, config),
        "dm_tests_211_common_days": joint_dm_tests(f01, forecasts, persistence, settled_days, config),
        "settlement": settlement_summary(daily),
        "aggregation_cost": aggregation,
        "headroom_decomposition": headroom_decomposition(comparison["total_eur"], aggregation,
                                                       "F08_LightGBM"),
        "profit_comparison": comparison,
    }


def load_inputs(config: dict, lear_class, root: Path, test_days: list) -> tuple:
    frames, maps, identities, forecasts, persistence = {}, {}, {}, [], []
    for target in config["targets"]:
        frames[target], maps[target] = load_target_frame(root, config, target)
        identities[target] = matrix_identity(lear_class, frames[target], maps[target], config)
        result = run_target(lear_class, frames[target], maps[target], test_days, config)
        forecasts.append(result.assign(target=target))
        baseline = persistence_forecast(frames[target], maps[target], test_days)
        persistence.append(baseline.assign(target=target))
    if not maps["day_ahead_price"].equals(maps["capacity_price"]):
        raise ValueError("Target day-index maps differ")
    columns = ["delivery_day", "hour", "target", "forecast", "actual"]
    forecast_frame = pd.concat(forecasts, ignore_index=True)[columns]
    persistence_frame = pd.concat(persistence, ignore_index=True)[columns]
    return maps, identities, forecast_frame, persistence_frame


def main() -> None:
    config = yaml.safe_load(parse_arguments().config.read_text(encoding="utf-8"))
    if lightgbm.__version__ != config["model"]["lightgbm_version"]:
        raise RuntimeError(f"Unexpected LightGBM version: {lightgbm.__version__}")
    np.random.seed(config["protocol"]["random_seed"])
    root = REPOSITORY_ROOT / config["data"]["root"]
    lear_class, runtime = load_paper_lear(config["runtime"])
    usable = usable_delivery_days(config["window"])
    split = split_boundaries(config, usable)
    test_days = [day for day in usable if day >= split["first_test_delivery_day"]]
    maps, identities, forecast_frame, persistence_frame = load_inputs(
        config, lear_class, root, test_days
    )
    f01 = pd.read_csv(REPOSITORY_ROOT / config["comparison"]["f01_forecasts_csv"])
    panels = build_panels(root, config)
    settled_days, threshold = settlement_protocol(panels, config)
    settled_forecasts = forecast_frame[forecast_frame["delivery_day"].isin(settled_days)]
    storage, arm = storage_settings(config), config["storage"]["arms"]["main"]
    daily = pd.DataFrame(forecast_driven_arm(panels, settled_forecasts, threshold, arm,
                                             storage, config["solver"]))
    daily.insert(1, "strategy", "F08_LightGBM")
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
    summary = build_summary(config, runtime, identities, forecast_frame, persistence_frame, f01,
                            test_days, settled_days, threshold, daily, aggregation, adjusted)
    directory = write_outputs(summary, forecast_frame, daily, maps["day_ahead_price"], config["output"])
    print(json.dumps(summary["profit_comparison"], indent=2, sort_keys=True), flush=True)
    print(f"F08 results written to {directory}", flush=True)


if __name__ == "__main__":
    main()
