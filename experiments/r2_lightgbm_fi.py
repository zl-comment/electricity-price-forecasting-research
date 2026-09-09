#!/usr/bin/env python3
"""Run the R2 Finnish aFRR partial reproduction."""

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

from f08_lightgbm_dk1.model import fit_predict_hourly
from r2_lightgbm_fi.data import chronological_splits, load_hourly_grid, split_metadata
from r2_lightgbm_fi.features import build_features


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def environment_fingerprint() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
            "lightgbm": lightgbm.__version__}


def validate_model_contract(config: dict) -> None:
    source_path = REPOSITORY_ROOT / config["model"]["source_config"]
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if config["model"]["params"] != source["model"]["params"]:
        raise ValueError("R2 LightGBM parameters differ from F08 model.params")
    if lightgbm.__version__ != config["model"]["lightgbm_version"]:
        raise ValueError(f"Unexpected LightGBM version: {lightgbm.__version__}")
    if config["model"]["hyperparameters"] != "inherited_from_f08_dk1_not_tuned":
        raise ValueError("R2 hyperparameter provenance is not pinned")


def training_matrices(frame: pd.DataFrame, features: pd.DataFrame,
                      splits: dict, horizon: int, target: str) -> tuple:
    train = splits["train"]
    first = frame.index.get_loc(train.index[0]) - 1
    last = frame.index.get_loc(train.index[-1]) - horizon
    origins = np.arange(first, last + 1)
    labels = np.asarray([[frame[target].iloc[pos + step] for step in range(1, horizon + 1)]
                         for pos in origins], dtype=float)
    complete_labels = np.isfinite(labels).all(axis=1)
    Xtrain = features.iloc[origins].to_numpy(dtype=float)[complete_labels]
    Ytrain = labels[complete_labels]
    if len(Xtrain) == 0 or not np.isfinite(Ytrain).all():
        raise ValueError(f"{target}: invalid direct multi-step training labels")
    metadata = {"candidate_samples": int(len(origins)),
                "excluded_for_missing_labels": int((~complete_labels).sum()),
                "effective_samples": int(complete_labels.sum()),
                "feature_count": int(Xtrain.shape[1]), "target_horizons": int(Ytrain.shape[1])}
    return Xtrain, Ytrain, metadata


def test_origins(frame: pd.DataFrame, test: pd.DataFrame, horizon: int) -> np.ndarray:
    first = frame.index.get_loc(test.index[0]) - 1
    origins = np.arange(first, first + len(test), horizon)
    if len(origins) * horizon != len(test):
        raise ValueError("Continuous test block cannot be tiled by the forecast horizon")
    return origins


def forecast_target(frame: pd.DataFrame, features: pd.DataFrame, splits: dict,
                    config: dict, target: str) -> tuple:
    protocol, model = config["protocol"], config["model"]
    horizon = protocol["forecast_horizon_hours"]
    Xtrain, Ytrain, training = training_matrices(frame, features, splits, horizon, target)
    rows = []
    for origin in test_origins(frame, splits["test"], horizon):
        prediction = fit_predict_hourly(
            Xtrain, Ytrain, features.iloc[[origin]].to_numpy(dtype=float),
            model["params"], protocol["random_seed"])
        for step, value in enumerate(prediction, start=1):
            timestamp = frame.index[origin + step]
            rows.append({"target": target, "forecast_origin_utc": frame.index[origin].isoformat(),
                         "horizon_hour": step, "timestamp_utc": timestamp.isoformat(),
                         "actual": float(frame[target].iloc[origin + step]),
                         "forecast": float(value)})
    result = pd.DataFrame(rows)
    if len(result) != len(splits["test"]) or not np.isfinite(result[["actual", "forecast"]]).all().all():
        raise ValueError(f"{target}: invalid continuous test forecast")
    return result, training


def metrics(actual: np.ndarray, forecast: np.ndarray, seasonality: int) -> dict:
    error = forecast - actual
    seasonal = actual[seasonality:] - actual[:-seasonality]
    if not np.isfinite(actual).all() or len(seasonal) == 0:
        raise ValueError("Metric inputs are invalid")
    return {
        "mae": float(np.mean(np.abs(error))),
        "nmae": float(np.sum(np.abs(error)) / np.sum(np.abs(actual))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mase": float(np.mean(np.abs(error)) / np.mean(np.abs(seasonal))),
        "rmsse": float(np.sqrt(np.mean(error ** 2) / np.mean(seasonal ** 2))),
    }


def metric_comparison(forecasts: pd.DataFrame, config: dict) -> dict:
    result = {}
    for target in ("Down", "Up"):
        rows = forecasts[forecasts["target"] == target].sort_values("timestamp_utc")
        local = metrics(rows["actual"].to_numpy(), rows["forecast"].to_numpy(),
                        config["protocol"]["seasonal_period_hours"])
        paper = config["paper_reported"][target]
        if set(local) != set(paper):
            raise ValueError(f"{target}: paper and local metric keys differ")
        result[target] = {
            "paper_reported": paper,
            "local": local,
            "local_minus_paper": {name: local[name] - paper[name] for name in local},
            "absolute_deviation_percent": {
                name: abs(local[name] - paper[name]) / abs(paper[name]) * 100 for name in local},
        }
    return result


def paper_metric_audit(forecasts: pd.DataFrame, comparison: dict, seasonality: int) -> dict:
    result = {}
    for target in ("Down", "Up"):
        rows = forecasts[forecasts["target"] == target].sort_values("timestamp_utc")
        actual = rows["actual"].to_numpy()
        seasonal = actual[seasonality:] - actual[:-seasonality]
        paper = comparison[target]["paper_reported"]
        implied = {"mase_denominator": paper["mae"] / paper["mase"],
                   "rmsse_denominator": paper["rmse"] / paper["rmsse"]}
        observed = {"mase_denominator": float(np.mean(np.abs(seasonal))),
                    "rmsse_denominator": float(np.sqrt(np.mean(seasonal ** 2)))}
        result[target] = {
            "paper_implied_seasonal_scale": implied,
            "published_test_target_seasonal_scale_m48": observed,
            "consistent_with_reported_formula": all(
                np.isclose(implied[name], observed[name], rtol=0.01) for name in implied),
        }
    return result


def build_summary(config: dict, frame: pd.DataFrame, splits: dict, features: pd.DataFrame,
                  training: dict, comparison: dict, metric_audit: dict) -> dict:
    judgement = {
        target: ("better_than_seasonal_naive" if all(
            comparison[target]["local"][metric] < 1.0 for metric in ("mase", "rmsse"))
                 else "not_better_than_seasonal_naive")
        for target in ("Down", "Up")
    }
    judgement["overall"] = "mixed_result_no_pass_fail_threshold"
    data = split_metadata(frame, splits)
    data.update({"source_csv": config["data"]["csv"], "sha256": config["data"]["sha256"]})
    return {
        "method_role": "partial reproduction; independent reimplementation",
        "reproduction_boundary": ("Data, split, and seed are aligned, and metrics implement the reported "
                                  "formulas. Model-output differences can only be attributed at the "
                                  "feature-subset and hyperparameter level."),
        "free_parameter_count": 2,
        "free_parameters": {
            "feature_selection": config["features"]["selection"],
            "hyperparameters": config["model"]["hyperparameters"],
        },
        "preprocessing": config["protocol"]["preprocessing"],
        "preprocessing_effect_on_gap": "no_local_choice_no_additional_divergence",
        "preprocessing_verification_boundary": ("The authors' 15-minute imputation and hourly aggregation "
                                                "cannot be independently verified from the published archive."),
        "environment_fingerprint": environment_fingerprint(),
        "random_seed": config["protocol"]["random_seed"],
        "data": data,
        "features": config["features"],
        "feature_count": int(features.shape[1]),
        "feature_names": features.columns.tolist(),
        "feature_time_semantics": "all inputs are observed at or before each forecast origin",
        "training": training,
        "forecast_horizon_hours": config["protocol"]["forecast_horizon_hours"],
        "evaluation": config["protocol"]["evaluation"],
        "validation_usage": "none_hyperparameters_inherited_from_f08",
        "metric_protocol": {
            "reported": ["mae", "nmae", "rmse", "mase", "rmsse"],
            "seasonal_period_hours": config["protocol"]["seasonal_period_hours"],
            "scaled_metric_denominator": "continuous_test_target_seasonal_differences",
        },
        "training_data_updates_during_test": False,
        "shared_f08_fit_calls_per_target": int(len(splits["test"]) /
                                                config["protocol"]["forecast_horizon_hours"]),
        "outlier_removal": config["protocol"]["outlier_removal"],
        "model": config["model"],
        "metrics": comparison,
        "paper_metric_consistency_audit": metric_audit,
        "mase_rmsse_judgement": judgement,
    }


def write_outputs(summary: dict, forecasts: pd.DataFrame, config: dict) -> Path:
    output = REPOSITORY_ROOT / config["output"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output / config["output"]["forecasts_csv"], index=False)
    with (output / config["output"]["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    return output


def main() -> None:
    config = yaml.safe_load(parse_arguments().config.read_text(encoding="utf-8"))
    validate_model_contract(config)
    np.random.seed(config["protocol"]["random_seed"])
    frame = load_hourly_grid(REPOSITORY_ROOT / config["data"]["csv"], config["data"])
    splits = chronological_splits(frame, config["protocol"])
    features = build_features(frame, config["features"])
    frames, training = [], {}
    for target in ("Down", "Up"):
        forecast, training[target] = forecast_target(frame, features, splits, config, target)
        frames.append(forecast)
    forecasts = pd.concat(frames, ignore_index=True)
    comparison = metric_comparison(forecasts, config)
    metric_audit = paper_metric_audit(
        forecasts, comparison, config["protocol"]["seasonal_period_hours"])
    summary = build_summary(config, frame, splits, features, training, comparison, metric_audit)
    output = write_outputs(summary, forecasts, config)
    print(json.dumps({"metrics": comparison,
                      "mase_rmsse_judgement": summary["mase_rmsse_judgement"]},
                     indent=2, sort_keys=True), flush=True)
    print(f"R2 results written to {output}", flush=True)


if __name__ == "__main__":
    main()
