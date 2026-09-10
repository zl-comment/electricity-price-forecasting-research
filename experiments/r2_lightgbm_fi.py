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

from r2_lightgbm_fi.data import chronological_splits, load_hourly_grid, split_metadata
from r2_lightgbm_fi.features import build_features
from r2_lightgbm_fi.model import fit_hourly_models, predict_hourly_models


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
    expected = {"learning_rate", "num_iterations", "num_leaves", "max_depth",
                "lambda_l1", "lambda_l2"}
    if set(config["tuning"]["search_space"]) != expected:
        raise ValueError("R2 tuning must cover only the paper-named parameter classes")
    if config["tuning"]["tuner"] != "seeded_random_search":
        raise ValueError("R2 tuner must be the declared seeded random search")
    if config["tuning"]["optuna_used"]:
        raise ValueError("R2 must not use Optuna")


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


def validation_origins(frame: pd.DataFrame, validation: pd.DataFrame,
                       horizon: int) -> tuple:
    first = frame.index.get_loc(validation.index[0]) - 1
    usable_hours = len(validation) // horizon * horizon
    origins = np.arange(first, first + usable_hours, horizon)
    metadata = {
        "validation_hours": int(len(validation)),
        "used_validation_hours": int(usable_hours),
        "dropped_validation_hours": int(len(validation) - usable_hours),
        "validation_origins": int(len(origins)),
    }
    return origins, metadata


def forecast_target(frame: pd.DataFrame, features: pd.DataFrame, splits: dict,
                    config: dict, target: str, params: dict, variant: str) -> tuple:
    protocol = config["protocol"]
    horizon = protocol["forecast_horizon_hours"]
    Xtrain, Ytrain, training = training_matrices(frame, features, splits, horizon, target)
    boosters = fit_hourly_models(
        Xtrain, Ytrain, params, protocol["random_seed"], config["runtime"]["fit_workers"])
    rows = []
    for origin in test_origins(frame, splits["test"], horizon):
        prediction = predict_hourly_models(
            boosters, features.iloc[[origin]].to_numpy(dtype=float))
        for step, value in enumerate(prediction, start=1):
            timestamp = frame.index[origin + step]
            rows.append({"variant": variant, "target": target,
                         "forecast_origin_utc": frame.index[origin].isoformat(),
                         "horizon_hour": step, "timestamp_utc": timestamp.isoformat(),
                         "actual": float(frame[target].iloc[origin + step]),
                         "forecast": float(value)})
    result = pd.DataFrame(rows)
    if len(result) != len(splits["test"]) or not np.isfinite(result[["actual", "forecast"]]).all().all():
        raise ValueError(f"{target}: invalid continuous test forecast")
    return result, training


def sampled_value(rng: np.random.RandomState, specification: dict):
    sampling = specification["sampling"]
    if sampling == "log_uniform":
        return float(np.exp(rng.uniform(
            np.log(specification["minimum"]), np.log(specification["maximum"]))))
    if sampling == "integer_log_uniform":
        value = int(np.rint(np.exp(rng.uniform(
            np.log(specification["minimum"]), np.log(specification["maximum"])))))
        return int(np.clip(value, specification["minimum"], specification["maximum"]))
    if sampling == "uniform_choice":
        return int(rng.choice(specification["choices"]))
    if sampling == "zero_or_log_uniform":
        if rng.uniform() < specification["zero_probability"]:
            return 0.0
        return float(np.exp(rng.uniform(
            np.log(specification["minimum"]), np.log(specification["maximum"]))))
    raise ValueError(f"Unknown tuning sampler: {sampling}")


def sample_parameters(base: dict, search_space: dict,
                      rng: np.random.RandomState) -> tuple:
    sampled = {name: sampled_value(rng, specification)
               for name, specification in search_space.items()}
    params = dict(base)
    params.update(sampled)
    return params, sampled


def validation_mae(frame: pd.DataFrame, features: pd.DataFrame, target: str,
                   origins: np.ndarray, horizon: int, boosters: list) -> float:
    actual, predicted = [], []
    for origin in origins:
        prediction = predict_hourly_models(
            boosters, features.iloc[[origin]].to_numpy(dtype=float))
        predicted.extend(prediction.tolist())
        actual.extend(frame[target].iloc[origin + 1:origin + horizon + 1].tolist())
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if len(actual_array) != len(origins) * horizon or not np.isfinite(actual_array).all():
        raise ValueError(f"{target}: invalid validation target block")
    return float(np.mean(np.abs(predicted_array - actual_array)))


def tune_target(frame: pd.DataFrame, features: pd.DataFrame, splits: dict,
                config: dict, target: str) -> dict:
    protocol, tuning = config["protocol"], config["tuning"]
    horizon = protocol["forecast_horizon_hours"]
    Xtrain, Ytrain, training = training_matrices(frame, features, splits, horizon, target)
    origins, validation = validation_origins(frame, splits["validation"], horizon)
    rng = np.random.RandomState(tuning["tuning_seed"])
    best_mae, best_params, best_sampled, best_trial = np.inf, None, None, None
    trials = []
    for trial in range(1, tuning["n_trials"] + 1):
        params, sampled = sample_parameters(
            config["model"]["params"], tuning["search_space"], rng)
        boosters = fit_hourly_models(
            Xtrain, Ytrain, params, protocol["random_seed"],
            config["runtime"]["fit_workers"])
        score = validation_mae(frame, features, target, origins, horizon, boosters)
        trials.append({"trial": trial, "validation_mae": score,
                       "sampled_parameters": sampled})
        if score < best_mae:
            best_mae, best_params, best_sampled, best_trial = score, params, sampled, trial
        print(json.dumps({"target": target, "trial": trial,
                          "trials": tuning["n_trials"], "validation_mae": score,
                          "best_validation_mae": best_mae}), flush=True)
    if best_params is None:
        raise ValueError(f"{target}: tuning selected no parameters")
    return {
        "selected_trial": best_trial,
        "selected_validation_mae": best_mae,
        "selected_parameters": best_params,
        "selected_sampled_parameters": best_sampled,
        "trials": trials,
        "training": training,
        "validation": validation,
    }


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


def mase_rmsse_judgement(comparison: dict) -> dict:
    result = {
        target: ("better_than_seasonal_naive" if all(
            comparison[target]["local"][metric] < 1.0 for metric in ("mase", "rmsse"))
                 else "not_better_than_seasonal_naive")
        for target in ("Down", "Up")
    }
    result["overall"] = "mixed_result_no_pass_fail_threshold"
    return result


def tuning_summary(config: dict, tuning_results: dict) -> dict:
    return {
        "tuner": config["tuning"]["tuner"],
        "optuna_used": config["tuning"]["optuna_used"],
        "optuna_reason": config["tuning"]["optuna_reason"],
        "objective": config["tuning"]["objective"],
        "n_trials": config["tuning"]["n_trials"],
        "tuning_seed": config["tuning"]["tuning_seed"],
        "search_space": config["tuning"]["search_space"],
        "targets": tuning_results,
    }


def hyperparameter_cost(variants: dict) -> dict:
    return {
        "definition": "inherited_mae_minus_tuned_mae_by_target",
        **{target: (variants["inherited"]["metrics"][target]["local"]["mae"]
                    - variants["tuned"]["metrics"][target]["local"]["mae"])
           for target in ("Down", "Up")},
    }


def anchor_judgement(variants: dict) -> dict:
    improvements = {
        target: variants["tuned"]["degenerate_fit_audit"][target][
            "improvement_over_constant_percent"]
        for target in ("Down", "Up")
    }
    threshold = 10.0
    degenerate_targets = [target for target, value in improvements.items()
                          if value < threshold]
    if degenerate_targets:
        return {
            "status": "tuned_still_degenerate_f08_implementation_not_validated",
            "statement_zh": "调参后仍退化，F08 实现能力未获验证",
            "minimum_improvement_percent": threshold,
            "degenerate_targets": degenerate_targets,
        }
    return {
        "status": "implementation_anchor_established",
        "statement_zh": "调参后两个目标均脱离常数退化，F08 锚定成立",
        "minimum_improvement_percent": threshold,
        "degenerate_targets": [],
    }


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
                  training: dict, comparison: dict, metric_audit: dict,
                  variants: dict, tuning_results: dict, forecasts: pd.DataFrame) -> dict:
    judgement = mase_rmsse_judgement(comparison)
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
            "hyperparameters": config["tuning"]["free_parameter_provenance"],
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
        "validation_usage": "tuned_variant_selected_by_validation_mae",
        "metric_protocol": {
            "reported": ["mae", "nmae", "rmse", "mase", "rmsse"],
            "seasonal_period_hours": config["protocol"]["seasonal_period_hours"],
            "scaled_metric_denominator": "continuous_test_target_seasonal_differences",
        },
        "training_data_updates_during_test": False,
        "shared_f08_fit_calls_per_target": config["protocol"]["forecast_horizon_hours"],
        "fit_workers": config["runtime"]["fit_workers"],
        "predict_calls_per_target": int(len(splits["test"]) /
                                        config["protocol"]["forecast_horizon_hours"]),
        "outlier_removal": config["protocol"]["outlier_removal"],
        "model": config["model"],
        "metrics": comparison,
        "paper_metric_consistency_audit": metric_audit,
        "mase_rmsse_judgement": judgement,
        "degenerate_fit_audit": variants["inherited"]["degenerate_fit_audit"],
        "variants": variants,
        "tuning": tuning_summary(config, tuning_results),
        "hyperparameter_cost_eur_free": hyperparameter_cost(variants),
        "anchor_judgement": anchor_judgement(variants),
        "reference_headroom": reference_headroom(forecasts, splits, config),
    }


def degenerate_fit_audit(forecasts: pd.DataFrame, splits: dict,
                         interpretation: str) -> dict:
    """A near-constant model can match a volatile target's MAE. Measure that, do not assume it away."""
    result = {}
    for target in ("Down", "Up"):
        rows = forecasts[forecasts["target"] == target].sort_values("timestamp_utc")
        actual, predicted = rows["actual"].to_numpy(), rows["forecast"].to_numpy()
        constant = float(splits["train"][target].median())
        constant_mae = float(np.abs(actual - constant).mean())
        model_mae = float(np.abs(actual - predicted).mean())
        result[target] = {
            "train_median_constant": constant,
            "constant_predictor_mae": constant_mae,
            "model_mae": model_mae,
            "improvement_over_constant_percent": (constant_mae - model_mae) / constant_mae * 100,
            "forecast_std_over_actual_std": float(predicted.std() / actual.std()),
            "forecast_mean_minus_actual_mean": float(predicted.mean() - actual.mean()),
        }
    result["interpretation"] = interpretation
    return result


def reference_headroom(forecasts: pd.DataFrame, splits: dict, config: dict) -> dict:
    """How far the paper's own model separates from the same constant we are measured against."""
    result = {}
    for target in ("Down", "Up"):
        rows = forecasts[forecasts["target"] == target]
        actual = rows["actual"].to_numpy()
        constant = float(splits["train"][target].median())
        constant_mae = float(np.abs(actual - constant).mean())
        paper_mae = float(config["paper_reported"][target]["mae"])
        result[target] = {
            "constant_predictor_mae": constant_mae,
            "paper_reported_mae": paper_mae,
            "paper_improvement_over_constant_percent":
                (constant_mae - paper_mae) / constant_mae * 100,
        }
    result["interpretation"] = (
        "The constant baseline applies to the reference as well. Where the paper's own model barely "
        "separates from it, that target carries little predictable signal at this horizon and cannot "
        "anchor an implementation for anyone; the anchoring verdict rests on the target where the "
        "reference does separate."
    )
    return result


def variant_summary(forecasts: pd.DataFrame, splits: dict, config: dict,
                    params: dict, source: str, validation_mae_by_target: dict,
                    interpretation: str) -> dict:
    comparison = metric_comparison(forecasts, config)
    return {
        "hyperparameter_source": source,
        "selected_parameters": params,
        "validation_mae": validation_mae_by_target,
        "metrics": comparison,
        "mase_rmsse_judgement": mase_rmsse_judgement(comparison),
        "degenerate_fit_audit": degenerate_fit_audit(
            forecasts, splits, interpretation),
        "fit_calls_per_target": config["protocol"]["forecast_horizon_hours"],
        "fit_workers": config["runtime"]["fit_workers"],
        "predict_calls_per_target": int(
            len(splits["test"]) / config["protocol"]["forecast_horizon_hours"]),
    }


def run_test_variants(frame: pd.DataFrame, features: pd.DataFrame, splits: dict,
                      config: dict, tuning_results: dict) -> tuple:
    parameter_sets = {
        "inherited": {target: config["model"]["params"] for target in ("Down", "Up")},
        "tuned": {target: tuning_results[target]["selected_parameters"]
                  for target in ("Down", "Up")},
    }
    frames, training, forecast_sets = [], {}, {}
    for variant, params_by_target in parameter_sets.items():
        variant_frames = []
        for target in ("Down", "Up"):
            forecast, training[target] = forecast_target(
                frame, features, splits, config, target, params_by_target[target], variant)
            variant_frames.append(forecast)
        forecast_sets[variant] = pd.concat(variant_frames, ignore_index=True)
        frames.extend(variant_frames)
    return pd.concat(frames, ignore_index=True), training, forecast_sets, parameter_sets


def build_variant_summaries(forecast_sets: dict, parameter_sets: dict, splits: dict,
                            config: dict, tuning_results: dict) -> dict:
    inherited_interpretation = (
        "The inherited model barely separates from the training-median constant, so matching the "
        "paper's MAE is not evidence that this implementation reproduces the paper's model. "
        "Hyperparameters were inherited untuned; the paper tuned them on its validation split."
    )
    tuned_interpretation = (
        "The tuned model must improve more than 10 percent over the training-median constant before "
        "it can anchor implementation capability; metric proximity alone is insufficient."
    )
    return {
        "inherited": variant_summary(
            forecast_sets["inherited"], splits, config, parameter_sets["inherited"],
            "configs/f08_lightgbm_dk1.yaml values without tuning",
            {target: None for target in ("Down", "Up")}, inherited_interpretation),
        "tuned": variant_summary(
            forecast_sets["tuned"], splits, config, parameter_sets["tuned"],
            "seeded random search minimizing validation MAE",
            {target: tuning_results[target]["selected_validation_mae"]
             for target in ("Down", "Up")}, tuned_interpretation),
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
    validation_end = splits["validation"].index[-1]
    tuning_frame = frame.loc[:validation_end]
    tuning_features = build_features(tuning_frame, config["features"])
    tuning_splits = {"train": splits["train"], "validation": splits["validation"]}
    tuning_results = {
        target: tune_target(tuning_frame, tuning_features, tuning_splits, config, target)
        for target in ("Down", "Up")
    }
    features = build_features(frame, config["features"])
    forecasts, training, forecast_sets, parameter_sets = run_test_variants(
        frame, features, splits, config, tuning_results)
    comparison = metric_comparison(forecast_sets["inherited"], config)
    metric_audit = paper_metric_audit(
        forecast_sets["inherited"], comparison, config["protocol"]["seasonal_period_hours"])
    variants = build_variant_summaries(
        forecast_sets, parameter_sets, splits, config, tuning_results)
    summary = build_summary(config, frame, splits, features, training, comparison,
                            metric_audit, variants, tuning_results, forecasts)
    output = write_outputs(summary, forecasts, config)
    print(json.dumps({"variants": {
        name: {"metrics": result["metrics"],
               "degenerate_fit_audit": result["degenerate_fit_audit"]}
        for name, result in variants.items()}},
                     indent=2, sort_keys=True), flush=True)
    print(f"R2 results written to {output}", flush=True)


if __name__ == "__main__":
    main()
