#!/usr/bin/env python3
"""Run the paper-era LEAR acceptance test on EPF-DE."""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from epf_harness.backtest import load_paper_lear, run_window
from epf_harness.data import load_epf_data
from epf_harness.metrics import (
    absolute_bias_percent,
    dm_p_value,
    forecast_metrics,
    point_difference_statistics,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def add_comparison_columns(
    comparison: pd.DataFrame, name: str, local: pd.Series, author: pd.Series
) -> None:
    prefix = f"lear_{name.lower()}"
    comparison[f"{prefix}_local"] = local
    comparison[f"{prefix}_author"] = author
    comparison[f"{prefix}_difference"] = local - author
    comparison[f"{prefix}_absolute_difference"] = (local - author).abs()


def summarize_method(
    name: str, local: pd.Series, author: pd.Series, actual: pd.Series, config: dict
) -> dict:
    metrics_config = config["metrics"]
    protocol = config["protocol"]
    local_metrics = forecast_metrics(actual, local, metrics_config["rmae_seasonality"])
    author_metrics = forecast_metrics(actual, author, metrics_config["rmae_seasonality"])
    target_mae = config["reference"]["target_mae"][name]
    if abs(author_metrics["mae"] - target_mae) >= config["acceptance"]["author_target_rounding_tolerance"]:
        raise RuntimeError(f"Author MAE does not match target for {name}")
    point_difference = point_difference_statistics(local, author)
    summary = {
        "local": local_metrics,
        "author": author_metrics,
        "target_mae": target_mae,
        "mae_bias_percent": absolute_bias_percent(local_metrics["mae"], author_metrics["mae"]),
        "median_point_absolute_difference": point_difference["median"],
        "percentile_99_point_absolute_difference": point_difference["percentile_99"],
        "maximum_point_absolute_difference": point_difference["maximum"],
        "maximum_difference_timestamp": point_difference["maximum_timestamp"],
        "dm_p_value_local_vs_author": dm_p_value(
            actual,
            local,
            author,
            protocol["hours_per_day"],
            metrics_config["dm_norm"],
            metrics_config["dm_version"],
        ),
    }
    acceptance = config["acceptance"]
    summary["accepted"] = (
        summary["median_point_absolute_difference"]
        < acceptance["median_point_absolute_difference_max"]
        and summary["percentile_99_point_absolute_difference"]
        < acceptance["percentile_99_point_absolute_difference_max"]
        and summary["mae_bias_percent"] < acceptance["mae_bias_percent_max"]
    )
    return summary


def write_outputs(comparison: pd.DataFrame, summary: dict, output: dict) -> None:
    directory = REPOSITORY_ROOT / output["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(directory / output["comparison_csv"], index_label="timestamp")
    with (directory / output["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def enforce_acceptance(name: str, result: dict, acceptance: dict) -> None:
    if name not in acceptance["hard_windows"] or result["accepted"]:
        return
    raise RuntimeError(
        f"LEAR {name} failed: median_abs_diff="
        f"{result['median_point_absolute_difference']:.6f}, "
        f"p99_abs_diff={result['percentile_99_point_absolute_difference']:.6f}, "
        f"MAE_bias={result['mae_bias_percent']:.6f}%, "
        f"max_abs_diff={result['maximum_point_absolute_difference']:.6f} at "
        f"{result['maximum_difference_timestamp']}"
    )


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    np.random.seed(config["protocol"]["random_seed"])
    data = load_epf_data(REPOSITORY_ROOT, config["data"])
    lear_class, runtime = load_paper_lear(config["runtime"])
    actual = data.test["Price"]
    comparison = pd.DataFrame({"real_price": actual})
    environment_fingerprint = runtime["environment_fingerprint"]
    comparison["environment_fingerprint"] = json.dumps(
        environment_fingerprint, sort_keys=True, separators=(",", ":")
    )
    summary = {
        "random_seed": config["protocol"]["random_seed"],
        "environment_fingerprint": environment_fingerprint,
        "runtime": runtime,
        "methods": {},
    }
    member_forecasts = []

    for window in config["protocol"]["calibration_windows"]:
        name = str(window)
        print(f"Running LEAR {name}", flush=True)
        local = run_window(lear_class, data, window, config["protocol"])
        author = data.author[config["reference"]["author_columns"][name]]
        member_forecasts.append(local)
        add_comparison_columns(comparison, name, local, author)
        summary["methods"][name] = summarize_method(name, local, author, actual, config)
        write_outputs(comparison, summary, config["output"])
        enforce_acceptance(name, summary["methods"][name], config["acceptance"])

    ensemble = pd.concat(member_forecasts, axis=1).mean(axis=1)
    author_ensemble = data.author[config["reference"]["author_columns"]["Ensemble"]]
    add_comparison_columns(comparison, "Ensemble", ensemble, author_ensemble)
    summary["methods"]["Ensemble"] = summarize_method(
        "Ensemble", ensemble, author_ensemble, actual, config
    )
    write_outputs(comparison, summary, config["output"])
    enforce_acceptance("Ensemble", summary["methods"]["Ensemble"], config["acceptance"])
    print(json.dumps(summary["methods"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
