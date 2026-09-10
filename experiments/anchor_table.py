#!/usr/bin/env python3
"""Build the external-baseline anchor table from committed result summaries."""

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_inputs(config: dict) -> dict:
    return {
        name: json.loads((REPOSITORY_ROOT / path).read_text(encoding="utf-8"))
        for name, path in config["inputs"].items()
    }


def source(file_key: str, field: str, config: dict) -> dict:
    return {"file": config["inputs"][file_key], "field": field}


def dk1_values(method: str, inputs: dict) -> dict:
    if method == "F01_LEAR":
        summary = inputs["f01_summary"]
        threshold = summary["settlement"]["total_eur"]["total"]
        free = inputs["s2c_summary"]["arms"]["B2_F01"]["total_eur"]
    else:
        summary = inputs["f08_summary"]
        threshold = summary["settlement"]["total_eur"]
        free = inputs["s2c_summary"]["arms"]["B2_F08"]["total_eur"]
    metrics = summary["prediction_metrics_211_common_days"]
    return {
        "dk1_day_ahead_mae": metrics["day_ahead_price"]["model"]["mae"],
        "dk1_capacity_mae": metrics["capacity_price"]["model"]["mae"],
        "dk1_threshold_arm_eur": threshold,
        "dk1_free_arm_eur": free,
    }


def dk1_sources(method: str, config: dict) -> dict:
    summary = "f01_summary" if method == "F01_LEAR" else "f08_summary"
    arm = "B2_F01" if method == "F01_LEAR" else "B2_F08"
    total_field = "settlement.total_eur.total" if method == "F01_LEAR" else "settlement.total_eur"
    return {
        "dk1_day_ahead_mae": source(
            summary, "prediction_metrics_211_common_days.day_ahead_price.model.mae", config),
        "dk1_capacity_mae": source(
            summary, "prediction_metrics_211_common_days.capacity_price.model.mae", config),
        "dk1_threshold_arm_eur": source(summary, total_field, config),
        "dk1_free_arm_eur": source("s2c_summary", f"arms.{arm}.total_eur", config),
    }


def lear_rows(config: dict, inputs: dict) -> tuple:
    values = dk1_values("F01_LEAR", inputs)
    shared_sources = dk1_sources("F01_LEAR", config)
    rows, provenance = [], []
    for window in config["lear_windows"]:
        key = str(window)
        result = inputs["r1_summary"]["methods"][key]
        method = "F01_LEAR_ensemble" if key == "Ensemble" else f"F01_LEAR_cw{key}"
        verification = ("point_level_reproduction_passed" if result["accepted"]
                        else "point_level_reproduction_failed")
        row = {
            "method": method,
            "original_task": "EPF-DE day-ahead price forecasting",
            "original_target_product": "day-ahead electricity price",
            "paper_reported": result["target_mae"],
            "local_on_original_task": result["local"]["mae"],
            "deviation_percent": result["mae_bias_percent"],
            "verification_type": verification,
            **values,
            "cross_market_mae_comparison": "prohibited",
            "note": result.get("not_reproducible_reason", ""),
        }
        r1 = config["inputs"]["r1_summary"]
        fields = {name: {"file": r1, "field": f"methods.{key}.{field}"} for name, field in {
            "paper_reported": "target_mae", "local_on_original_task": "local.mae",
            "deviation_percent": "mae_bias_percent"}.items()}
        rows.append(row)
        provenance.append({"method": row["method"], "numeric_sources": {**fields, **shared_sources}})
    return rows, provenance


def f08_row(config: dict, inputs: dict) -> tuple:
    r2 = inputs["r2_summary"]
    metrics = r2["variants"]["tuned"]["metrics"]
    paper_mase = "; ".join(f"{target} MASE={metrics[target]['paper_reported']['mase']}"
                           for target in ("Down", "Up"))
    local_mase = "; ".join(f"{target} MASE={metrics[target]['local']['mase']}"
                           for target in ("Down", "Up"))
    deviation = "; ".join(
        f"{target} MASE={metrics[target]['absolute_deviation_percent']['mase']}"
        for target in ("Down", "Up"))
    feature = r2["free_parameters"]["feature_selection"]
    hyperparameters = r2["free_parameters"]["hyperparameters"]
    anchor_status = r2["anchor_judgement"]["status"]
    row = {
        "method": "F08_LightGBM", "original_task": "Finnish aFRR energy price forecasting",
        "original_target_product": "aFRR energy (activation) price",
        "paper_reported": paper_mase, "local_on_original_task": local_mase,
        "deviation_percent": deviation,
        "verification_type": "partial_reproduction_free_hyperparameters",
        **dk1_values("F08_LightGBM", inputs),
        "cross_market_mae_comparison": "prohibited",
        "note": (f"partial reproduction with feature_selection={feature} and "
                 f"hyperparameters={hyperparameters}; hyperparameters were selected on validation "
                 "data, while the search space, budget, and seed were repository-defined; "
                 f"anchor_status={anchor_status}; "
                 "Finnish target is aFRR energy price, "
                 "whereas the DK1 target is aFRR capacity price"),
    }
    provenance = {
        "method": row["method"],
        "numeric_sources": {
            **dk1_sources("F08_LightGBM", config),
            "paper_reported": [source(
                "r2_summary", f"variants.tuned.metrics.{target}.paper_reported.mase", config)
                for target in ("Down", "Up")],
            "local_on_original_task": [source(
                "r2_summary", f"variants.tuned.metrics.{target}.local.mase", config)
                for target in ("Down", "Up")],
            "deviation_percent": [source(
                "r2_summary", f"variants.tuned.metrics.{target}.absolute_deviation_percent.mase", config)
                for target in ("Down", "Up")],
        },
        "note_sources": [source("r2_summary", "free_parameters.feature_selection", config),
                         source("r2_summary", "free_parameters.hyperparameters", config),
                         source("r2_summary", "tuning.search_space", config),
                         source("r2_summary", "tuning.n_trials", config),
                         source("r2_summary", "tuning.tuning_seed", config),
                         source("r2_summary", "anchor_judgement.status", config)],
    }
    return row, provenance


def empty_method_row() -> dict:
    columns = ["original_task", "original_target_product", "paper_reported",
               "local_on_original_task", "deviation_percent", "dk1_day_ahead_mae",
               "dk1_capacity_mae", "dk1_threshold_arm_eur", "dk1_free_arm_eur",
               "cross_market_mae_comparison", "note"]
    return {"method": "FB2_B3", **{column: None for column in columns},
            "verification_type": "no_prior_work"}


def validate(table: pd.DataFrame, config: dict, inputs: dict) -> dict:
    allowed = set(config["allowed_verification_types"])
    counts = Counter(table["verification_type"])
    if len(table) != 7 or not set(counts) <= allowed:
        raise ValueError(f"Invalid anchor table shape or verification type: {counts}")
    if counts["point_level_reproduction_passed"] != 2:
        raise ValueError(f"Expected 2 passed LEAR rows: {counts}")
    if counts["point_level_reproduction_failed"] != 3:
        raise ValueError(f"Expected 3 failed LEAR rows: {counts}")
    if counts["partial_reproduction_free_hyperparameters"] != 1:
        raise ValueError(f"Expected 1 partial F08 reproduction row: {counts}")
    if counts["no_prior_work"] != 1 or counts["no_anchor"] != 0:
        raise ValueError(f"Unexpected unanchored row counts: {counts}")
    for window in config["lear_windows"]:
        method = "F01_LEAR_ensemble" if window == "Ensemble" else f"F01_LEAR_cw{window}"
        row = table.loc[table["method"] == method].iloc[0]
        result = inputs["r1_summary"]["methods"][str(window)]
        if row["local_on_original_task"] != result["local"]["mae"]:
            raise ValueError(f"Local MAE differs for {window}")
        if row["deviation_percent"] != result["mae_bias_percent"]:
            raise ValueError(f"MAE deviation differs for {window}")
    f08 = table.loc[table["method"] == "F08_LightGBM"].iloc[0]
    if (f08["verification_type"] != "partial_reproduction_free_hyperparameters"
            or pd.isna(f08["paper_reported"])):
        raise ValueError("F08 must contain the completed R2 partial reproduction")
    return dict(sorted(counts.items()))


def main() -> None:
    config = yaml.safe_load(parse_arguments().config.read_text(encoding="utf-8"))
    np.random.seed(config["random_seed"])
    inputs = load_inputs(config)
    rows, provenance = lear_rows(config, inputs)
    f08, f08_provenance = f08_row(config, inputs)
    rows.extend([f08, empty_method_row()])
    provenance.extend([f08_provenance, {"method": "FB2_B3", "numeric_sources": {}}])
    table = pd.DataFrame(rows)
    counts = validate(table, config, inputs)
    output = REPOSITORY_ROOT / config["output"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / config["output"]["table_csv"], index=False)
    summary = {
        "environment_fingerprint": {"python": platform.python_version(),
                                    "numpy": np.__version__, "pandas": pd.__version__},
        "random_seed": config["random_seed"], "rows": len(table),
        "verification_type_counts": counts,
        "metric_boundary": "Original-task and DK1 MAE columns must not be compared across markets.",
        "row_provenance": provenance,
    }
    with (output / config["output"]["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"rows": len(table), "verification_type_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
