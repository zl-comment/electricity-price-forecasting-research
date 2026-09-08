#!/usr/bin/env python3
"""Run the S1 coupling audit: reserve-gate crowd-out and activation opportunity cost on DK1."""

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from epf_harness.dk1_coupling import (
    activation_table,
    capacity_envelope,
    coupling_findings,
    crowd_out_table,
    load_coupled_frames,
    restrict_to_days,
    split_boundaries,
    stress_thresholds,
    usable_delivery_days,
)


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


def describe(series: pd.Series) -> dict:
    return {
        "n": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "p50": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
    }


def arm_summary(crowd_out: pd.DataFrame, activation: pd.DataFrame, arm: dict) -> dict:
    reserve_days = int(crowd_out["reserve_dominates"].sum())
    return {
        "power_mw": arm["power_mw"],
        "energy_mwh": arm["energy_mwh"],
        "capacity_fraction": arm["capacity_fraction"],
        "market_share_cap": arm["market_share_cap"],
        "days": int(len(crowd_out)),
        "days_reserve_dominates": reserve_days,
        "days_energy_dominates": int(len(crowd_out)) - reserve_days,
        "reserve_full_day_eur": describe(crowd_out["reserve_full_day_eur"]),
        "energy_one_cycle_eur": describe(crowd_out["energy_one_cycle_eur"]),
        "crowd_out_eur": describe(crowd_out["crowd_out_eur"]),
        "activated_mwh_per_mw_per_day": describe(activation["activated_mwh_per_mw"]),
        "net_activation_eur_per_mw_per_day": describe(activation["net_activation_eur_per_mw"]),
        "soc_throughput_share_of_energy": describe(activation["soc_throughput_share_of_energy"]),
    }


def write_outputs(summary: dict, findings: list, tables: dict, output: dict) -> Path:
    directory = REPOSITORY_ROOT / output["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / output["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    tables["crowd_out"].to_csv(directory / output["crowd_out_csv"], index=False)
    tables["activation"].to_csv(directory / output["activation_csv"], index=False)
    with (directory / "findings.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "check", "finding", "value"])
        writer.writerows(findings)
    return directory


def run_arms(config: dict, frames: dict) -> tuple:
    storage = config["storage"]
    summaries, findings, crowd_frames, activation_frames = {}, [], [], []
    for name, arm in storage["arms"].items():
        print(f"Arm {name}: {arm['power_mw']} MW / {arm['energy_mwh']} MWh", flush=True)
        crowd_out = crowd_out_table(frames["prices"], frames["capacity"], arm, storage)
        activation = activation_table(frames["activation"], arm)
        merged = crowd_out.merge(activation, on="delivery_day", how="inner")
        if len(merged) != len(crowd_out):
            raise ValueError(f"Arm {name}: crowd-out and activation cover different delivery days")
        summaries[name] = arm_summary(crowd_out, activation, arm)
        summaries[name]["capacity_envelope"] = capacity_envelope(arm, storage)
        findings += [(name,) + row for row in coupling_findings(crowd_out, config["acceptance"])]
        crowd_frames.append(crowd_out.assign(arm=name))
        activation_frames.append(activation.assign(arm=name))
    tables = {
        "crowd_out": pd.concat(crowd_frames, ignore_index=True),
        "activation": pd.concat(activation_frames, ignore_index=True),
    }
    return summaries, findings, tables


def main() -> None:
    arguments = parse_arguments()
    config = yaml.safe_load(arguments.config.open(encoding="utf-8"))
    np.random.seed(config["protocol"]["random_seed"])
    root = REPOSITORY_ROOT / config["data"]["root"]
    days = usable_delivery_days(config["window"])
    split = split_boundaries(config, days)
    frames = restrict_to_days(load_coupled_frames(root, config), days)
    lookback = {d for d in days if d < split["first_test_delivery_day"]}
    summaries, findings, tables = run_arms(config, frames)
    summary = {
        "random_seed": config["protocol"]["random_seed"],
        "environment_fingerprint": environment_fingerprint(),
        "split": split,
        "reference_pv_plant_mwp": config["storage"]["reference_pv_plant_mwp"],
        "round_trip_efficiency": config["storage"]["round_trip_efficiency"],
        "reserve_delivery_hours": config["storage"]["reserve_delivery_hours"],
        "arms": summaries,
        "stress_thresholds": stress_thresholds(
            frames["capacity"], frames["activation"], frames["net_load"], lookback, config["stress_scenarios"]
        ),
        "findings": [{"arm": a, "check": c, "finding": f, "value": v} for a, c, f, v in findings],
    }
    directory = write_outputs(summary, findings, tables, config["output"])
    print(json.dumps(summary["split"], indent=2, ensure_ascii=False), flush=True)
    print(f"{len(findings)} findings written to {directory}", flush=True)


if __name__ == "__main__":
    main()
