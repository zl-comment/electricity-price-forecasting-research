#!/usr/bin/env python3
"""Run the S2 storage baselines: B0, B1 and Oracle over one feasible region on DK1."""

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

from epf_harness.dk1_audit import load_table
from epf_harness.dk1_coupling import add_delivery_day, split_boundaries, usable_delivery_days
from epf_harness.dk1_storage import (
    check_boundary_feasibility,
    exclusivity_violations,
    hand_check,
    settle_day,
    solve_day,
    threshold_rule,
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


def build_panels(root: Path, config: dict) -> dict:
    """One dict of numpy arrays per delivery day. Hour index is position based, verified by S0."""
    area, timezone = config["data"]["primary_price_area"], config["protocol"]["civil_timezone"]
    prices = load_table(root, config["datasets"]["day_ahead_prices"])
    prices = add_delivery_day(prices[prices["PriceArea"] == area], "TimeUTC", timezone)
    capacity = add_delivery_day(load_table(root, config["datasets"]["afrr_capacity_market"]), "TimeUTC", timezone)
    imbalance = add_delivery_day(
        load_table(root, config["datasets"]["imbalance_and_activation"]), "TimeUTC", timezone
    )
    panels = {}
    for day, quarter in prices.groupby("delivery_day"):
        hourly = capacity[capacity["delivery_day"] == day]
        activation = imbalance[imbalance["delivery_day"] == day]
        n_slots, n_hours = len(quarter), len(hourly)
        if n_slots != 4 * n_hours or len(activation) != n_slots:
            raise ValueError(f"{day}: {n_slots} quarters, {n_hours} hours, {len(activation)} activation rows")
        procured = hourly["UpProcuredMW"].to_numpy()
        panels[day] = {
            "day_ahead_price": quarter["DayAheadPriceEUR"].to_numpy(),
            "capacity_price": hourly["UpPriceEUR"].to_numpy(),
            "procured_mw": procured,
            "hour_of_slot": np.arange(n_slots) // 4,
            "activation_share": activation["aFRRUpMW"].to_numpy() / np.repeat(procured, 4),
            "activation_price": activation["aFRRVWAUpEUR"].to_numpy(),
            "shortfall_price": activation[config["settlement"]["shortfall_price_column"]].to_numpy(),
        }
    return panels


def resize(series: np.ndarray, length: int) -> np.ndarray:
    """Explicit length policy for the two DST days: truncate, or repeat the last value."""
    if len(series) == length:
        return series
    if len(series) > length:
        return series[:length]
    return np.concatenate([series, np.repeat(series[-1], length - len(series))])


def persisted_inputs(panels: dict, day: str, previous: str, config: dict) -> dict:
    """What the planner may believe at the gates: yesterday's realised curves, nothing from D."""
    target = panels[day]
    source = panels[previous]
    return {
        "day_ahead_price": resize(source["day_ahead_price"], len(target["day_ahead_price"])),
        "capacity_price": resize(source["capacity_price"], len(target["capacity_price"])),
        "adjusted": bool(len(source["day_ahead_price"]) != len(target["day_ahead_price"])),
    }


def plan_day(name: str, strategy: dict, panel: dict, believed: dict, threshold: float,
             arm: dict, storage: dict, solver: dict) -> dict:
    if name == "Oracle":
        return solve_day(panel["day_ahead_price"], panel["capacity_price"], panel["procured_mw"],
                         panel["hour_of_slot"], arm, storage, solver, None)
    if strategy["commits_reserve"]:
        fixed = threshold_rule(believed["capacity_price"], threshold, arm, panel["procured_mw"])
    else:
        fixed = np.zeros(len(panel["capacity_price"]))
    return solve_day(believed["day_ahead_price"], believed["capacity_price"], panel["procured_mw"],
                     panel["hour_of_slot"], arm, storage, solver, fixed)


def reserve_threshold(panels: dict, lookback: list, quantile: float) -> float:
    prices = np.concatenate([panels[day]["capacity_price"] for day in lookback])
    return float(np.quantile(prices, quantile))


def evaluate(panels: dict, days: list, config: dict, arm: dict, storage: dict, threshold: float) -> tuple:
    rows, violations, adjusted = [], {name: 0 for name in config["strategies"]}, 0
    for day, previous in days:
        believed = persisted_inputs(panels, day, previous, config)
        adjusted += int(believed["adjusted"])
        for name, strategy in config["strategies"].items():
            plan = plan_day(name, strategy, panels[day], believed, threshold, arm, storage, config["solver"])
            violations[name] += exclusivity_violations(plan, config["solver"]["exclusivity_tolerance_mw"])
            settled = settle_day(plan, panels[day], arm, storage)
            rows.append({"delivery_day": day, "strategy": name,
                         "planned_profit_eur": plan["planned_profit_eur"], **settled})
    return pd.DataFrame(rows), violations, adjusted


def describe(series: pd.Series) -> dict:
    return {
        "n": int(series.count()), "mean": float(series.mean()), "std": float(series.std(ddof=0)),
        "min": float(series.min()), "p10": float(series.quantile(0.10)),
        "p50": float(series.quantile(0.50)), "p90": float(series.quantile(0.90)),
        "max": float(series.max()), "total": float(series.sum()),
    }


def strategy_summary(frame: pd.DataFrame, violations: int) -> dict:
    columns = ["total_eur", "energy_eur", "capacity_eur", "activation_eur", "shortfall_cost_eur",
               "schedule_gap_cost_eur", "reserve_energy_gap_mwh", "reserve_power_gap_mwh",
               "day_ahead_gap_mwh", "committed_mw_hours", "cycles",
               "terminal_soc_deviation_mwh", "min_soc_mwh"]
    return {
        "days": int(len(frame)),
        "exclusivity_violations": int(violations),
        "days_with_shortfall": int((frame["slots_with_shortfall"] > 0).sum()),
        "days_with_schedule_gap": int((frame["slots_with_schedule_gap"] > 0).sum()),
        "profit_cvar_5pct_eur": float(frame["total_eur"].nsmallest(max(1, len(frame) // 20)).mean()),
        **{column: describe(frame[column]) for column in columns},
    }


def check_findings(summaries: dict, config: dict) -> list:
    findings = []
    for name, summary in summaries.items():
        if summary["exclusivity_violations"]:
            findings.append((name, "simultaneous_charge_and_discharge", str(summary["exclusivity_violations"])))
        deviation = abs(summary["terminal_soc_deviation_mwh"]["max"])
        if deviation > config["solver"]["soc_tolerance_mwh"] and name == "B0":
            findings.append((name, "terminal_soc_not_restored", f"{deviation:.6f}"))
        if summary["min_soc_mwh"]["min"] < -config["solver"]["soc_tolerance_mwh"]:
            findings.append((name, "negative_soc", f"{summary['min_soc_mwh']['min']:.6f}"))
    if summaries["Oracle"]["total_eur"]["total"] < summaries["B1"]["total_eur"]["total"]:
        findings.append(("Oracle", "oracle_below_deployable_baseline", "total_eur"))
    return findings


def write_outputs(summary: dict, daily: pd.DataFrame, findings: list, output: dict) -> Path:
    directory = REPOSITORY_ROOT / output["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / output["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    daily.to_csv(directory / output["daily_csv"], index=False)
    with (directory / output["findings_csv"]).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["strategy", "finding", "value"])
        writer.writerows(findings)
    return directory


def main() -> None:
    arguments = parse_arguments()
    config = yaml.safe_load(arguments.config.open(encoding="utf-8"))
    np.random.seed(config["protocol"]["random_seed"])
    storage = dict(config["storage"], slot_hours=config["protocol"]["slot_hours"],
                   soc_tolerance_mwh=config["solver"]["soc_tolerance_mwh"])
    arm = config["storage"]["arms"]["main"]
    reserved = check_boundary_feasibility(arm, storage)
    checked = hand_check(config)
    print(f"Hand check passed: {checked['profit_eur']:.6f} EUR; "
          f"full-power commitment reserves {reserved:.4f} MWh", flush=True)
    usable = usable_delivery_days(config["window"])
    split = split_boundaries(config, usable)
    panels = build_panels(REPOSITORY_ROOT / config["data"]["root"], config)
    available = set(usable) & set(panels)
    lag = config["persistence"]["day_ahead_lag_days"]
    pairs = [(d, (pd.Timestamp(d) - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")) for d in sorted(available)]
    test = [(d, p) for d, p in pairs if d >= split["first_test_delivery_day"] and p in available]
    print(f"Evaluating {len(test)} of {split['test_delivery_days']} frozen test days", flush=True)
    threshold = reserve_threshold(panels, [d for d in sorted(available) if d < split["first_test_delivery_day"]],
                                  config["reserve_rule"]["threshold_quantile"])
    daily, violations, adjusted = evaluate(panels, test, config, arm, storage, threshold)
    summaries = {name: strategy_summary(daily[daily["strategy"] == name], violations[name])
                 for name in config["strategies"]}
    findings = check_findings(summaries, config)
    summary = {
        "random_seed": config["protocol"]["random_seed"],
        "environment_fingerprint": environment_fingerprint(),
        "solver": config["solver"]["method"],
        "hand_check": checked,
        "full_power_reservation_mwh": reserved,
        "split": split,
        "evaluated_test_days": len(test),
        "days_dropped_for_missing_persistence": split["test_delivery_days"] - len(test),
        "days_with_dst_length_adjustment": adjusted,
        "reserve_threshold_eur_per_mw": threshold,
        "arm": arm,
        "strategies": summaries,
        "findings": [{"strategy": s, "finding": f, "value": v} for s, f, v in findings],
    }
    directory = write_outputs(summary, daily, findings, config["output"])
    for name, item in summaries.items():
        print(f"{name:7s} total {item['total_eur']['total']:10.0f} EUR  "
              f"energy {item['energy_eur']['total']:9.0f}  capacity {item['capacity_eur']['total']:8.0f}  "
              f"activation {item['activation_eur']['total']:9.0f}  "
              f"shortfall {item['shortfall_cost_eur']['total']:8.0f}  "
              f"gap {item['reserve_energy_gap_mwh']['total']:7.2f} MWh", flush=True)
    print(f"{len(findings)} findings written to {directory}", flush=True)


if __name__ == "__main__":
    main()
