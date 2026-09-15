#!/usr/bin/env python3
"""Run the frozen S3-B same-protocol forecast-and-decision comparison."""

import argparse
from hashlib import sha256
import json
import multiprocessing as mp
import platform
from pathlib import Path
import sys

import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml
from scipy import stats
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from epf_harness.dk1_storage import (activation_oracle_constraints, activation_selector,
                                     activation_soc_matrix, day_bounds, day_constraints,
                                     efficiencies, exclusivity_violations, hour_selector,
                                     settle_day, soc_matrix, solve_day, solve_recovery,
                                     threshold_rule)
from f01_lear_dk1.settlement import build_panels, resize, settlement_protocol
from s3_comparison import scenario_export
from s3_comparison.hand_checks import (check_degenerate_solution, check_nonanticipativity,
                                       check_risk_scoring, check_two_scenario_solution,
                                       degenerate_case, two_scenario_case)
from s3_comparison.published_rules import generate_x14_archive, write_x14_archive
from s3_comparison.published_rules import x06_deterministic_scenario
from s3_comparison.risk_scoring import (aggregate_calibration, calibration_row,
                                        cross_score_dm, empirical_tail_metrics,
                                        joint_var_cvar_score, paired_metrics)
from s3_comparison.scenario_io import (load_common_scenarios, load_premium_history,
                                       prepare_day_scenarios)
from s3_comparison.three_stage import (cluster_capacity, deployment_plan,
                                       solve_delivery_first, solve_stochastic)
from s3_forecast_side.panel import window_days


WORKER_STATE = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_configs(path: Path) -> tuple:
    study = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    forecast = yaml.safe_load((ROOT / study["base_config"]).read_text(encoding="utf-8"))
    storage = yaml.safe_load((ROOT / study["storage_config"]).read_text(encoding="utf-8"))
    _require(study["recourse"]["delivery_0730"] == "full", "07:30 recourse is not frozen")
    _require(study["recourse"]["delivery_1200"] == "delivery_first",
             "12:00 recourse is not frozen")
    return study, forecast, storage


def storage_settings(source: dict) -> dict:
    return dict(source["storage"], slot_hours=source["protocol"]["slot_hours"],
                soc_tolerance_mwh=source["solver"]["soc_tolerance_mwh"],
                recovery_constraint_tolerance_mwh=(
                    source["solver"]["recovery_constraint_tolerance_mwh"]))


def environment_fingerprint() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
            "lightgbm": lightgbm.__version__}


def _paper_checks(study: dict) -> dict:
    result = {}
    fields = set(study["published_rules"]["extraction_fields"])
    for name in ("x06", "x07", "x09", "x13", "x14"):
        rule = study["published_rules"][name]
        path = ROOT / rule["source"]
        _require(path.is_file(), f"{name}: source PDF is missing")
        digest = sha256(path.read_bytes()).hexdigest()
        _require(digest == rule["sha256"], f"{name}: source PDF hash differs")
        _require(set(rule["source_locator"]) == fields, f"{name}: extraction fields differ")
        result[name] = {"sha256": digest, "execution": rule["execution"],
                        "undetermined": rule["undetermined"]}
    return result


def _scenario_from_panel(panel: dict) -> dict:
    keys = ["day_ahead_price", "capacity_price", "activation_share",
            "activation_price", "shortfall_price"]
    return {key: np.asarray(panel[key], dtype=float)[None, :] for key in keys}


def _solve_case(case: dict, study: dict, chi: float, delivery_mode: str) -> dict:
    keys = ["day_ahead_price", "capacity_price", "activation_share",
            "activation_price", "shortfall_price"]
    scenarios = {key: case[key] for key in keys}
    return solve_stochastic(
        scenarios, case["scenario_probability"], case["cluster_labels"],
        case["procured_mw"], case["hour_of_slot"], case["arm"], case["storage"],
        study["solver"], study["risk"], chi, None, delivery_mode, "profit", None)


def _hand_checks(study: dict) -> dict:
    one = degenerate_case(study)
    one_solution = _solve_case(one, study, 0.0, study["recourse"]["delivery_0730"])
    _require(one_solution["success"], "Degenerate stochastic LP failed")
    first = check_degenerate_solution(one_solution["objective_eur"],
                                      one_solution["r_up"], study)
    two = two_scenario_case(study)
    chi0 = _solve_case(two, study, 0.0, study["recourse"]["delivery_0730"])
    chi1 = _solve_case(two, study, 1.0, study["recourse"]["delivery_0730"])
    _require(chi0["success"] and chi1["success"], "Two-scenario stochastic LP failed")
    second = check_two_scenario_solution(
        {"objective_eur": chi0["objective_eur"], "reserve_mw": chi0["r_up"][0]},
        {"objective_eur": chi1["objective_eur"], "reserve_mw": chi1["r_up"][0]}, study)
    scenario_charge = chi0["p_ch_cluster"][chi0["labels"]]
    scenario_discharge = chi0["p_dis_cluster"][chi0["labels"]]
    third = check_nonanticipativity(
        chi0["labels"], scenario_charge, scenario_discharge, chi0["r_up"],
        chi0["r_up"].copy(), study)
    tenth = check_risk_scoring(joint_var_cvar_score, empirical_tail_metrics, study)
    return {"degenerate_consistency": first, "two_scenario": second,
            "nonanticipativity": third, "risk_scoring": tenth}


def _perfect_day(panel: dict, study: dict, storage_source: dict,
                 arm: dict, storage: dict, delivery_mode: str) -> tuple:
    scenarios = _scenario_from_panel(panel)
    probabilities, labels = np.ones(1), np.zeros(1, dtype=int)
    early = solve_stochastic(
        scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
        arm, storage, study["solver"], study["risk"], 0.0, None, delivery_mode,
        "profit", None)
    _require(early["success"], f"Perfect-information 07:30 LP failed: {early}")
    if delivery_mode == study["recourse"]["shortfall_sensitivity"]:
        late = solve_stochastic(
            scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
            arm, storage, study["solver"], study["risk"], 0.0, early["r_up"],
            delivery_mode, "profit", None)
    else:
        late = solve_delivery_first(
            scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
            arm, storage, study["solver"], study["risk"], 0.0, early["r_up"],
            study["recourse"]["delivery_tolerance_mwh"])
    _require(late["success"], f"Perfect-information 12:00 LP failed: {late}")
    settled = settle_day(deployment_plan(late), panel, arm, storage, storage_source["solver"])
    return settled, late


def _oracle_with_base_constraints(panel: dict, arm: dict, storage: dict, solver: dict) -> dict:
    """Frozen activation-aware Oracle LP plus solve_day's no-activation SOC band and reservation."""
    prices, capacity = panel["day_ahead_price"], panel["capacity_price"]
    n_slots, n_hours = len(prices), len(capacity)
    delta = storage["slot_hours"]
    eta_c, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_0 = arm["energy_mwh"] * storage["initial_soc_share"]
    activation = activation_selector(panel["activation_share"], panel["hour_of_slot"], n_hours)
    hours = hour_selector(panel["hour_of_slot"], n_slots, n_hours)
    soc = activation_soc_matrix(n_slots, n_hours, activation, eta_c, eta_d, delta)
    oracle_ub, oracle_b = activation_oracle_constraints(
        soc, activation, hours[:, 2 * n_slots:], arm, storage, soc_0)
    base = soc_matrix(n_slots, eta_c, eta_d, delta, n_hours)
    day_ub, day_b = day_constraints(base, hours, arm, storage, soc_0)
    day_ub = np.hstack([day_ub, np.zeros((len(day_b), n_slots))])
    cost = np.concatenate([delta * prices, -delta * prices,
                           -capacity - activation.T @ panel["activation_price"],
                           delta * panel["shortfall_price"]])
    terminal = np.zeros((2, 3 * n_slots + n_hours))
    terminal[0, :2 * n_slots] = base[-1, :2 * n_slots]
    terminal[1] = soc[-1]
    scale = solver["objective_scale_eur"]
    bounds = day_bounds(arm, n_slots, panel["procured_mw"], None) + [(0.0, arm["power_mw"])] * n_slots
    result = linprog(cost / scale, A_ub=np.vstack([oracle_ub, day_ub]),
                     b_ub=np.concatenate([oracle_b, day_b]), A_eq=terminal, b_eq=np.zeros(2),
                     bounds=bounds, method=solver["method"])
    _require(result.success, f"Constrained Oracle LP did not solve: {result.message}")
    x = result.x
    return {"p_ch": x[:n_slots], "p_dis": x[n_slots:2 * n_slots],
            "r_up": x[2 * n_slots:2 * n_slots + n_hours], "p_recovery": x[2 * n_slots + n_hours:],
            "planned_profit_eur": float(-result.fun * scale), "soc_plan": soc_0 + soc @ x,
            "base_terminal_soc_deviation_mwh": float(terminal[0] @ x)}


def _perfect_information(study: dict, storage_source: dict, panels: dict,
                         days: list) -> dict:
    arm = storage_source["storage"]["arms"][study["storage"]["arm"]]
    storage, solver = storage_settings(storage_source), storage_source["solver"]
    totals = {"full_delivery": 0.0, "economic_shortfall": 0.0, "constrained_oracle": 0.0}
    maximum_delivery_gap = 0.0
    for day in days:
        full, solution = _perfect_day(
            panels[day], study, storage_source, arm, storage,
            study["recourse"]["delivery_0730"])
        shortfall, _ = _perfect_day(
            panels[day], study, storage_source, arm, storage,
            study["recourse"]["shortfall_sensitivity"])
        oracle = _oracle_with_base_constraints(panels[day], arm, storage, solver)
        totals["constrained_oracle"] += settle_day(oracle, panels[day], arm, storage, solver)["total_eur"]
        totals["full_delivery"] += full["total_eur"]
        totals["economic_shortfall"] += shortfall["total_eur"]
        maximum_delivery_gap = max(
            maximum_delivery_gap, solution["delivery_shortfall_from_maximum_mwh"])
    s2_oracle = float(json.loads((ROOT / study["references"]["s2_baseline_summary"]).read_text())[
        "strategies"]["Oracle"]["total_eur"]["total"])
    reference = totals["constrained_oracle"]
    difference = totals["full_delivery"] - reference
    relative = abs(difference) / abs(reference)
    _require(relative <= study["acceptance"]["oracle_relative_tolerance"],
             f"Perfect-information relative difference is {relative}")
    return {"b3_total_eur": totals["full_delivery"],
            "reference": "s2_oracle_lp_plus_solve_day_no_activation_constraints",
            "reference_total_eur": reference, "difference_eur": difference,
            "relative_absolute_difference": relative, "s2_oracle_total_eur": s2_oracle,
            "s2_oracle_no_activation_relaxation_value_eur": s2_oracle - reference,
            "shortfall_sensitivity_total_eur": totals["economic_shortfall"],
            "maximum_delivery_shortfall_from_optimum_mwh": maximum_delivery_gap}


def _forecast_vector(frame: pd.DataFrame, day: str, target: str) -> np.ndarray:
    rows = frame.loc[frame["delivery_day"].eq(day) & frame["target"].eq(target)].sort_values("hour")
    values = rows["forecast"].to_numpy(dtype=float)
    _require(len(values) == 24 and np.isfinite(values).all(), f"{day} {target}: bad forecast")
    return values


def _f08_settlement(study: dict, storage_source: dict, panels: dict,
                    days: list) -> dict:
    forecasts = pd.read_csv(ROOT / storage_source["comparison"]["f08_forecasts_csv"])
    arm = storage_source["storage"]["arms"][study["storage"]["arm"]]
    storage = storage_settings(storage_source)
    total = 0.0
    for day in days:
        panel = panels[day]
        day_ahead = resize(np.repeat(_forecast_vector(forecasts, day, "day_ahead_price"), 4),
                           len(panel["day_ahead_price"]))
        capacity = resize(_forecast_vector(forecasts, day, "capacity_price"),
                          len(panel["capacity_price"]))
        plan = solve_day(day_ahead, capacity, panel["procured_mw"], panel["hour_of_slot"],
                         arm, storage, storage_source["solver"], None)
        total += settle_day(plan, panel, arm, storage, storage_source["solver"])["total_eur"]
    expected = json.loads((ROOT / study["references"]["s2c_summary"]).read_text())[
        "arms"]["B2_F08"]["total_eur"]
    error = abs(total - float(expected))
    _require(error <= study["acceptance"]["settlement_absolute_tolerance_eur"],
             f"F08 frozen settlement differs by {error}")
    return {"total_eur": total, "reference_eur": float(expected), "absolute_error_eur": error}


def _x14_check(study: dict, forecast: dict, panels: dict, days: list) -> dict:
    values, checks = generate_x14_archive(panels, days, forecast, study)
    expected = (len(days), int(study["export"]["saved_scenario_count"]), 4, 24)
    _require(values.shape == expected and values.dtype == np.float32,
             f"X14 archive shape or dtype differs: {values.shape}, {values.dtype}")
    _require(checks["all_provenance_checks"], "X14 provenance check failed")
    return {"shape": list(values.shape), **checks}


def run_checks(path: Path) -> dict:
    study, forecast, storage_source = load_configs(path)
    panels = build_panels(ROOT / storage_source["data"]["root"], storage_source)
    days, threshold = settlement_protocol(panels, storage_source)
    _require(len(days) == 211 and threshold == 6.69, "Frozen settlement protocol changed")
    wave0 = json.loads((ROOT / study["output"]["directory"]
                        / study["output"]["summary_json"]).read_text())
    _require(all(wave0["acceptance"].values()), "Wave-0 acceptance contains a failure")
    return {"papers": _paper_checks(study), "hand_checks": _hand_checks(study),
            "perfect_information": _perfect_information(study, storage_source, panels, days),
            "f08_settlement": _f08_settlement(study, storage_source, panels, days),
            "x14_generation": _x14_check(study, forecast, panels, days),
            "environment": environment_fingerprint(), "test_days": len(days)}


def run_export(path: Path) -> dict:
    study, forecast, storage_source = load_configs(path)
    result = scenario_export.run(str(path))
    panels = build_panels(ROOT / storage_source["data"]["root"], storage_source)
    days, _ = settlement_protocol(panels, storage_source)
    result["x14"] = write_x14_archive(ROOT, panels, days, forecast, study)
    return result


def _load_x14(study: dict, days: list) -> dict:
    path = (ROOT / study["output"]["directory"]
            / study["output"]["scenarios_0730_directory"] / "x14_rebuild.npz")
    _require(path.is_file(), "X14 archive is missing; run --export first")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    result.update({"path": path, "delivery_days": result["delivery_days"].astype(str).tolist(),
                   "targets": result["targets"].astype(str).tolist(),
                   "evaluated_days": result["evaluated_days"].astype(str).tolist(),
                   "values": result["values"],
                   "activation_representation": str(result["activation_representation"].item()),
                   "information_set": str(result["information_set"].item())})
    _require(result["delivery_days"] == days, "X14 delivery days differ")
    return result


def _archive_index(loaded: dict, x14: dict) -> dict:
    output = {}
    for item in loaded.values():
        key = (item["path"].parent.name, item["path"].stem)
        _require(key not in output, f"Duplicate scenario archive key: {key}")
        output[key] = item
    output[(x14["path"].parent.name, "x14_rebuild")] = x14
    return output


def _b3_specs() -> list:
    return [
        ("B3_hist_paired", "hist_paired", None, "full"),
        ("B3_hist_independent", "hist_independent", None, "full"),
        ("B3_pooled_independent", "pooled_independent", None, "full"),
        ("B3_pooled_gaussian_no_cross", "pooled_gaussian_no_cross_target", None, "full"),
        ("B3_pooled_gaussian", "pooled_gaussian", None, "full"),
        ("B3_pooled_gaussian_gate", "pooled_gaussian",
         "pooled_gaussian__gaussian_24_hour", "full"),
        ("B3_pooled_gaussian_directqr", "pooled_gaussian",
         "pooled_gaussian__direct_qr", "full"),
        ("B3_pooled_empirical", "pooled_empirical", None, "full"),
        ("B3_pooled_empirical_analog", "pooled_empirical",
         "pooled_empirical__analog", "full"),
        ("B3_pooled_empirical_directqr", "pooled_empirical",
         "pooled_empirical__direct_qr", "full"),
        ("B3_fb2_s3a", "fb2_gate", "fb2_gate", "full"),
        ("B3_fb2_plus_weather_s3a", "fb2_plus_weather", "fb2_plus_weather", "full"),
        ("B3_hist_paired_shortfall", "hist_paired", None, "economic"),
    ]


def _load_points(study: dict) -> dict:
    early = pd.read_csv(ROOT / "p1_paper/results/s3_forecast_side/point_0730/fb0_lgbm.csv")
    late = pd.read_csv(ROOT / study["output"]["directory"]
                       / study["output"]["point_1200_directory"]
                       / f'{study["export"]["point_1200_id"]}.csv')
    return {"early": early, "late": late}


def _capacity_history(day: str, panels: dict, forecast: dict, hours: int) -> np.ndarray:
    fit = window_days(day, forecast)["fit"]
    _require(set(fit).issubset(panels), f"{day}: capacity F(D) is incomplete")
    return np.stack([resize(panels[source]["capacity_price"], hours) for source in fit])


def _prepared(archive: dict, day: str, panel: dict, history: pd.DataFrame,
              forecast: dict, seed: int) -> dict:
    result = prepare_day_scenarios(
        archive, day, panel["procured_mw"], history, forecast, "gate_0730", seed)
    result["capacity_price"] = (result["capacity_price"] if "capacity_price" in result
                                else np.repeat(panel["capacity_price"][None, :],
                                               len(result["day_ahead_price"]), axis=0))
    return result


def _late_prepared(archive: dict, day: str, panel: dict, history: pd.DataFrame,
                   forecast: dict, seed: int) -> dict:
    result = prepare_day_scenarios(
        archive, day, panel["procured_mw"], history, forecast, "gate_1200", seed)
    result["capacity_price"] = np.repeat(
        panel["capacity_price"][None, :], len(result["day_ahead_price"]), axis=0)
    return result


def _init_full_worker(state: dict) -> None:
    WORKER_STATE.update(state)


def _fallback_plan(day: str, panel: dict) -> dict:
    state = WORKER_STATE
    storage_source, arm, storage = state["storage_source"], state["arm"], state["storage"]
    lag = int(storage_source["persistence"]["day_ahead_lag_days"])
    previous = (pd.Timestamp(day) - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
    _require(previous in state["panels"], f"{day}: B1 fallback persistence day is missing")
    source = state["panels"][previous]
    day_ahead = resize(source["day_ahead_price"], len(panel["day_ahead_price"]))
    capacity = resize(source["capacity_price"], len(panel["capacity_price"]))
    fixed = threshold_rule(capacity, state["threshold"], arm, panel["procured_mw"])
    plan = solve_day(day_ahead, capacity, panel["procured_mw"], panel["hour_of_slot"],
                     arm, storage, storage_source["solver"], fixed)
    return {**plan, "solver_settings": "b1_rule_solve_day"}


def _daily_record(day: str, strategy: str, chi: float, plan: dict,
                  panel: dict, fallback: bool, early_status: int,
                  delivery_status: int, profit_status: int,
                  delivery_gap: float) -> dict:
    state = WORKER_STATE
    closed = (solve_recovery(plan, panel, state["arm"], state["storage"],
                             state["storage_source"]["solver"])
              if "p_recovery" not in plan else
              {"required": plan["r_up"][panel["hour_of_slot"]] * panel["activation_share"],
               "delivered": plan["r_up"][panel["hour_of_slot"]] * panel["activation_share"]})
    settled = settle_day(plan, panel, state["arm"], state["storage"],
                         state["storage_source"]["solver"])
    required = settled["required_activation_mwh"]
    undelivered = required - settled["delivered_activation_mwh"]
    hourly_volume = np.asarray([
        np.sum(panel["activation_share"][panel["hour_of_slot"] == hour]
               * panel["procured_mw"][hour]) * state["storage"]["slot_hours"]
        for hour in range(len(panel["procured_mw"]))])
    stress_hours = ((panel["capacity_price"] >= state["study"]["evaluation"][
        "stress_capacity_price_eur_per_mw"])
        | (hourly_volume >= state["study"]["evaluation"][
            "stress_activation_volume_mwh_per_hour"]))
    stress_slots = stress_hours[panel["hour_of_slot"]]
    stress_required = float(np.sum(closed["required"][stress_slots]))
    stress_delivered = float(np.sum(closed["delivered"][stress_slots]))
    return {"delivery_day": day, "strategy": strategy, "chi": float(chi),
            "planned_profit_eur": plan["planned_profit_eur"],
            "base_terminal_soc_deviation_mwh": plan["base_terminal_soc_deviation_mwh"],
            "exclusivity_violations": exclusivity_violations(
                plan, state["study"]["solver"]["exclusivity_tolerance_mw"]),
            "fallback": int(fallback), "solver_settings": plan["solver_settings"],
            "solver_0730_status": int(early_status),
            "solver_1200_delivery_status": int(delivery_status),
            "solver_1200_profit_status": int(profit_status),
            "delivery_shortfall_from_maximum_mwh": float(delivery_gap),
            "undelivered_activation_mwh": float(undelivered),
            "undelivered_activation_share": float(undelivered / required) if required > 0 else 0.0,
            "stress_hours": int(stress_hours.sum()),
            "stress_required_activation_mwh": stress_required,
            "stress_delivered_activation_mwh": stress_delivered,
            **settled}


def _scenario_profit_distribution(plan: dict, scenarios: dict, panel: dict) -> np.ndarray:
    state = WORKER_STATE
    profits = []
    for index in range(len(scenarios["day_ahead_price"])):
        realised = {"day_ahead_price": scenarios["day_ahead_price"][index],
                    "capacity_price": scenarios["capacity_price"][index],
                    "activation_share": scenarios["activation_share"][index],
                    "activation_price": scenarios["activation_price"][index],
                    "shortfall_price": scenarios["shortfall_price"][index],
                    "procured_mw": panel["procured_mw"],
                    "hour_of_slot": panel["hour_of_slot"]}
        profits.append(settle_day(plan, realised, state["arm"], state["storage"],
                                  state["storage_source"]["solver"])["total_eur"])
    return np.asarray(profits)


def _solve_b3(day: str, index: int, strategy: str, early_name: str,
              late_name: str, mode: str, chi: float) -> tuple:
    state = WORKER_STATE
    panel, study, forecast = state["panels"][day], state["study"], state["forecast"]
    seed = int(study["recourse"]["premium_sampling_seed"]) + index
    early_archive = state["archives"][(study["output"]["scenarios_0730_directory"], early_name)]
    if day not in early_archive["evaluated_days"]:
        return None, None, None
    early_scenarios = _prepared(early_archive, day, panel, state["premium"], forecast, seed)
    history = _capacity_history(day, state["panels"], forecast, len(panel["capacity_price"]))
    labels = cluster_capacity(early_scenarios["capacity_price"], history, study)
    probabilities = np.full(len(labels), 1.0 / len(labels))
    early = solve_stochastic(
        early_scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
        state["arm"], state["storage"], study["solver"], study["risk"], chi, None,
        mode, "profit", None)
    if not early["success"]:
        plan = _fallback_plan(day, panel)
        row = _daily_record(day, strategy, chi, plan, panel, True,
                            early["status"], -1, -1, 0.0)
        return row, None, {"plan": plan, "scenarios": early_scenarios}
    late_archive = early_archive if late_name is None else state["archives"][
        (study["output"]["scenarios_1200_directory"], late_name)]
    late_scenarios = (_prepared(late_archive, day, panel, state["premium"], forecast, seed)
                      if late_name is None else _late_prepared(
                          late_archive, day, panel, state["premium"], forecast, seed))
    late_scenarios["capacity_price"] = np.repeat(
        panel["capacity_price"][None, :], len(late_scenarios["day_ahead_price"]), axis=0)
    late_labels = np.zeros(len(late_scenarios["day_ahead_price"]), dtype=int)
    late_probabilities = np.full(len(late_labels), 1.0 / len(late_labels))
    late = _late_solve(late_scenarios, late_probabilities, late_labels, panel, early, mode, chi)
    if not late["success"]:
        plan = _fallback_plan(day, panel)
        row = _daily_record(day, strategy, chi, plan, panel, True, early["solver_status"],
                            late.get("delivery_first_status", -1), late["status"], 0.0)
        return row, None, {"plan": plan, "scenarios": late_scenarios}
    plan = deployment_plan(late)
    plan["solver_settings"] = f'{early["solver_setting"]}|{late["solver_setting"]}'
    row = _daily_record(day, strategy, chi, plan, panel, False, early["solver_status"],
                        late.get("delivery_first_status", -1), late["solver_status"],
                        late.get("delivery_shortfall_from_maximum_mwh", 0.0))
    if row["fallback"]:
        plan = _fallback_plan(day, panel)
    profits = _scenario_profit_distribution(plan, late_scenarios, panel)
    risk = calibration_row(day, strategy, chi, profits, row["total_eur"],
                           study["risk"]["cvar_level"])
    return row, risk, {"plan": plan, "scenarios": late_scenarios}


def _late_solve(scenarios: dict, probabilities: np.ndarray, labels: np.ndarray,
                panel: dict, early: dict, mode: str, chi: float) -> dict:
    state, study = WORKER_STATE, WORKER_STATE["study"]
    if mode == study["recourse"]["shortfall_sensitivity"]:
        return solve_stochastic(
            scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
            state["arm"], state["storage"], study["solver"], study["risk"], chi,
            early["r_up"], mode, "profit", None)
    return solve_delivery_first(
        scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
        state["arm"], state["storage"], study["solver"], study["risk"], chi,
        early["r_up"], study["recourse"]["delivery_tolerance_mwh"])


def _point_plan(frame: pd.DataFrame, day: str, panel: dict,
                fixed_reserve: np.ndarray) -> dict:
    state = WORKER_STATE
    day_ahead = resize(np.repeat(_forecast_vector(frame, day, "day_ahead_price"), 4),
                       len(panel["day_ahead_price"]))
    capacity = resize(_forecast_vector(frame, day, "capacity_price"),
                      len(panel["capacity_price"])) if "capacity_price" in set(frame["target"]) \
        else panel["capacity_price"]
    if fixed_reserve is not None:
        ceiling = np.minimum(state["arm"]["power_mw"],
                             state["arm"]["market_share_cap"] * panel["procured_mw"])
        cleaned = np.minimum(np.maximum(fixed_reserve, 0.0), ceiling)
        error = float(np.max(np.abs(cleaned - fixed_reserve)))
        _require(error <= state["study"]["solver"]["exclusivity_tolerance_mw"],
                 f"{day}: fixed point reserve violates bounds by {error}")
        fixed_reserve = cleaned
    plan = solve_day(day_ahead, capacity, panel["procured_mw"], panel["hour_of_slot"],
                     state["arm"], state["storage"], state["storage_source"]["solver"],
                     fixed_reserve)
    return {**plan, "solver_settings": "solve_day"}


def _point_or_fallback(frame: pd.DataFrame, day: str, panel: dict,
                       fixed_reserve: np.ndarray) -> tuple:
    try:
        return _point_plan(frame, day, panel, fixed_reserve), False, 0
    except ValueError as error:
        if not str(error).startswith("LP did not solve:"):
            raise
        return _fallback_plan(day, panel), True, 2


def _solve_b2(day: str) -> list:
    panel, points = WORKER_STATE["panels"][day], WORKER_STATE["points"]
    early, early_fallback, early_status = _point_or_fallback(
        points["early"], day, panel, None)
    plain = _daily_record(day, "B2_FB0", 0.0, early, panel, early_fallback,
                          early_status, -1, -1, 0.0)
    if early_fallback:
        late, late_fallback, late_status = early, True, early_status
    else:
        late, late_fallback, late_status = _point_or_fallback(
            points["late"], day, panel, early["r_up"])
    gate = _daily_record(day, "B2_FB0_gate", 0.0, late, panel, late_fallback,
                         early_status, -1, late_status, 0.0)
    return [plain, gate]


def _x14_cluster(labels: np.ndarray, capacity: np.ndarray,
                 realised: np.ndarray, history: np.ndarray) -> np.ndarray:
    scale = np.std(history, axis=0)
    _require((scale > 0.0).all(), "X14 capacity scale contains zero")
    centres = np.stack([capacity[labels == label].mean(axis=0)
                        for label in sorted(np.unique(labels))])
    selected = int(np.argmin(np.sum(((centres - realised) / scale) ** 2, axis=1)))
    indices = np.flatnonzero(labels == selected)
    _require(len(indices) > 0, "X14 selected scenario cluster is empty")
    return indices


def _subset_scenarios(scenarios: dict, indices: np.ndarray) -> dict:
    count = len(scenarios["day_ahead_price"])
    return {key: (value[indices] if isinstance(value, np.ndarray)
                  and value.ndim > 0 and len(value) == count else value)
            for key, value in scenarios.items()}


def _solve_x14(day: str, index: int, chi: float) -> tuple:
    state = WORKER_STATE
    panel, study, forecast = state["panels"][day], state["study"], state["forecast"]
    seed = int(study["recourse"]["premium_sampling_seed"]) + index
    archive = state["archives"][(study["output"]["scenarios_0730_directory"], "x14_rebuild")]
    scenarios = _prepared(archive, day, panel, state["premium"], forecast, seed)
    history = _capacity_history(day, state["panels"], forecast, len(panel["capacity_price"]))
    labels = cluster_capacity(scenarios["capacity_price"], history, study)
    probabilities = np.full(len(labels), 1.0 / len(labels))
    early = solve_stochastic(
        scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
        state["arm"], state["storage"], study["solver"], study["risk"], chi, None,
        "full", "profit", None)
    strategy = "X14_rebuild"
    if not early["success"]:
        plan = _fallback_plan(day, panel)
        return _daily_record(day, strategy, chi, plan, panel, True, early["status"], -1, -1, 0.0), None
    selected = _x14_cluster(labels, scenarios["capacity_price"], panel["capacity_price"], history)
    late_scenarios = _subset_scenarios(scenarios, selected)
    late_scenarios["capacity_price"] = np.repeat(
        panel["capacity_price"][None, :], len(selected), axis=0)
    late_labels, late_probabilities = np.zeros(len(selected), dtype=int), np.full(len(selected), 1 / len(selected))
    late = _late_solve(late_scenarios, late_probabilities, late_labels, panel, early, "full", chi)
    if not late["success"]:
        plan = _fallback_plan(day, panel)
        return _daily_record(day, strategy, chi, plan, panel, True, early["solver_status"],
                             late.get("delivery_first_status", -1), late["status"], 0.0), None
    plan = deployment_plan(late)
    plan["solver_settings"] = f'{early["solver_setting"]}|{late["solver_setting"]}'
    row = _daily_record(day, strategy, chi, plan, panel, False, early["solver_status"],
                        late["delivery_first_status"], late["solver_status"],
                        late["delivery_shortfall_from_maximum_mwh"])
    if row["fallback"]:
        plan = _fallback_plan(day, panel)
    profits = _scenario_profit_distribution(plan, late_scenarios, panel)
    return row, calibration_row(day, strategy, chi, profits, row["total_eur"],
                                study["risk"]["cvar_level"])


def _solve_x09(day: str, index: int) -> dict:
    state = WORKER_STATE
    panel, study, forecast = state["panels"][day], state["study"], state["forecast"]
    archive = state["archives"][(study["output"]["scenarios_0730_directory"], "x09_block_copula")]
    seed = int(study["recourse"]["premium_sampling_seed"]) + index
    scenarios = _prepared(archive, day, panel, state["premium"], forecast, seed)
    labels, probabilities = np.zeros(len(scenarios["day_ahead_price"]), dtype=int), \
        np.full(len(scenarios["day_ahead_price"]), 1 / len(scenarios["day_ahead_price"]))
    solution = solve_stochastic(
        scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
        state["arm"], state["storage"], study["solver"], study["risk"], 0.0, None,
        "full", "profit", None)
    if not solution["success"]:
        return _daily_record(day, "X09_rebuild", 0.0, _fallback_plan(day, panel), panel,
                             True, solution["status"], -1, -1, 0.0)
    return _daily_record(day, "X09_rebuild", 0.0, deployment_plan(solution), panel,
                         False, solution["solver_status"], -1, -1, 0.0)


def _solve_x06(day: str, index: int) -> dict:
    state = WORKER_STATE
    panel, study, forecast = state["panels"][day], state["study"], state["forecast"]
    early = state["points"]["early"]
    day_ahead = _forecast_vector(early, day, "day_ahead_price")
    capacity = _forecast_vector(early, day, "capacity_price")
    climatology = state["archives"][(study["output"]["scenarios_0730_directory"], "climatology")]
    seed = int(study["recourse"]["premium_sampling_seed"]) + index
    scenarios = x06_deterministic_scenario(
        day_ahead, capacity, climatology, day, panel["procured_mw"], state["premium"],
        forecast, seed, float(study["published_rules"]["x06"]["epsilon"]))
    probabilities, labels = np.ones(1), np.zeros(1, dtype=int)
    solution = solve_stochastic(
        scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
        state["arm"], state["storage"], study["solver"], study["risk"], 0.0, None,
        "full", "profit", None)
    if not solution["success"]:
        return _daily_record(day, "X06_rebuild", 0.0, _fallback_plan(day, panel), panel,
                             True, solution["status"], -1, -1, 0.0)
    return _daily_record(day, "X06_rebuild", 0.0, deployment_plan(solution), panel,
                         False, solution["solver_status"], -1, -1, 0.0)


def _cross_source_names() -> set:
    return {"B3_hist_paired", "B3_hist_independent", "B3_pooled_independent",
            "B3_pooled_gaussian_no_cross", "B3_pooled_gaussian",
            "B3_pooled_empirical"}


def _cross_rows(day: str, artifacts: dict) -> list:
    state = WORKER_STATE
    alpha = 1.0 - float(state["study"]["risk"]["cvar_level"])
    rows = []
    for model, model_artifact in sorted(artifacts.items()):
        actual = model_artifact["actual_profit_eur"] / 1000.0
        for source, source_artifact in sorted(artifacts.items()):
            profits = _scenario_profit_distribution(
                model_artifact["plan"], source_artifact["scenarios"], state["panels"][day])
            tail = empirical_tail_metrics(profits / 1000.0, alpha)
            rows.append({"delivery_day": day, "model_strategy": model,
                         "scenario_strategy": source, **tail,
                         "realised_profit_k_eur": actual,
                         "joint_score": joint_var_cvar_score(
                             tail["var_k_eur"], tail["cvar_k_eur"], actual, alpha)})
    return rows


def _day_worker(item: tuple) -> dict:
    index, day = item
    rows, risks, cross_artifacts = _solve_b2(day), [], {}
    for strategy, early_name, late_name, mode in _b3_specs():
        for chi in WORKER_STATE["study"]["risk"]["chi_levels"]:
            row, risk, artifact = _solve_b3(
                day, index, strategy, early_name, late_name, mode, float(chi))
            if row is None:
                continue
            rows.append(row)
            if risk is not None:
                risks.append(risk)
            if float(chi) == 0.5 and strategy in _cross_source_names():
                artifact["actual_profit_eur"] = row["total_eur"]
                cross_artifacts[strategy] = artifact
    for chi in WORKER_STATE["study"]["risk"]["x14_chi_levels"]:
        row, risk = _solve_x14(day, index, float(chi))
        rows.append(row)
        if risk is not None:
            risks.append(risk)
    rows.extend([_solve_x09(day, index), _solve_x06(day, index)])
    _require(set(cross_artifacts) == _cross_source_names(),
             f"{day}: cross-score arm set is incomplete")
    return {"daily": rows, "risk": risks, "cross": _cross_rows(day, cross_artifacts)}


def _arm_summary(daily: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (strategy, chi), frame in daily.groupby(["strategy", "chi"], sort=True):
        required = float(frame["required_activation_mwh"].sum())
        undelivered = float(frame["undelivered_activation_mwh"].sum())
        tail = empirical_tail_metrics(frame["total_eur"].to_numpy() / 1000.0, 0.05)
        records.append({"strategy": strategy, "chi": chi, "days": len(frame),
                        "total_eur": float(frame["total_eur"].sum()),
                        "mean_daily_eur": float(frame["total_eur"].mean()),
                        "profit_cvar_5pct_eur": 1000.0 * tail["cvar_k_eur"],
                        "energy_eur": float(frame["energy_eur"].sum()),
                        "capacity_eur": float(frame["capacity_eur"].sum()),
                        "activation_eur": float(frame["activation_eur"].sum()),
                        "recovery_cost_eur": float(frame["recovery_cost_eur"].sum()),
                        "shortfall_cost_eur": float(frame["shortfall_cost_eur"].sum()),
                        "required_activation_mwh": required,
                        "delivered_activation_mwh": float(frame["delivered_activation_mwh"].sum()),
                        "undelivered_activation_mwh": undelivered,
                        "undelivered_activation_share": undelivered / required if required else 0.0,
                        "days_with_shortfall": int(frame["slots_with_shortfall"].gt(0).sum()),
                        "committed_mw_hours": float(frame["committed_mw_hours"].sum()),
                        "fallback_days": int(frame["fallback"].sum()),
                        "max_terminal_soc_deviation_mwh": float(
                            frame["terminal_soc_deviation_mwh"].abs().max()),
                        "max_committed_gross_power_mw": float(
                            frame["max_committed_gross_power_mw"].max())})
    return pd.DataFrame(records)


def _paired_one(daily: pd.DataFrame, left: tuple, right: tuple,
                metric: str, study: dict) -> dict:
    left_rows = daily.loc[(daily["strategy"] == left[0]) & (daily["chi"] == left[1]),
                          ["delivery_day", metric]]
    right_rows = daily.loc[(daily["strategy"] == right[0]) & (daily["chi"] == right[1]),
                           ["delivery_day", metric]]
    joined = left_rows.merge(right_rows, on="delivery_day", suffixes=("_left", "_right"),
                             validate="one_to_one")
    _require(len(joined) == min(len(left_rows), len(right_rows)),
             f"{left}/{right}: paired days differ")
    result = paired_metrics(joined[f"{metric}_left"], joined[f"{metric}_right"], study)
    return {"strategy": left[0], "chi": left[1], "comparator": right[0],
            "comparator_chi": right[1], "metric": metric,
            "mean_difference_strategy_minus_comparator": -result[
                "mean_loss_difference_a_minus_b"],
            "confidence_lower": -result["confidence_upper"],
            "confidence_upper": -result["confidence_lower"],
            "dm_statistic": -result["dm_statistic"], "p_value": result["p_value"],
            "n_days": result["n_valid_days"]}


def _comparison_pairs(summary: pd.DataFrame) -> list:
    arms = [(row.strategy, float(row.chi)) for row in summary.itertuples()
            if row.strategy != "B2_FB0"]
    pairs = [(arm, ("B2_FB0", 0.0)) for arm in arms]
    pairs += [(arm, ("X14_rebuild", 0.0)) for arm in arms
              if arm != ("X14_rebuild", 0.0)]
    internal = [("B3_hist_paired", "B3_hist_independent"),
                ("B3_pooled_gaussian", "B3_pooled_independent"),
                ("B3_pooled_empirical", "B3_pooled_independent"),
                ("B3_pooled_gaussian", "B3_pooled_gaussian_no_cross"),
                ("B3_pooled_gaussian", "B3_pooled_empirical"),
                ("B3_pooled_gaussian_gate", "B3_pooled_gaussian"),
                ("B3_pooled_gaussian_directqr", "B3_pooled_gaussian"),
                ("B3_pooled_empirical_analog", "B3_pooled_empirical"),
                ("B3_pooled_empirical_directqr", "B3_pooled_empirical"),
                ("B3_pooled_gaussian", "B3_fb2_s3a"),
                ("B3_fb2_plus_weather_s3a", "B3_fb2_s3a"),
                ("B3_hist_paired", "B3_hist_paired_shortfall")]
    for chi in (0.0, 0.5):
        pairs.extend([((left, chi), (right, chi)) for left, right in internal])
    risk_arms = sorted(set(row.strategy for row in summary.itertuples()
                           if row.strategy.startswith("B3_")))
    pairs.extend([((arm, 0.5), (arm, 0.0)) for arm in risk_arms])
    return list(dict.fromkeys(pairs))


def _paired_differences(daily: pd.DataFrame, summary: pd.DataFrame,
                        study: dict) -> pd.DataFrame:
    rows = []
    present = set(zip(summary["strategy"], summary["chi"].astype(float)))
    for left, right in _comparison_pairs(summary):
        if left not in present or right not in present:
            continue
        for metric in ("total_eur", "undelivered_activation_share", "committed_mw_hours"):
            rows.append(_paired_one(daily, left, right, metric, study))
    return pd.DataFrame(rows)


def _frontier(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in summary.itertuples():
        dominated = ((summary["total_eur"] >= candidate.total_eur)
                     & (summary["undelivered_activation_share"]
                        <= candidate.undelivered_activation_share)
                     & ((summary["total_eur"] > candidate.total_eur)
                        | (summary["undelivered_activation_share"]
                           < candidate.undelivered_activation_share))).any()
        rows.append({"strategy": candidate.strategy, "chi": candidate.chi,
                     "total_eur": candidate.total_eur,
                     "undelivered_activation_share": candidate.undelivered_activation_share,
                     "dominated": int(dominated)})
    return pd.DataFrame(rows)


def _monthly(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = frame["delivery_day"].str[:7]
    rows = []
    for (strategy, chi, month), group in frame.groupby(
            ["strategy", "chi", "month"], sort=True):
        required = float(group["required_activation_mwh"].sum())
        undelivered = float(group["undelivered_activation_mwh"].sum())
        rows.append({"strategy": strategy, "chi": chi, "month": month,
                     "days": len(group), "total_eur": float(group["total_eur"].sum()),
                     "required_activation_mwh": required,
                     "undelivered_activation_mwh": undelivered,
                     "undelivered_activation_share": undelivered / required if required else 0.0})
    return pd.DataFrame(rows)


def _stress_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, chi), group in daily.groupby(["strategy", "chi"], sort=True):
        required = float(group["stress_required_activation_mwh"].sum())
        delivered = float(group["stress_delivered_activation_mwh"].sum())
        rows.append({"strategy": strategy, "chi": chi,
                     "stress_hours": int(group["stress_hours"].sum()),
                     "stress_required_activation_mwh": required,
                     "stress_delivered_activation_mwh": delivered,
                     "stress_undelivered_activation_mwh": required - delivered,
                     "stress_undelivered_activation_share": (
                         (required - delivered) / required if required else 0.0)})
    return pd.DataFrame(rows)


def _cross_summary(daily_cross: pd.DataFrame, study: dict) -> pd.DataFrame:
    rows = []
    for (model, source), frame in daily_cross.groupby(
            ["model_strategy", "scenario_strategy"], sort=True):
        own = daily_cross.loc[(daily_cross["model_strategy"] == model)
                              & (daily_cross["scenario_strategy"] == model),
                              ["delivery_day", "joint_score"]]
        joined = frame[["delivery_day", "joint_score"]].merge(
            own, on="delivery_day", suffixes=("_candidate", "_self"),
            validate="one_to_one")
        if source == model:
            test = {"mean_score_difference_i_minus_self": 0.0,
                    "dm_statistic": 0.0, "p_value_two_sided": 1.0,
                    "n_days": len(joined),
                    "newey_west_lags": study["evaluation"]["dm_newey_west_lags"],
                    "null": "candidate score minus self score >= 0"}
        else:
            test = cross_score_dm(joined["joint_score_candidate"],
                                  joined["joint_score_self"], study)
        rows.append({"model_strategy": model, "scenario_strategy": source,
                     "mean_joint_score": float(frame["joint_score"].mean()), **test})
    return pd.DataFrame(rows)


def _forecast_score_table() -> pd.DataFrame:
    first = pd.read_csv(ROOT / "p1_paper/results/s3_forecast_side/daily_scores.csv")
    second = pd.read_csv(ROOT / "p1_paper/results/s3_cross_market/daily_joint_scores.csv")
    mapping = {"B3_hist_paired": (first, "hist_paired"),
               "B3_hist_independent": (first, "hist_independent"),
               "B3_pooled_independent": (second, "own_pooled__independent"),
               "B3_pooled_gaussian_no_cross": (first, "fb2_no_cross_target"),
               "B3_pooled_gaussian": (second, "own_pooled__gaussian"),
               "B3_pooled_empirical": (second, "own_pooled__empirical")}
    rows = []
    for strategy, (frame, arm) in mapping.items():
        selected = frame.loc[(frame["arm"] == arm) & (frame["seed"] == 1)
                             & (frame["block"] == "all")]
        _require(len(selected) == 211, f"{strategy}: RQ4 score days differ")
        rows.append({"strategy": strategy,
                     "energy_score": float(selected["energy_score"].sum()),
                     "variogram_score": float(selected["variogram_score"].sum())})
    return pd.DataFrame(rows)


def _rq4(summary: pd.DataFrame) -> pd.DataFrame:
    forecast_scores = _forecast_score_table()
    rows = []
    for chi in (0.0, 0.5):
        decision = summary.loc[summary["chi"].eq(chi)].merge(
            forecast_scores, on="strategy", validate="one_to_one")
        for forecast_metric in ("energy_score", "variogram_score"):
            for decision_metric, sign in (("total_eur", -1.0),
                                          ("undelivered_activation_share", 1.0)):
                test = stats.kendalltau(decision[forecast_metric],
                                        sign * decision[decision_metric])
                rows.append({"chi": chi, "forecast_metric": forecast_metric,
                             "decision_metric": decision_metric, "arms": len(decision),
                             "kendall_tau": float(test.statistic),
                             "p_value": float(test.pvalue)})
    return pd.DataFrame(rows)


def _reference_rows(study: dict) -> pd.DataFrame:
    baseline = json.loads((ROOT / study["references"]["s2_baseline_summary"]).read_text())
    decision = json.loads((ROOT / study["references"]["s2c_summary"]).read_text())
    rows = []
    for strategy in ("B0", "B1", "Oracle"):
        source = baseline["strategies"][strategy]
        required = source["required_activation_mwh"]["total"]
        delivered = source["delivered_activation_mwh"]["total"]
        rows.append({"strategy": strategy, "chi": 0.0, "days": source["days"],
                     "total_eur": source["total_eur"]["total"],
                     "mean_daily_eur": source["total_eur"]["mean"],
                     "profit_cvar_5pct_eur": source["profit_cvar_5pct_eur"],
                     "undelivered_activation_share": (
                         (required - delivered) / required if required else 0.0),
                     "evidence_role": "read_only_reference"})
    for strategy in ("B2_F01", "B2_F08", "oracle_price_reserve"):
        source = decision["arms"][strategy]
        rows.append({"strategy": strategy, "chi": 0.0, "days": source["days"],
                     "total_eur": source["total_eur"],
                     "mean_daily_eur": source["total_eur"] / source["days"],
                     "profit_cvar_5pct_eur": np.nan,
                     "undelivered_activation_share": source["undelivered_activation_share"],
                     "evidence_role": "read_only_reference_116_day_forecast_window"})
    return pd.DataFrame(rows)


def _output_hashes(directory: Path) -> dict:
    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix in {".csv", ".npz"})
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _validate_full(daily: pd.DataFrame, study: dict, arm: dict,
                   storage: dict) -> dict:
    fallback = daily.groupby(["strategy", "chi"])["fallback"].agg(["sum", "count"])
    fallback["share"] = fallback["sum"] / fallback["count"]
    maximum_fallback = float(fallback["share"].max())
    _require(maximum_fallback <= study["solver"]["max_fallback_share"],
             f"Fallback share exceeds the frozen limit: {maximum_fallback}")
    terminal = float(daily["terminal_soc_deviation_mwh"].abs().max())
    _require(terminal <= storage["soc_tolerance_mwh"],
             f"Terminal SOC deviation exceeds tolerance: {terminal}")
    gross = float(daily[["max_committed_gross_power_mw",
                         "max_realised_gross_power_mw"]].max().max())
    _require(gross <= arm["power_mw"] + study["solver"]["exclusivity_tolerance_mw"],
             f"Gross power exceeds the frozen limit: {gross}")
    full = daily.loc[daily["strategy"].str.startswith("B3_")
                     & ~daily["strategy"].eq("B3_hist_paired_shortfall")
                     & daily["fallback"].eq(0)]
    delivery_gap = float(full["delivery_shortfall_from_maximum_mwh"].max())
    _require(delivery_gap <= study["recourse"]["delivery_tolerance_mwh"],
             f"12:00 delivery-first gap exceeds tolerance: {delivery_gap}")
    return {"maximum_fallback_share": maximum_fallback,
            "maximum_terminal_soc_deviation_mwh": terminal,
            "maximum_gross_power_mw": gross,
            "maximum_delivery_shortfall_from_optimum_mwh": delivery_gap}


def _decision_value(paired: pd.DataFrame, arm_summary: pd.DataFrame) -> dict:
    reference = float(arm_summary.loc[
        (arm_summary["strategy"] == "B2_FB0") & arm_summary["chi"].eq(0.0),
        "mean_daily_eur"].iloc[0])
    result = {}
    for chi in (0.0, 0.5):
        selected = paired.loc[(paired["strategy"] == "B3_hist_paired")
                              & paired["chi"].eq(chi)
                              & (paired["comparator"] == "B2_FB0")]
        profit = selected.loc[selected["metric"] == "total_eur"].iloc[0]
        undelivered = selected.loc[
            selected["metric"] == "undelivered_activation_share"].iloc[0]
        valuable = (profit["confidence_lower"] > 0.0
                    or (undelivered["confidence_upper"] < 0.0
                        and profit["confidence_lower"] > -0.01 * reference))
        result[str(chi)] = {"conclusion": "distribution_has_decision_value" if valuable
                            else "distribution_has_no_decision_value",
                            "profit_ci_lower_eur_per_day": profit["confidence_lower"],
                            "undelivered_share_ci_upper": undelivered["confidence_upper"]}
    return result


def _setup_full(path: Path) -> tuple:
    study, forecast, storage_source = load_configs(path)
    panels = build_panels(ROOT / storage_source["data"]["root"], storage_source)
    days, threshold = settlement_protocol(panels, storage_source)
    loaded = load_common_scenarios(ROOT, study, forecast, days)
    x14 = _load_x14(study, days)
    state = {"study": study, "forecast": forecast, "storage_source": storage_source,
             "panels": panels, "days": days, "threshold": threshold,
             "archives": _archive_index(loaded, x14), "points": _load_points(study),
             "premium": load_premium_history(ROOT, study, forecast, storage_source),
             "arm": storage_source["storage"]["arms"][study["storage"]["arm"]],
             "storage": storage_settings(storage_source)}
    return study, forecast, storage_source, state


def _write_outputs(study: dict, daily: pd.DataFrame, risk: pd.DataFrame,
                   cross: pd.DataFrame) -> dict:
    directory = ROOT / study["output"]["directory"]
    computed = _arm_summary(daily)
    computed["evidence_role"] = "same_protocol_computation"
    paired = _paired_differences(daily, computed, study)
    frames = {study["output"]["daily_results_csv"]: daily,
              study["output"]["arm_summary_csv"]: pd.concat(
                  [computed, _reference_rows(study)], ignore_index=True),
              study["output"]["paired_differences_csv"]: paired,
              study["output"]["frontier_csv"]: _frontier(computed),
              study["output"]["monthly_csv"]: _monthly(daily),
              study["output"]["stress_hours_csv"]: _stress_summary(daily),
              study["output"]["risk_calibration_csv"]: aggregate_calibration(risk),
              study["output"]["cross_scoring_csv"]: _cross_summary(cross, study),
              study["output"]["rq4_rank_agreement_csv"]: _rq4(computed)}
    for name, frame in frames.items():
        _write_frame(frame, directory / name)
    return {"computed": computed, "paired": paired, "frames": frames}


def _summary_payload(path: Path, study: dict, state: dict, checks: dict,
                     validation: dict, written: dict) -> dict:
    directory = ROOT / study["output"]["directory"]
    old = json.loads((directory / study["output"]["summary_json"]).read_text())
    computed, paired = written["computed"], written["paired"]
    published = computed.loc[computed["strategy"].isin(
        ["X14_rebuild", "X09_rebuild", "X06_rebuild"])]
    rebuilt = [{"strategy": row.strategy, "chi": float(row.chi),
                "total_eur": float(row.total_eur),
                "undelivered_activation_share": float(row.undelivered_activation_share)}
               for row in published.itertuples()]
    skipped = [{"strategy": name.upper() + "_rebuild",
                "reason": study["published_rules"][name]["skip_reason"]}
               for name in ("x07", "x13")]
    config_path = ROOT / path
    return {"stage": "S3-B complete", "environment": environment_fingerprint(),
            "seeds": {"scenario_sampling": study["export"]["sampling_seed"],
                      "premium_sampling": study["recourse"]["premium_sampling_seed"],
                      "clustering": study["tree"]["seed"],
                      "bootstrap": study["evaluation"]["bootstrap_seed"]},
            "test_days": len(state["days"]),
            "test_day_sha256": sha256("\n".join(state["days"]).encode()).hexdigest(),
            "comparison_config_sha256": sha256(config_path.read_bytes()).hexdigest(),
            "source_config_sha256": sha256(
                (ROOT / study["export"]["source_config"]).read_bytes()).hexdigest(),
            "export_generated_with_comparison_config_sha256": sha256(
                config_path.read_bytes()).hexdigest(),
            "export_checks": old["export_checks"], "formal_checks": checks,
            "validation": {**validation,
                           "repeat_run_hash_check": "skipped_by_user_decision_2026-09-15"},
            "decision_value": _decision_value(paired, computed),
            "published_rebuilds": rebuilt, "skipped_arms": skipped,
            "published_rule_deviations": {
                name: study["published_rules"][name]["deviation"]
                for name in ("x06", "x07", "x09", "x13", "x14")},
            "evidence_boundary": {
                "official_code_or_same_protocol_rebuild": "not_found",
                "errata": "not_found_not_proof_of_absence",
                "paid_pdfs_committed": False},
            "dst_handling": ("Forecast and scenario vectors stay at 24 hours; 23-hour days "
                             "truncate the last hour and 25-hour days repeat the last hour. "
                             "Realised settlement retains 23, 24, or 25 civil hours."),
            "file_sha256": _output_hashes(directory)}


def run_full(path: Path) -> dict:
    checks = run_checks(path)
    study, _, _, state = _setup_full(path)
    directory = ROOT / study["output"]["directory"]
    WORKER_STATE.clear()
    WORKER_STATE.update(state)
    context = mp.get_context("spawn")
    with context.Pool(int(study["runtime"]["worker_processes"]),
                      initializer=_init_full_worker, initargs=(state,)) as pool:
        batches = []
        for completed, batch in enumerate(
                pool.imap(_day_worker, enumerate(state["days"])), start=1):
            batches.append(batch)
            if completed % 10 == 0 or completed == len(state["days"]):
                print(json.dumps({"completed_days": completed,
                                  "total_days": len(state["days"])}), flush=True)
    daily = pd.DataFrame([row for batch in batches for row in batch["daily"]])
    risk = pd.DataFrame([row for batch in batches for row in batch["risk"]])
    cross = pd.DataFrame([row for batch in batches for row in batch["cross"]])
    daily = daily.sort_values(["delivery_day", "strategy", "chi"]).reset_index(drop=True)
    risk = risk.sort_values(["delivery_day", "strategy", "chi"]).reset_index(drop=True)
    cross = cross.sort_values(
        ["delivery_day", "model_strategy", "scenario_strategy"]).reset_index(drop=True)
    validation = _validate_full(daily, study, state["arm"], state["storage"])
    written = _write_outputs(study, daily, risk, cross)
    payload = _summary_payload(path, study, state, checks, validation, written)
    (directory / study["output"]["summary_json"]).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"days": len(state["days"]), "daily_rows": len(daily),
            "risk_rows": len(risk), "cross_score_rows": len(cross),
            "validation": validation, "decision_value": payload["decision_value"]}


def main() -> None:
    arguments = parse_arguments()
    _require(not (arguments.export and arguments.check), "Choose at most one mode")
    if arguments.export:
        result = run_export(arguments.config)
    elif arguments.check:
        result = run_checks(arguments.config)
    else:
        result = run_full(arguments.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
