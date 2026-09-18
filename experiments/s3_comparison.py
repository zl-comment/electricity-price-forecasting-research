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
                                     settle_day, settlement_trace, soc_matrix, solve_day,
                                     solve_recovery, threshold_rule)
from epf_harness.dk1_audit import load_table
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
from s3_forecast_side.panel import decision_time_utc, window_days


WORKER_STATE = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--proxy-analysis", action="store_true")
    parser.add_argument("--proxy-decision", action="store_true")
    return parser.parse_args()


def load_configs(path: Path) -> tuple:
    study = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    forecast = yaml.safe_load((ROOT / study["base_config"]).read_text(encoding="utf-8"))
    storage = yaml.safe_load((ROOT / study["storage_config"]).read_text(encoding="utf-8"))
    _require(study["recourse"]["delivery_0730"] == "full", "07:30 recourse is not frozen")
    _require(study["recourse"]["delivery_1200"] == "delivery_first",
             "12:00 recourse is not frozen")
    proxy = study["midday_evening_analysis"]["proxy_aware_decision"]
    credits = [float(item["alignment_credit_eur_per_mwh"])
               for item in proxy["strategies"]]
    names = [item["strategy"] for item in proxy["strategies"]]
    _require(len(credits) == len(set(credits)) and min(credits) > 0.0,
             "Proxy-aware credits must be unique and positive")
    _require(len(names) == len(set(names)), "Proxy-aware strategy names are duplicated")
    _require(float(proxy["risk_weight"]) in [float(value) for value in study["risk"]["chi_levels"]],
             "Proxy-aware risk weight is outside the evaluated levels")
    base = [item for item in _b3_specs() if item[0] == "B3_pooled_gaussian"]
    _require(len(base) == 1, "Proxy-aware market-only control is missing")
    _require(tuple(base[0][1:]) == (proxy["early_scenario_source"],
                                    proxy["late_scenario_source"],
                                    proxy["delivery_mode"]),
             "Proxy-aware arms do not share the market-only control protocol")
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
        _require(path.is_file(), f"{name}: source artifact is missing")
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
    _require(all(wave0["export_checks"].values()), "Wave-0 export checks contain a failure")
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


def _trace_file(strategy: str, chi: float) -> str:
    specifications = WORKER_STATE["study"]["midday_evening_analysis"]["trace_arms"]
    matches = [item["file"] for item in specifications
               if item["strategy"] == strategy and float(item["chi"]) == float(chi)]
    _require(len(matches) <= 1, f"{strategy}/{chi}: duplicate trace specifications")
    return matches[0] if matches else None


def _trace_payload(day: str, strategy: str, chi: float, plan: dict,
                   panel: dict, closed: dict, fallback: bool) -> dict:
    trace = settlement_trace(plan, panel, closed, WORKER_STATE["arm"], WORKER_STATE["storage"])
    trace.update({"delivery_day": day, "strategy": strategy, "chi": float(chi),
                  "fallback": int(fallback), "timestamp_utc": panel["timestamp_utc"],
                  "timestamp_local": panel["timestamp_local"],
                  "local_hour": panel["local_hour"]})
    return trace


def _daily_record(day: str, strategy: str, chi: float, plan: dict,
                  panel: dict, fallback: bool, early_status: int,
                  delivery_status: int, profit_status: int,
                  delivery_gap: float) -> dict:
    state = WORKER_STATE
    closed = (solve_recovery(plan, panel, state["arm"], state["storage"],
                             state["storage_source"]["solver"])
              if "p_recovery" not in plan else
              {"p_recovery": plan["p_recovery"],
               "required": plan["r_up"][panel["hour_of_slot"]] * panel["activation_share"],
               "delivered": plan["r_up"][panel["hour_of_slot"]] * panel["activation_share"],
               "soc_trace": plan["soc_plan"]})
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
    record = {"delivery_day": day, "strategy": strategy, "chi": float(chi),
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
              "undelivered_activation_share": (
                  float(undelivered / required) if required > 0 else 0.0),
              "stress_hours": int(stress_hours.sum()),
              "stress_required_activation_mwh": stress_required,
              "stress_delivered_activation_mwh": stress_delivered,
              **settled}
    if "forecast_alignment_mwh" in plan:
        record.update({
            "forecast_alignment_mwh": float(plan["forecast_alignment_mwh"]),
            "alignment_credit_eur": float(plan["alignment_credit_eur"]),
            "proxy_alignment_credit_eur_per_mwh": float(
                plan["proxy_alignment_credit_eur_per_mwh"]),
        })
    else:
        record.update({"forecast_alignment_mwh": np.nan,
                       "alignment_credit_eur": np.nan,
                       "proxy_alignment_credit_eur_per_mwh": np.nan})
    if _trace_file(strategy, chi) is not None:
        record["_trace"] = _trace_payload(
            day, strategy, chi, plan, panel, closed, fallback)
    return record


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
              late_name: str, mode: str, chi: float,
              proxy_forecasts=None, alignment_credit: float = 0.0) -> tuple:
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
    early_proxy = None if proxy_forecasts is None else proxy_forecasts["gate_0730"]
    early = solve_stochastic(
        early_scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
        state["arm"], state["storage"], study["solver"], study["risk"], chi, None,
        mode, "profit", None, early_proxy, alignment_credit)
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
    late_proxy = None if proxy_forecasts is None else proxy_forecasts["gate_1200"]
    late = _late_solve(late_scenarios, late_probabilities, late_labels, panel, early,
                       mode, chi, late_proxy, alignment_credit)
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
                panel: dict, early: dict, mode: str, chi: float,
                proxy_forecast_mwh=None, alignment_credit: float = 0.0) -> dict:
    state, study = WORKER_STATE, WORKER_STATE["study"]
    if mode == study["recourse"]["shortfall_sensitivity"]:
        return solve_stochastic(
            scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
            state["arm"], state["storage"], study["solver"], study["risk"], chi,
            early["r_up"], mode, "profit", None,
            proxy_forecast_mwh, alignment_credit)
    return solve_delivery_first(
        scenarios, probabilities, labels, panel["procured_mw"], panel["hour_of_slot"],
        state["arm"], state["storage"], study["solver"], study["risk"], chi,
        early["r_up"], study["recourse"]["delivery_tolerance_mwh"],
        proxy_forecast_mwh, alignment_credit)


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
    proxy_settings = WORKER_STATE["study"]["midday_evening_analysis"][
        "proxy_aware_decision"]
    if day in WORKER_STATE["proxy_decision_signals"]:
        for specification in proxy_settings["strategies"]:
            row, risk, _ = _solve_b3(
                day, index, specification["strategy"],
                proxy_settings["early_scenario_source"],
                proxy_settings["late_scenario_source"],
                proxy_settings["delivery_mode"], float(proxy_settings["risk_weight"]),
                WORKER_STATE["proxy_decision_signals"][day],
                float(specification["alignment_credit_eur_per_mwh"]))
            _require(row is not None,
                     f"{day}/{specification['strategy']}: proxy-aware solve day is missing")
            rows.append(row)
            if risk is not None:
                risks.append(risk)
    for chi in WORKER_STATE["study"]["risk"]["x14_chi_levels"]:
        row, risk = _solve_x14(day, index, float(chi))
        rows.append(row)
        if risk is not None:
            risks.append(risk)
    rows.extend([_solve_x09(day, index), _solve_x06(day, index)])
    _require(set(cross_artifacts) == _cross_source_names(),
             f"{day}: cross-score arm set is incomplete")
    return {"daily": rows, "risk": risks, "cross": _cross_rows(day, cross_artifacts)}


def _proxy_day_worker(item: tuple) -> dict:
    index, day = item
    settings = WORKER_STATE["study"]["midday_evening_analysis"]["proxy_aware_decision"]
    rows, risks = [], []
    for specification in settings["strategies"]:
        row, risk, _ = _solve_b3(
            day, index, specification["strategy"], settings["early_scenario_source"],
            settings["late_scenario_source"], settings["delivery_mode"],
            float(settings["risk_weight"]), WORKER_STATE["proxy_decision_signals"][day],
            float(specification["alignment_credit_eur_per_mwh"]))
        _require(row is not None,
                 f"{day}/{specification['strategy']}: proxy-aware solve day is missing")
        rows.append(row)
        if risk is not None:
            risks.append(risk)
    return {"daily": rows, "risk": risks, "cross": []}


def _arm_summary(daily: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (strategy, chi), frame in daily.groupby(["strategy", "chi"], sort=True):
        required = float(frame["required_activation_mwh"].sum())
        undelivered = float(frame["undelivered_activation_mwh"].sum())
        credits = frame["proxy_alignment_credit_eur_per_mwh"].dropna().unique()
        _require(len(credits) <= 1, f"{strategy}/{chi}: multiple proxy-alignment credits")
        credit = float(credits[0]) if len(credits) == 1 else np.nan
        tail = empirical_tail_metrics(frame["total_eur"].to_numpy() / 1000.0, 0.05)
        records.append({"strategy": strategy, "chi": chi, "days": len(frame),
                        "proxy_alignment_credit_eur_per_mwh": credit,
                        "mean_forecast_alignment_mwh": float(
                            frame["forecast_alignment_mwh"].mean()),
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
    difference = (joined[f"{metric}_left"] - joined[f"{metric}_right"]).to_numpy(dtype=float)
    if np.ptp(difference) == 0.0:
        return {"strategy": left[0], "chi": left[1], "comparator": right[0],
                "comparator_chi": right[1], "metric": metric, "status": "constant_difference",
                "mean_difference_strategy_minus_comparator": float(difference[0]),
                "confidence_lower": float(difference[0]), "confidence_upper": float(difference[0]),
                "dm_statistic": np.nan, "p_value": np.nan, "n_days": len(difference)}
    result = paired_metrics(joined[f"{metric}_left"], joined[f"{metric}_right"], study)
    return {"strategy": left[0], "chi": left[1], "comparator": right[0],
            "comparator_chi": right[1], "metric": metric, "status": "computed",
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
    first = first.loc[first["seed"] == 1]
    second = pd.read_csv(ROOT / "p1_paper/results/s3_cross_market/daily_joint_scores.csv")
    cross_summary = json.loads((ROOT / "p1_paper/results/s3_cross_market/summary.json").read_text())
    _require(cross_summary["sampling_seed"] == 1, "s3_cross_market joint scores are not seed 1")
    mapping = {"B3_hist_paired": (first, "hist_paired"),
               "B3_hist_independent": (first, "hist_independent"),
               "B3_pooled_independent": (second, "own_pooled__independent"),
               "B3_pooled_gaussian_no_cross": (first, "fb2_no_cross_target"),
               "B3_pooled_gaussian": (second, "own_pooled__gaussian"),
               "B3_pooled_empirical": (second, "own_pooled__empirical")}
    rows = []
    for strategy, (frame, arm) in mapping.items():
        selected = frame.loc[(frame["arm"] == arm) & (frame["block"] == "all")]
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
                             "kendall_tau": float(test.correlation),
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
    frame.to_csv(path, index=False, float_format="%.12g", line_terminator="\n")


def _expected_civil_hours(day: str, timezone: str) -> int:
    start = pd.Timestamp(day).tz_localize(timezone)
    end = (pd.Timestamp(day) + pd.DateOffset(days=1)).tz_localize(timezone)
    return int((end.tz_convert("UTC") - start.tz_convert("UTC")).total_seconds() / 3600)


def _proxy_hourly(study: dict) -> tuple:
    settings = study["midday_evening_analysis"]
    proxy_config = yaml.safe_load((ROOT / settings["proxy_config"]).read_text(encoding="utf-8"))
    dataset = proxy_config["datasets"]["production_consumption_settlement"]
    root = ROOT / proxy_config["data"]["root"]
    frame = load_table(root, dataset)
    frame = frame.loc[frame["PriceArea"].eq(proxy_config["data"]["primary_price_area"])].copy()
    utc = frame[dataset["time_column"]].dt.tz_localize("UTC")
    local = utc.dt.tz_convert(proxy_config["protocol"]["civil_timezone"])
    frame["delivery_day"] = local.dt.strftime("%Y-%m-%d")
    window = proxy_config["window"]
    frame = frame.loc[frame["delivery_day"].between(
        window["first_delivery_day"], window["last_delivery_day"])].copy()
    counts = frame.groupby("delivery_day").size().to_dict()
    calendar = pd.date_range(window["first_delivery_day"], window["last_delivery_day"], freq="D")
    timezone = proxy_config["protocol"]["civil_timezone"]
    expected = {day.strftime("%Y-%m-%d"): _expected_civil_hours(day.strftime("%Y-%m-%d"), timezone)
                for day in calendar}
    complete = sorted(day for day, count in counts.items() if count == expected[day])
    incomplete = {day: {"observed_hours": int(counts.get(day, 0)),
                        "expected_hours": int(expected[day])}
                  for day in expected if counts.get(day, 0) != expected[day]}
    frame = frame.loc[frame["delivery_day"].isin(complete)].copy()
    columns = proxy_config["net_load"]
    wind = frame[columns["wind_columns"]].sum(axis=1, min_count=len(columns["wind_columns"]))
    solar = frame[columns["solar_columns"]].sum(axis=1, min_count=len(columns["solar_columns"]))
    load = frame[columns["consumption_column"]]
    _require(pd.concat([wind, solar, load], axis=1).notna().all().all(),
             "A complete proxy day contains a missing VRE or load observation")
    local = frame[dataset["time_column"]].dt.tz_localize("UTC").dt.tz_convert(timezone)
    result = pd.DataFrame({
        "timestamp_utc": frame[dataset["time_column"]].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_local": local.astype(str), "delivery_day": frame["delivery_day"],
        "month": local.dt.strftime("%Y-%m"), "local_hour": local.dt.hour.astype(int),
        "wind_mwh": wind, "solar_mwh": solar, "vre_mwh": wind + solar,
        "gross_consumption_mwh": load,
        "residual_load_proxy_mwh": load - wind - solar,
    })
    result["proxy_surplus_mwh"] = result["residual_load_proxy_mwh"].clip(upper=0).abs()
    result["proxy_positive"] = result["proxy_surplus_mwh"].gt(0).astype(int)
    result = result.sort_values("timestamp_utc").reset_index(drop=True)
    coverage = {"complete_days": len(complete), "complete_hours": len(result),
                "first_complete_day": complete[0], "last_complete_day": complete[-1],
                "incomplete_days": incomplete}
    return result, coverage


def _proxy_group_summary(frame: pd.DataFrame, keys: list) -> pd.DataFrame:
    rows = []
    for labels, group in frame.groupby(keys, sort=True):
        labels = labels if isinstance(labels, tuple) else (labels,)
        positive = group["proxy_positive"].eq(1)
        row = dict(zip(keys, labels))
        row.update({"valid_hours": len(group), "positive_hours": int(positive.sum()),
                    "positive_hour_share": float(positive.mean()),
                    "proxy_surplus_mwh": float(group["proxy_surplus_mwh"].sum()),
                    "observed_days": int(group["delivery_day"].nunique()),
                    "days_with_proxy_surplus": int(
                        group.loc[positive, "delivery_day"].nunique())})
        rows.append(row)
    return pd.DataFrame(rows)


def _proxy_outputs(frame: pd.DataFrame, study: dict) -> tuple:
    monthly = _proxy_group_summary(frame, ["month"])
    clock = _proxy_group_summary(frame, ["local_hour"])
    month_clock = _proxy_group_summary(frame, ["month", "local_hour"])
    settings = study["midday_evening_analysis"]
    midday = frame["local_hour"].between(
        int(settings["midday_start_hour"]), int(settings["midday_end_hour_exclusive"]) - 1)
    positive = frame["proxy_positive"].eq(1)
    summary = {
        "positive_hours": int(positive.sum()),
        "positive_hour_share": float(positive.mean()),
        "proxy_surplus_mwh": float(frame["proxy_surplus_mwh"].sum()),
        "days_with_proxy_surplus": int(frame.loc[positive, "delivery_day"].nunique()),
        "midday_valid_hours": int(midday.sum()),
        "midday_positive_hours": int((midday & positive).sum()),
        "midday_positive_hour_share": float(frame.loc[midday, "proxy_positive"].mean()),
        "midday_proxy_surplus_mwh": float(frame.loc[midday, "proxy_surplus_mwh"].sum()),
    }
    peak_month = monthly.sort_values(
        ["positive_hour_share", "proxy_surplus_mwh"], ascending=False).iloc[0]
    peak_hour = clock.sort_values(
        ["positive_hour_share", "proxy_surplus_mwh"], ascending=False).iloc[0]
    summary.update({"peak_month": str(peak_month["month"]),
                    "peak_month_positive_hour_share": float(
                        peak_month["positive_hour_share"]),
                    "peak_clock_hour": int(peak_hour["local_hour"]),
                    "peak_clock_hour_positive_share": float(
                        peak_hour["positive_hour_share"])})
    return monthly, clock, month_clock, summary


def _proxy_decomposition(proxy: pd.DataFrame, study: dict) -> tuple:
    settings = study["midday_evening_analysis"]
    source = yaml.safe_load((ROOT / settings["proxy_config"]).read_text(encoding="utf-8"))
    dataset = source["datasets"]["production_consumption_settlement"]
    raw = load_table(ROOT / source["data"]["root"], dataset)
    raw = raw.loc[raw["PriceArea"].eq(source["data"]["primary_price_area"])].copy()
    raw["timestamp_utc"] = raw[dataset["time_column"]].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    fields = settings["proxy_decomposition"]
    required = fields["exchange_columns"] + fields["conventional_columns"]
    required.append(fields["power_to_heat_column"])
    _require(raw[required].notna().all().all(), "Proxy decomposition inputs contain missing values")
    raw["net_exchange_mwh"] = raw[fields["exchange_columns"]].sum(axis=1)
    raw["net_export_mwh"] = (-raw["net_exchange_mwh"]).clip(lower=0)
    raw["net_import_mwh"] = raw["net_exchange_mwh"].clip(lower=0)
    raw["conventional_generation_mwh"] = raw[fields["conventional_columns"]].sum(axis=1)
    raw["power_to_heat_mwh"] = raw[fields["power_to_heat_column"]]
    columns = ["timestamp_utc", "net_exchange_mwh", "net_export_mwh", "net_import_mwh",
               "conventional_generation_mwh", "power_to_heat_mwh"]
    result = proxy.merge(raw[columns], on="timestamp_utc", validate="one_to_one")
    return _finish_proxy_decomposition(result, fields)


def _finish_proxy_decomposition(frame: pd.DataFrame, settings: dict) -> tuple:
    frame["balance_residual_mwh"] = (
        frame["vre_mwh"] + frame["conventional_generation_mwh"]
        + frame["net_exchange_mwh"] - frame["gross_consumption_mwh"])
    frame["proxy_from_balance_mwh"] = (
        frame["net_export_mwh"] - frame["net_import_mwh"]
        - frame["conventional_generation_mwh"] + frame["balance_residual_mwh"]).clip(lower=0)
    frame["export_coverage_mwh"] = frame[["proxy_surplus_mwh", "net_export_mwh"]].min(axis=1)
    frame["proxy_after_export_mwh"] = (
        frame["proxy_surplus_mwh"] - frame["net_export_mwh"]).clip(lower=0)
    frame["proxy_without_power_to_heat_mwh"] = (
        frame["vre_mwh"] - (frame["gross_consumption_mwh"]
                            - frame["power_to_heat_mwh"])).clip(lower=0)
    frame["power_to_heat_absorption_mwh"] = (
        frame["proxy_without_power_to_heat_mwh"] - frame["proxy_surplus_mwh"])
    tolerance = float(settings["balance_tolerance_mwh"])
    _require(frame["balance_residual_mwh"].abs().max() <= tolerance,
             "Observed production-exchange-consumption balance exceeds tolerance")
    _require((frame["proxy_from_balance_mwh"] - frame["proxy_surplus_mwh"]).abs().max()
             <= tolerance, "Proxy decomposition identity exceeds tolerance")
    monthly = _proxy_decomposition_groups(frame, ["month"])
    clock = _proxy_decomposition_groups(frame, ["local_hour"])
    summary = _proxy_decomposition_summary(frame)
    return frame, monthly, clock, summary


def _proxy_decomposition_groups(frame: pd.DataFrame, keys: list) -> pd.DataFrame:
    rows = []
    for labels, group in frame.groupby(keys, sort=True):
        labels = labels if isinstance(labels, tuple) else (labels,)
        row = dict(zip(keys, labels))
        row.update(_proxy_decomposition_summary(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _proxy_decomposition_summary(frame: pd.DataFrame) -> dict:
    positive = frame.loc[frame["proxy_positive"].eq(1)]
    proxy = float(positive["proxy_surplus_mwh"].sum())
    export = float(positive["net_export_mwh"].sum())
    no_p2h = float(positive["proxy_without_power_to_heat_mwh"].sum())
    absorbed = float(positive["power_to_heat_absorption_mwh"].sum())
    return {
        "valid_hours": len(frame), "proxy_positive_hours": len(positive),
        "proxy_positive_mwh": proxy,
        "net_export_mwh_on_positive_hours": export,
        "conventional_mwh_on_positive_hours": float(
            positive["conventional_generation_mwh"].sum()),
        "power_to_heat_mwh_on_positive_hours": float(positive["power_to_heat_mwh"].sum()),
        "export_coverage_share": float(positive["export_coverage_mwh"].sum() / proxy),
        "conventional_share_of_export": float(
            positive["conventional_generation_mwh"].sum() / export),
        "power_to_heat_counterfactual_share": float(absorbed / no_p2h),
        "maximum_absolute_balance_residual_mwh": float(
            frame["balance_residual_mwh"].abs().max()),
    }


def _gate_weather(study: dict, forecast: dict) -> pd.DataFrame:
    settings = study["midday_evening_analysis"]["gate_proxy_forecast"]
    weather = pd.read_csv(ROOT / settings["weather_csv"])
    for column in ("run_time_utc", "available_at_utc", "timestamp_utc"):
        weather[column] = pd.to_datetime(weather[column], utc=True)
    _require(weather["source_grid_points"].eq(settings["expected_grid_points"]).all(),
             "Gate weather has an unexpected spatial-grid size")
    _require(not weather.duplicated(["gate", "timestamp_utc"]).any(),
             "Gate weather has duplicate gate/time keys")
    decision = weather.apply(
        lambda row: decision_time_utc(row["delivery_day"], row["gate"], forecast), axis=1)
    weather["decision_time_utc"] = pd.to_datetime(decision, utc=True)
    _require(weather["available_at_utc"].le(weather["decision_time_utc"]).all(),
             "Gate weather contains a forecast unavailable at decision time")
    latency = (weather["available_at_utc"] - weather["run_time_utc"]).dt.total_seconds() / 3600
    _require(latency.eq(settings["maximum_publication_latency_hours"]).all(),
             "Gate weather uses an unexpected publication-latency assumption")
    return weather


def _forecast_base(proxy: pd.DataFrame, weather: pd.DataFrame, settings: dict) -> pd.DataFrame:
    actual_all = proxy.copy()
    actual_all["timestamp_utc"] = pd.to_datetime(actual_all["timestamp_utc"], utc=True)
    counts = actual_all.groupby("delivery_day").size()
    days = counts.loc[counts.eq(24)].index
    actual = actual_all.loc[actual_all["delivery_day"].isin(days)].copy()
    columns = ["timestamp_utc", "delivery_day", "month", "local_hour"] + settings["targets"]
    columns += ["proxy_surplus_mwh", "proxy_positive"]
    base = weather.merge(actual[columns], on=["timestamp_utc", "delivery_day"],
                         validate="many_to_one")
    unique_lag = ~actual_all.duplicated(["delivery_day", "local_hour"], keep=False)
    lag_source = actual_all.loc[unique_lag]
    for target in settings["targets"]:
        for lag in settings["lag_days"]:
            lookup = lag_source[["delivery_day", "local_hour", "timestamp_utc", target]].copy()
            lookup["delivery_day"] = (
                pd.to_datetime(lookup["delivery_day"]) + pd.DateOffset(days=lag)).dt.strftime("%Y-%m-%d")
            availability = f"{target}_lag_{lag}_available_at"
            lookup["timestamp_utc"] += pd.Timedelta(
                minutes=settings["settlement_observation_lag_minutes"])
            lookup = lookup.rename(columns={target: f"{target}_lag_{lag}",
                                            "timestamp_utc": availability})
            base = base.merge(lookup, on=["delivery_day", "local_hour"], how="left",
                              validate="many_to_one")
            visible = base[f"{target}_lag_{lag}"].notna()
            _require(base.loc[visible, availability].le(
                base.loc[visible, "decision_time_utc"]).all(),
                f"{target} lag {lag} is not visible at the decision gate")
    return base.sort_values(["gate", "timestamp_utc"]).reset_index(drop=True)


def _proxy_design(frame: pd.DataFrame, target: str, settings: dict) -> np.ndarray:
    hour = frame["local_hour"].to_numpy(dtype=float)
    columns = [np.ones(len(frame))]
    for harmonic in range(1, int(settings["fourier_harmonics"]) + 1):
        angle = 2.0 * np.pi * harmonic * hour / 24.0
        columns.extend([np.sin(angle), np.cos(angle)])
    weekday = pd.to_datetime(frame["delivery_day"]).dt.dayofweek.to_numpy()
    columns.extend([(weekday == value).astype(float) for value in range(1, 7)])
    columns.extend(frame[name].to_numpy(dtype=float) for name in settings["weather_columns"])
    temperature = frame[settings["temperature_column"]].to_numpy(dtype=float)
    wind = frame[settings["wind_speed_column"]].to_numpy(dtype=float)
    columns.extend([temperature ** 2, wind ** 2, wind ** 3])
    columns.extend(frame[f"{target}_lag_{lag}"].to_numpy(dtype=float)
                   for lag in settings["lag_days"])
    return np.column_stack(columns)


def _component_forecast(base: pd.DataFrame, target: str, day: str,
                        fit_days: list, settings: dict) -> pd.DataFrame:
    train = base.loc[base["delivery_day"].isin(fit_days)].copy()
    test = base.loc[base["delivery_day"].eq(day)].copy()
    input_columns = settings["weather_columns"] + [
        f"{target}_lag_{lag}" for lag in settings["lag_days"]]
    train = train.dropna(subset=input_columns + [target])
    test = test.dropna(subset=input_columns)
    _require(len(train) >= int(settings["minimum_training_rows"]),
             f"{day}: insufficient component-forecast training rows")
    design_train = _proxy_design(train, target, settings)
    design_test = _proxy_design(test, target, settings)
    coefficient = np.linalg.lstsq(
        design_train, train[target].to_numpy(dtype=float), rcond=None)[0]
    name = settings["target_forecast_columns"][target]
    result = test[["timestamp_utc"]].copy()
    result[name] = np.maximum(design_test @ coefficient, 0.0)
    return result


def _forecast_gate_day(base: pd.DataFrame, gate: str, day: str,
                       forecast: dict, settings: dict) -> pd.DataFrame:
    gate_base = base.loc[base["gate"].eq(gate)]
    fit_days = window_days(day, forecast)["fit"]
    components = [_component_forecast(gate_base, target, day, fit_days, settings)
                  for target in settings["targets"]]
    result = gate_base.loc[gate_base["delivery_day"].eq(day)].copy()
    for component in components:
        result = result.merge(component, on="timestamp_utc", validate="one_to_one")
    result["forecast_vre_mwh"] = result["forecast_wind_mwh"] + result["forecast_solar_mwh"]
    result["forecast_proxy_surplus_mwh"] = (
        result["forecast_vre_mwh"] - result["forecast_gross_consumption_mwh"]).clip(lower=0)
    result["forecast_proxy_positive"] = result["forecast_proxy_surplus_mwh"].gt(0).astype(int)
    return result


def _forecast_metric_row(group: pd.DataFrame, gate: str, scope: str) -> dict:
    error = group["forecast_proxy_surplus_mwh"] - group["proxy_surplus_mwh"]
    observed = group["proxy_positive"].eq(1)
    predicted = group["forecast_proxy_positive"].eq(1)
    true_positive = int((observed & predicted).sum())
    return {
        "gate": gate, "scope": scope, "hours": len(group),
        "wind_mae_mwh": float((group["forecast_wind_mwh"] - group["wind_mwh"]).abs().mean()),
        "solar_mae_mwh": float((group["forecast_solar_mwh"] - group["solar_mwh"]).abs().mean()),
        "load_mae_mwh": float((group["forecast_gross_consumption_mwh"]
                                - group["gross_consumption_mwh"]).abs().mean()),
        "proxy_mae_mwh": float(error.abs().mean()),
        "proxy_rmse_mwh": float(np.sqrt(np.mean(error ** 2))),
        "observed_positive_hours": int(observed.sum()),
        "forecast_positive_hours": int(predicted.sum()),
        "positive_precision": float(true_positive / predicted.sum()) if predicted.any() else np.nan,
        "positive_recall": float(true_positive / observed.sum()) if observed.any() else np.nan,
        "observed_proxy_mwh": float(group["proxy_surplus_mwh"].sum()),
        "forecast_proxy_mwh": float(group["forecast_proxy_surplus_mwh"].sum()),
    }


def _gate_proxy_forecasts(proxy: pd.DataFrame, study: dict, forecast: dict,
                          test_days: list) -> tuple:
    settings = study["midday_evening_analysis"]["gate_proxy_forecast"]
    weather = _gate_weather(study, forecast)
    base = _forecast_base(proxy, weather, settings)
    rows = []
    for day in test_days:
        for gate in settings["gates"]:
            rows.append(_forecast_gate_day(base, gate, day, forecast, settings))
    hourly = pd.concat(rows, ignore_index=True).sort_values(
        ["gate", "timestamp_utc"]).reset_index(drop=True)
    summary_rows = []
    midday = study["midday_evening_analysis"]
    for gate, group in hourly.groupby("gate", sort=True):
        summary_rows.append(_forecast_metric_row(group, gate, "all_hours"))
        selected = group["local_hour"].between(
            midday["midday_start_hour"], midday["midday_end_hour_exclusive"] - 1)
        summary_rows.append(_forecast_metric_row(group.loc[selected], gate, "midday"))
    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame([_forecast_metric_row(group, gate, month)
                            for (gate, month), group in hourly.groupby(["gate", "month"])])
    audit = _forecast_revision_audit(hourly, weather, settings)
    return hourly, summary, monthly, audit


def _proxy_decision_signals(hourly: pd.DataFrame, study: dict) -> tuple:
    settings = study["midday_evening_analysis"]
    gates = settings["gate_proxy_forecast"]["gates"]
    expected_hours = set(range(24))
    signals, incomplete = {}, {}
    for day, day_frame in hourly.groupby("delivery_day", sort=True):
        gate_signals, reasons = {}, []
        for gate in gates:
            frame = day_frame.loc[day_frame["gate"].eq(gate)].sort_values("local_hour")
            hours = frame["local_hour"].astype(int)
            if len(frame) != 24 or set(hours) != expected_hours or hours.duplicated().any():
                reasons.append(f"{gate}:{len(frame)}_rows")
                continue
            values = frame["forecast_proxy_surplus_mwh"].to_numpy(dtype=float)
            _require(np.isfinite(values).all() and (values >= 0.0).all(),
                     f"{day}/{gate}: invalid proxy decision signal")
            midday = hours.between(
                int(settings["midday_start_hour"]),
                int(settings["midday_end_hour_exclusive"]) - 1).to_numpy()
            gate_signals[gate] = np.where(midday, values, 0.0)
        if len(gate_signals) == len(gates):
            signals[str(day)] = gate_signals
        else:
            incomplete[str(day)] = reasons
    return signals, {"complete_decision_days": len(signals),
                     "incomplete_decision_days": incomplete,
                     "gates": gates,
                     "midday_start_hour": int(settings["midday_start_hour"]),
                     "midday_end_hour_exclusive": int(
                         settings["midday_end_hour_exclusive"])}


def _forecast_revision_audit(hourly: pd.DataFrame, weather: pd.DataFrame,
                             settings: dict) -> dict:
    columns = ["timestamp_utc", "forecast_proxy_surplus_mwh", "forecast_proxy_positive"]
    early = hourly.loc[hourly["gate"].eq("gate_0730"), columns]
    late = hourly.loc[hourly["gate"].eq("gate_1200"), columns]
    joined = early.merge(late, on="timestamp_utc", suffixes=("_0730", "_1200"),
                         validate="one_to_one")
    margin = (weather["decision_time_utc"] - weather["available_at_utc"])
    return {
        "forecast_hours_per_gate": hourly.groupby("gate").size().astype(int).to_dict(),
        "forecast_delivery_days_per_gate": hourly.groupby("gate")[
            "delivery_day"].nunique().astype(int).to_dict(),
        "common_revision_hours": len(joined),
        "weather_rows_with_missing_predictors_by_gate": weather.assign(
            missing=weather[settings["weather_columns"]].isna().any(axis=1)).groupby(
                                 "gate")["missing"].sum().astype(int).to_dict(),
        "mean_absolute_0730_to_1200_proxy_revision_mwh": float((
            joined["forecast_proxy_surplus_mwh_1200"]
            - joined["forecast_proxy_surplus_mwh_0730"]).abs().mean()),
        "positive_classification_switches": int((
            joined["forecast_proxy_positive_0730"]
            != joined["forecast_proxy_positive_1200"]).sum()),
        "minimum_weather_visibility_margin_hours": float(margin.dt.total_seconds().min() / 3600),
    }


def _trace_rows(batches: list) -> tuple:
    daily_rows, traces = [], []
    for batch in batches:
        for source in batch["daily"]:
            row = dict(source)
            payload = row.pop("_trace", None)
            if payload is not None:
                traces.append(pd.DataFrame(payload))
            daily_rows.append(row)
    trace = pd.concat(traces, ignore_index=True)
    trace = trace.sort_values(["strategy", "chi", "timestamp_utc"]).reset_index(drop=True)
    return daily_rows, trace


def _trace_validation(trace: pd.DataFrame, daily: pd.DataFrame, study: dict) -> dict:
    tolerance = float(study["midday_evening_analysis"]["aggregation_tolerance"])
    keys = ["delivery_day", "strategy", "chi"]
    counts = trace.groupby(keys).size()
    _require(set(counts.unique()).issubset({92, 96, 100}),
             f"Unexpected 15-minute trace counts: {sorted(counts.unique())}")
    _require(not trace.duplicated(["strategy", "chi", "timestamp_utc"]).any(),
             "Dispatch trace has duplicate strategy/time keys")
    sums = trace.groupby(keys)[["energy_eur", "capacity_eur", "activation_eur",
                                "shortfall_cost_eur", "recovery_cost_eur", "total_eur",
                                "required_activation_mwh", "delivered_activation_mwh",
                                "undelivered_activation_mwh"]].sum().reset_index()
    sums["recovery_grid_mwh"] = (trace.assign(
        recovery_grid_mwh=0.25 * trace["recovery_power_mw"])
        .groupby(keys)["recovery_grid_mwh"].sum().to_numpy())
    terminal = trace.sort_values("timestamp_utc").groupby(keys).tail(1)
    terminal = terminal[keys + ["soc_realised_end_mwh"]].rename(
        columns={"soc_realised_end_mwh": "terminal_soc_mwh"})
    sums = sums.merge(terminal, on=keys, validate="one_to_one")
    selected = daily.merge(sums, on=keys, suffixes=("_daily", "_trace"), validate="one_to_one")
    fields = ["energy_eur", "capacity_eur", "activation_eur", "shortfall_cost_eur",
              "recovery_cost_eur", "total_eur", "required_activation_mwh",
              "delivered_activation_mwh", "undelivered_activation_mwh",
              "recovery_grid_mwh", "terminal_soc_mwh"]
    errors = {field: float((selected[f"{field}_daily"]
                            - selected[f"{field}_trace"]).abs().max()) for field in fields}
    _require(max(errors.values()) <= tolerance,
             f"Trace does not reproduce daily settlement: {errors}")
    return {"rows": len(trace), "arm_chi_pairs": int(trace.groupby(
        ["strategy", "chi"]).ngroups), "days_per_pair_min": int(counts.groupby(
            ["strategy", "chi"]).size().min()), "maximum_aggregation_error": max(errors.values()),
            "aggregation_error_by_field": errors}


def _hourly_charge_with_proxy(trace: pd.DataFrame, proxy: pd.DataFrame,
                              forecast_hourly: pd.DataFrame) -> pd.DataFrame:
    frame = trace.copy()
    frame["hour_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.floor("1H")
    rows = []
    keys = ["delivery_day", "strategy", "chi", "hour_utc"]
    for labels, group in frame.groupby(keys, sort=True):
        _require(len(group) == 4, f"{labels}: trace hour does not contain four slots")
        rows.append({**dict(zip(keys, labels)), "local_hour": int(group["local_hour"].iloc[0]),
                     "charge_mwh": float(0.25 * group["charge_mw"].sum())})
    hourly = pd.DataFrame(rows)
    proxy_fields = proxy[["timestamp_utc", "proxy_surplus_mwh", "proxy_positive"]].copy()
    proxy_fields["hour_utc"] = pd.to_datetime(proxy_fields["timestamp_utc"], utc=True)
    result = hourly.merge(proxy_fields.drop(columns="timestamp_utc"), on="hour_utc",
                          how="inner", validate="many_to_one")
    forecast = forecast_hourly.loc[forecast_hourly["gate"].eq("gate_1200"),
                                   ["timestamp_utc", "forecast_proxy_surplus_mwh"]].copy()
    forecast["hour_utc"] = pd.to_datetime(forecast["timestamp_utc"], utc=True)
    return result.merge(forecast.drop(columns="timestamp_utc"), on="hour_utc",
                        how="inner", validate="many_to_one")


def _soc_at_boundary(frame: pd.DataFrame, end_hour: int) -> float:
    selected = frame.loc[frame["local_hour"] < end_hour].sort_values("timestamp_utc")
    _require(len(selected) > 0, f"No trace interval precedes local hour {end_hour}")
    return float(selected["soc_realised_end_mwh"].iloc[-1])


def _midday_evening_daily(trace: pd.DataFrame, proxy: pd.DataFrame,
                          forecast_hourly: pd.DataFrame, decision_days: set,
                          study: dict) -> pd.DataFrame:
    settings = study["midday_evening_analysis"]
    trace = trace.loc[trace["delivery_day"].isin(decision_days)].copy()
    hourly = _hourly_charge_with_proxy(trace, proxy, forecast_hourly)
    rows = []
    for (day, strategy, chi), group in trace.groupby(
            ["delivery_day", "strategy", "chi"], sort=True):
        mid = group["local_hour"].between(
            int(settings["midday_start_hour"]), int(settings["midday_end_hour_exclusive"]) - 1)
        evening = group["local_hour"].between(
            int(settings["evening_start_hour"]), int(settings["evening_end_hour_exclusive"]) - 1)
        hour = hourly.loc[(hourly["delivery_day"] == day) & (hourly["strategy"] == strategy)
                          & hourly["chi"].eq(chi)]
        hour_mid = hour["local_hour"].between(
            int(settings["midday_start_hour"]), int(settings["midday_end_hour_exclusive"]) - 1)
        overlap = np.minimum(hour.loc[hour_mid, "charge_mwh"],
                             hour.loc[hour_mid, "proxy_surplus_mwh"])
        forecast_overlap = np.minimum(
            hour.loc[hour_mid, "charge_mwh"],
            hour.loc[hour_mid, "forecast_proxy_surplus_mwh"])
        required = float(group.loc[evening, "required_activation_mwh"].sum())
        delivered = float(group.loc[evening, "delivered_activation_mwh"].sum())
        charge = float(0.25 * group.loc[mid, "charge_mw"].sum())
        proxy_total = float(hour.loc[hour_mid, "proxy_surplus_mwh"].sum())
        forecast_proxy_total = float(
            hour.loc[hour_mid, "forecast_proxy_surplus_mwh"].sum())
        rows.append({"delivery_day": day, "strategy": strategy, "chi": float(chi),
                     "fallback": int(group["fallback"].max()),
                     "midday_charge_mwh": charge,
                     "midday_discharge_mwh": float(0.25 * group.loc[mid, "discharge_mw"].sum()),
                     "midday_negative_price_charge_mwh": float(
                         0.25 * group.loc[mid & group["day_ahead_price_eur_per_mwh"].lt(0),
                                          "charge_mw"].sum()),
                     "midday_charge_on_proxy_positive_hours_mwh": float(
                         hour.loc[hour_mid & hour["proxy_positive"].eq(1), "charge_mwh"].sum()),
                     "midday_proxy_overlap_mwh": float(overlap.sum()),
                     "midday_forecast_proxy_overlap_mwh": float(forecast_overlap.sum()),
                     "midday_proxy_surplus_mwh": proxy_total,
                     "midday_forecast_proxy_surplus_mwh": forecast_proxy_total,
                     "midday_alignment_share": float(overlap.sum() / charge) if charge > 0 else np.nan,
                     "midday_forecast_alignment_share": (
                         float(forecast_overlap.sum() / charge) if charge > 0 else np.nan),
                     "midday_proxy_absorption_ratio": (
                         float(overlap.sum() / proxy_total) if proxy_total > 0 else np.nan),
                     "midday_forecast_proxy_absorption_ratio": (
                         float(forecast_overlap.sum() / forecast_proxy_total)
                         if forecast_proxy_total > 0 else np.nan),
                     "soc_at_16_mwh": _soc_at_boundary(group, 16),
                     "soc_at_17_mwh": _soc_at_boundary(group, 17),
                     "soc_at_22_mwh": _soc_at_boundary(group, 22),
                     "evening_reserve_mw_hours": float(
                         0.25 * group.loc[evening, "reserve_commitment_mw"].sum()),
                     "evening_required_activation_mwh": required,
                     "evening_delivered_activation_mwh": delivered,
                     "evening_undelivered_activation_mwh": required - delivered,
                     "evening_delivery_rate": delivered / required if required > 0 else np.nan,
                     "recovery_grid_mwh": float(0.25 * group["recovery_power_mw"].sum()),
                     "recovery_cost_eur": float(group["recovery_cost_eur"].sum()),
                     "midday_total_eur": float(group.loc[mid, "total_eur"].sum()),
                     "evening_total_eur": float(group.loc[evening, "total_eur"].sum()),
                     "total_eur": float(group["total_eur"].sum())})
    return pd.DataFrame(rows)


def _midday_evening_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, chi), group in daily.groupby(["strategy", "chi"], sort=True):
        charge = float(group["midday_charge_mwh"].sum())
        overlap = float(group["midday_proxy_overlap_mwh"].sum())
        forecast_overlap = float(group["midday_forecast_proxy_overlap_mwh"].sum())
        proxy = float(group["midday_proxy_surplus_mwh"].sum())
        forecast_proxy = float(group["midday_forecast_proxy_surplus_mwh"].sum())
        required = float(group["evening_required_activation_mwh"].sum())
        delivered = float(group["evening_delivered_activation_mwh"].sum())
        rows.append({"strategy": strategy, "chi": float(chi), "days": len(group),
                     "fallback_days": int(group["fallback"].sum()),
                     "midday_charge_mwh": charge,
                     "mean_daily_midday_charge_mwh": float(group["midday_charge_mwh"].mean()),
                     "midday_charge_on_proxy_positive_hours_mwh": float(
                         group["midday_charge_on_proxy_positive_hours_mwh"].sum()),
                     "midday_proxy_overlap_mwh": overlap,
                     "midday_forecast_proxy_overlap_mwh": forecast_overlap,
                     "midday_alignment_share": overlap / charge if charge > 0 else np.nan,
                     "midday_forecast_alignment_share": (
                         forecast_overlap / charge if charge > 0 else np.nan),
                     "midday_proxy_absorption_ratio": overlap / proxy if proxy > 0 else np.nan,
                     "midday_forecast_proxy_absorption_ratio": (
                         forecast_overlap / forecast_proxy if forecast_proxy > 0 else np.nan),
                     "mean_soc_at_17_mwh": float(group["soc_at_17_mwh"].mean()),
                     "evening_reserve_mw_hours": float(group["evening_reserve_mw_hours"].sum()),
                     "evening_required_activation_mwh": required,
                     "evening_delivered_activation_mwh": delivered,
                     "evening_undelivered_activation_mwh": required - delivered,
                     "evening_delivery_rate": delivered / required if required > 0 else np.nan,
                     "recovery_grid_mwh": float(group["recovery_grid_mwh"].sum()),
                     "recovery_cost_eur": float(group["recovery_cost_eur"].sum()),
                     "total_eur": float(group["total_eur"].sum())})
    return pd.DataFrame(rows)


def _midday_evening_paired(daily: pd.DataFrame, study: dict) -> pd.DataFrame:
    metrics = ["midday_charge_mwh", "midday_proxy_overlap_mwh",
               "midday_forecast_proxy_overlap_mwh", "soc_at_17_mwh",
               "evening_reserve_mw_hours", "evening_delivered_activation_mwh",
               "evening_undelivered_activation_mwh", "recovery_grid_mwh", "total_eur"]
    rows = []
    for pair in study["midday_evening_analysis"]["paired_comparisons"]:
        left = daily.loc[(daily["strategy"] == pair["strategy"])
                         & daily["chi"].eq(float(pair["chi"]))]
        right = daily.loc[(daily["strategy"] == pair["comparator"])
                          & daily["chi"].eq(float(pair["comparator_chi"]))]
        joined = left.merge(right, on="delivery_day", suffixes=("_left", "_right"),
                            validate="one_to_one")
        _require(len(joined) == min(len(left), len(right)),
                 f"Paired mechanism common days differ: {pair}")
        for metric in metrics:
            difference = joined[f"{metric}_left"] - joined[f"{metric}_right"]
            if float(difference.max() - difference.min()) == 0.0:
                result = {"mean_loss_difference_a_minus_b": -float(difference.iloc[0]),
                          "confidence_lower": -float(difference.iloc[0]),
                          "confidence_upper": -float(difference.iloc[0]),
                          "dm_statistic": 0.0, "p_value": 1.0,
                          "n_valid_days": len(difference)}
            else:
                result = paired_metrics(joined[f"{metric}_left"],
                                        joined[f"{metric}_right"], study)
            rows.append({**pair, "metric": metric,
                         "mean_difference_strategy_minus_comparator": -result[
                             "mean_loss_difference_a_minus_b"],
                         "confidence_lower": -result["confidence_upper"],
                         "confidence_upper": -result["confidence_lower"],
                         "dm_statistic": -result["dm_statistic"],
                         "p_value": result["p_value"],
                         "n_days": result["n_valid_days"]})
    return pd.DataFrame(rows)


def _write_midday_outputs(study: dict, trace: pd.DataFrame, proxy: pd.DataFrame,
                          proxy_frames: tuple, mechanism_frames: tuple) -> None:
    directory = ROOT / study["output"]["directory"]
    trace_directory = directory / study["output"]["trace_directory"]
    trace_directory.mkdir(parents=True, exist_ok=True)
    expected_files = {item["file"] for item in study["midday_evening_analysis"]["trace_arms"]}
    existing_files = {path.name for path in trace_directory.glob("*.csv")}
    _require(existing_files.issubset(expected_files),
             f"Unexpected stale trace files: {sorted(existing_files - expected_files)}")
    for item in study["midday_evening_analysis"]["trace_arms"]:
        selected = trace.loc[(trace["strategy"] == item["strategy"])
                             & trace["chi"].eq(float(item["chi"]))]
        _require(len(selected) > 0, f"No trace rows for {item}")
        path = trace_directory / item["file"]
        _write_frame(selected, path)
        _require(path.stat().st_size <= 20 * 1024 * 1024, f"Trace file exceeds 20 MB: {path}")
    monthly, clock, month_clock = proxy_frames
    daily, summary, paired = mechanism_frames
    named = {study["output"]["proxy_hourly_csv"]: proxy,
             study["output"]["proxy_monthly_csv"]: monthly,
             study["output"]["proxy_clock_hour_csv"]: clock,
             study["output"]["proxy_month_clock_hour_csv"]: month_clock,
             study["output"]["midday_evening_daily_csv"]: daily,
             study["output"]["midday_evening_summary_csv"]: summary,
             study["output"]["midday_evening_paired_csv"]: paired}
    for name, frame in named.items():
        _write_frame(frame, directory / name)


def _write_proxy_extension_outputs(study: dict, decomposition: tuple,
                                   gate_forecasts: tuple) -> None:
    directory = ROOT / study["output"]["directory"]
    hourly, monthly, clock, _ = decomposition
    forecast_hourly, forecast_summary, forecast_monthly, _ = gate_forecasts
    named = {
        study["output"]["proxy_decomposition_hourly_csv"]: hourly,
        study["output"]["proxy_decomposition_monthly_csv"]: monthly,
        study["output"]["proxy_decomposition_clock_hour_csv"]: clock,
        study["output"]["gate_proxy_forecasts_hourly_csv"]: forecast_hourly,
        study["output"]["gate_proxy_forecast_summary_csv"]: forecast_summary,
        study["output"]["gate_proxy_forecast_monthly_csv"]: forecast_monthly,
    }
    for name, frame in named.items():
        _write_frame(frame, directory / name)


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
                     validation: dict, written: dict, trace_validation: dict,
                     proxy_coverage: dict, proxy_summary: dict,
                     mechanism_summary: pd.DataFrame, decomposition_summary: dict,
                     forecast_summary: pd.DataFrame, forecast_audit: dict,
                     proxy_decision_audit: dict) -> dict:
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
    mechanism_records = mechanism_summary.to_dict(orient="records")
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
            "midday_evening_analysis": {
                "trace_validation": trace_validation,
                "proxy_coverage": proxy_coverage,
                "proxy_summary": proxy_summary,
                "proxy_decomposition": decomposition_summary,
                "gate_proxy_forecast": {
                    "audit": forecast_audit,
                    "skill": json.loads(forecast_summary.to_json(orient="records")),
                },
                "proxy_aware_decision": proxy_decision_audit,
                "matched_complete_days": int(mechanism_summary["days"].min()),
                "strategy_results": mechanism_records,
            },
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
    study, forecast, _, state = _setup_full(path)
    directory = ROOT / study["output"]["directory"]
    proxy, proxy_coverage = _proxy_hourly(study)
    proxy_monthly, proxy_clock, proxy_month_clock, proxy_summary = _proxy_outputs(proxy, study)
    gate_forecasts = _gate_proxy_forecasts(proxy, study, forecast, state["days"])
    proxy_decision_signals, proxy_decision_audit = _proxy_decision_signals(
        gate_forecasts[0], study)
    state["proxy_decision_signals"] = proxy_decision_signals
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
    daily_rows, trace = _trace_rows(batches)
    daily = pd.DataFrame(daily_rows)
    risk = pd.DataFrame([row for batch in batches for row in batch["risk"]])
    cross = pd.DataFrame([row for batch in batches for row in batch["cross"]])
    daily = daily.sort_values(["delivery_day", "strategy", "chi"]).reset_index(drop=True)
    risk = risk.sort_values(["delivery_day", "strategy", "chi"]).reset_index(drop=True)
    cross = cross.sort_values(
        ["delivery_day", "model_strategy", "scenario_strategy"]).reset_index(drop=True)
    validation = _validate_full(daily, study, state["arm"], state["storage"])
    trace_validation = _trace_validation(trace, daily, study)
    decomposition = _proxy_decomposition(proxy, study)
    mechanism_daily = _midday_evening_daily(
        trace, proxy, gate_forecasts[0], set(proxy_decision_signals), study)
    mechanism_summary = _midday_evening_summary(mechanism_daily)
    mechanism_paired = _midday_evening_paired(mechanism_daily, study)
    written = _write_outputs(study, daily, risk, cross)
    _write_midday_outputs(
        study, trace, proxy, (proxy_monthly, proxy_clock, proxy_month_clock),
        (mechanism_daily, mechanism_summary, mechanism_paired))
    _write_proxy_extension_outputs(study, decomposition, gate_forecasts)
    payload = _summary_payload(
        path, study, state, checks, validation, written, trace_validation,
        proxy_coverage, proxy_summary, mechanism_summary, decomposition[3],
        gate_forecasts[1], gate_forecasts[3], proxy_decision_audit)
    (directory / study["output"]["summary_json"]).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"days": len(state["days"]), "daily_rows": len(daily),
            "risk_rows": len(risk), "cross_score_rows": len(cross),
            "trace_rows": len(trace), "proxy_complete_days": proxy_coverage["complete_days"],
            "proxy_decision_days": proxy_decision_audit["complete_decision_days"],
            "mechanism_days": int(mechanism_summary["days"].min()),
            "validation": validation, "decision_value": payload["decision_value"]}


def _existing_trace_with_proxy(study: dict, proxy_trace: pd.DataFrame) -> pd.DataFrame:
    directory = ROOT / study["output"]["directory"] / study["output"]["trace_directory"]
    proxy_names = {item["strategy"] for item in study["midday_evening_analysis"][
        "proxy_aware_decision"]["strategies"]}
    frames = [proxy_trace]
    for item in study["midday_evening_analysis"]["trace_arms"]:
        if item["strategy"] in proxy_names:
            continue
        path = directory / item["file"]
        _require(path.is_file(), f"Stored control trace is missing: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["strategy", "chi", "timestamp_utc"]).reset_index(drop=True)


def run_proxy_decision(path: Path) -> dict:
    checks = run_checks(path)
    study, forecast, _, state = _setup_full(path)
    directory = ROOT / study["output"]["directory"]
    proxy, proxy_coverage = _proxy_hourly(study)
    proxy_monthly, proxy_clock, proxy_month_clock, proxy_summary = _proxy_outputs(proxy, study)
    decomposition = _proxy_decomposition(proxy, study)
    gate_forecasts = _gate_proxy_forecasts(proxy, study, forecast, state["days"])
    signals, proxy_decision_audit = _proxy_decision_signals(gate_forecasts[0], study)
    state["proxy_decision_signals"] = signals
    WORKER_STATE.clear()
    WORKER_STATE.update(state)
    decision_days = sorted(signals)
    context = mp.get_context("spawn")
    with context.Pool(int(study["runtime"]["worker_processes"]),
                      initializer=_init_full_worker, initargs=(state,)) as pool:
        batches = []
        for completed, batch in enumerate(
                pool.imap(_proxy_day_worker, enumerate(decision_days)), start=1):
            batches.append(batch)
            if completed % 10 == 0 or completed == len(decision_days):
                print(json.dumps({"completed_proxy_days": completed,
                                  "total_proxy_days": len(decision_days)}), flush=True)
    new_rows, proxy_trace = _trace_rows(batches)
    new_daily = pd.DataFrame(new_rows).sort_values(
        ["delivery_day", "strategy", "chi"]).reset_index(drop=True)
    new_risk = pd.DataFrame([row for batch in batches for row in batch["risk"]])
    strategy_names = {item["strategy"] for item in study["midday_evening_analysis"][
        "proxy_aware_decision"]["strategies"]}
    stored_daily = pd.read_csv(directory / study["output"]["daily_results_csv"])
    stored_daily = stored_daily.loc[~stored_daily["strategy"].isin(strategy_names)]
    daily = pd.concat([stored_daily, new_daily], ignore_index=True).sort_values(
        ["delivery_day", "strategy", "chi"]).reset_index(drop=True)
    validation = _validate_full(daily, study, state["arm"], state["storage"])
    computed = _arm_summary(daily)
    computed["evidence_role"] = "same_protocol_computation"
    paired = _paired_differences(daily, computed, study)
    _write_frame(daily, directory / study["output"]["daily_results_csv"])
    _write_frame(pd.concat([computed, _reference_rows(study)], ignore_index=True),
                 directory / study["output"]["arm_summary_csv"])
    _write_frame(paired, directory / study["output"]["paired_differences_csv"])
    controls = computed.loc[~computed["strategy"].isin(strategy_names)]
    _write_frame(_frontier(controls), directory / study["output"]["frontier_csv"])
    _write_frame(_monthly(daily), directory / study["output"]["monthly_csv"])
    _write_frame(_stress_summary(daily), directory / study["output"]["stress_hours_csv"])
    stored_risk = pd.read_csv(directory / study["output"]["risk_calibration_csv"])
    stored_risk = stored_risk.loc[~stored_risk["strategy"].isin(strategy_names)]
    risk_summary = pd.concat(
        [stored_risk, aggregate_calibration(new_risk)], ignore_index=True).sort_values(
            ["strategy", "chi"]).reset_index(drop=True)
    _write_frame(risk_summary, directory / study["output"]["risk_calibration_csv"])
    trace = _existing_trace_with_proxy(study, proxy_trace)
    trace_validation = _trace_validation(trace, daily, study)
    mechanism_daily = _midday_evening_daily(
        trace, proxy, gate_forecasts[0], set(signals), study)
    mechanism_summary = _midday_evening_summary(mechanism_daily)
    mechanism_paired = _midday_evening_paired(mechanism_daily, study)
    _write_midday_outputs(
        study, trace, proxy, (proxy_monthly, proxy_clock, proxy_month_clock),
        (mechanism_daily, mechanism_summary, mechanism_paired))
    _write_proxy_extension_outputs(study, decomposition, gate_forecasts)
    written = {"computed": computed, "paired": paired}
    payload = _summary_payload(
        path, study, state, checks, validation, written, trace_validation,
        proxy_coverage, proxy_summary, mechanism_summary, decomposition[3],
        gate_forecasts[1], gate_forecasts[3], proxy_decision_audit)
    (directory / study["output"]["summary_json"]).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return {"proxy_decision_days": len(decision_days),
            "proxy_daily_rows": len(new_daily), "proxy_trace_rows": len(proxy_trace),
            "validation": validation}


def run_proxy_analysis(path: Path) -> dict:
    study, forecast, _ = load_configs(path)
    directory = ROOT / study["output"]["directory"]
    daily = pd.read_csv(directory / study["output"]["daily_results_csv"])
    test_days = sorted(daily["delivery_day"].astype(str).unique())
    _require(len(test_days) == forecast["split"]["expected_test_days"],
             "Stored decision results contain the wrong test-day count")
    proxy, coverage = _proxy_hourly(study)
    proxy_monthly, proxy_clock, proxy_month_clock, _ = _proxy_outputs(proxy, study)
    decomposition = _proxy_decomposition(proxy, study)
    gate_forecasts = _gate_proxy_forecasts(proxy, study, forecast, test_days)
    signals, proxy_decision_audit = _proxy_decision_signals(gate_forecasts[0], study)
    trace_directory = directory / study["output"]["trace_directory"]
    trace = pd.concat([
        pd.read_csv(trace_directory / item["file"])
        for item in study["midday_evening_analysis"]["trace_arms"]
    ], ignore_index=True).sort_values(
        ["strategy", "chi", "timestamp_utc"]).reset_index(drop=True)
    trace_validation = _trace_validation(trace, daily, study)
    mechanism_daily = _midday_evening_daily(
        trace, proxy, gate_forecasts[0], set(signals), study)
    mechanism_summary = _midday_evening_summary(mechanism_daily)
    mechanism_paired = _midday_evening_paired(mechanism_daily, study)
    _write_midday_outputs(
        study, trace, proxy, (proxy_monthly, proxy_clock, proxy_month_clock),
        (mechanism_daily, mechanism_summary, mechanism_paired))
    _write_proxy_extension_outputs(study, decomposition, gate_forecasts)
    summary_path = directory / study["output"]["summary_json"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    section = summary["midday_evening_analysis"]
    section["proxy_decomposition"] = decomposition[3]
    section["gate_proxy_forecast"] = {
        "audit": gate_forecasts[3],
        "skill": json.loads(gate_forecasts[1].to_json(orient="records")),
    }
    section["proxy_aware_decision"] = proxy_decision_audit
    section["trace_validation"] = trace_validation
    section["matched_complete_days"] = int(mechanism_summary["days"].min())
    section["strategy_results"] = mechanism_summary.to_dict(orient="records")
    summary["proxy_analysis_environment"] = environment_fingerprint()
    summary["proxy_analysis_config_sha256"] = sha256((ROOT / path).read_bytes()).hexdigest()
    summary["file_sha256"] = _output_hashes(directory)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"proxy_complete_days": coverage["complete_days"],
            "decomposition_positive_hours": decomposition[3]["proxy_positive_hours"],
            "forecast_rows": len(gate_forecasts[0]),
            "proxy_decision_days": len(signals),
            "forecast_audit": gate_forecasts[3]}


def main() -> None:
    arguments = parse_arguments()
    modes = [arguments.export, arguments.check, arguments.proxy_analysis,
             arguments.proxy_decision]
    _require(sum(bool(mode) for mode in modes) <= 1, "Choose at most one mode")
    if arguments.export:
        result = run_export(arguments.config)
    elif arguments.check:
        result = run_checks(arguments.config)
    elif arguments.proxy_analysis:
        result = run_proxy_analysis(arguments.config)
    elif arguments.proxy_decision:
        result = run_proxy_decision(arguments.config)
    else:
        result = run_full(arguments.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
