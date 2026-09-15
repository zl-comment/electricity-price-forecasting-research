"""Independent synthetic checks for the S3-B planner and risk score."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linprog

from epf_harness.dk1_storage import solve_day


ROOT = Path(__file__).resolve().parents[2]


def _storage(slot_hours: float, power: float, energy: float,
             efficiency: float, initial_soc: float, reserve_hours: float) -> tuple:
    arm = {"power_mw": power, "energy_mwh": energy, "market_share_cap": 1.0}
    storage = {
        "round_trip_efficiency": efficiency,
        "reserve_delivery_hours": reserve_hours,
        "slot_hours": slot_hours,
        "initial_soc_share": initial_soc / energy,
        "soc_min_share": 0.0,
        "soc_max_share": 1.0,
    }
    return arm, storage


def _slot_hours(config: dict) -> float:
    path = ROOT / config["storage_config"]
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = float(source["protocol"]["slot_hours"])
    if value <= 0.0:
        raise ValueError("slot_hours must be positive")
    return value


def degenerate_case(config: dict) -> dict:
    """Return the S=K=1 fixture and its frozen solve_day reference."""
    slot_hours = _slot_hours(config)
    slots = int(round(1.0 / slot_hours))
    if not np.isclose(slots * slot_hours, 1.0):
        raise ValueError("slot_hours must partition the one-hour hand check")
    arm, storage = _storage(
        slot_hours, 4.0, 8.0, 0.85, 4.4,
        float(config["storage"]["reserve_delivery_hours"]),
    )
    reference = solve_day(
        np.full(slots, 2.0), np.array([10.0]), np.array([4.0]),
        np.zeros(slots, dtype=int), arm, storage, config["solver"], None,
    )
    return {
        "scenario_probability": np.array([1.0]),
        "cluster_labels": np.array([0], dtype=int),
        "day_ahead_price": np.full((1, slots), 2.0),
        "capacity_price": np.array([[10.0]]),
        "activation_share": np.zeros((1, slots)),
        "activation_price": np.zeros((1, slots)),
        "shortfall_price": np.zeros((1, slots)),
        "procured_mw": np.array([4.0]),
        "hour_of_slot": np.zeros(slots, dtype=int),
        "arm": arm, "storage": storage, "chi": 0.0,
        "expected_reserve_mw": 4.0,
        "expected_objective_eur": 40.0,
        "solve_day_reference": reference,
    }


def two_scenario_case(config: dict) -> dict:
    """Return the S=2, K=1 activation fixture and exact optima."""
    slot_hours = _slot_hours(config)
    slots = int(round(1.0 / slot_hours))
    if not np.isclose(slots * slot_hours, 1.0):
        raise ValueError("slot_hours must partition the one-hour hand check")
    arm, storage = _storage(
        slot_hours, 1.0, 4.0, 0.81, 2.0,
        float(config["storage"]["reserve_delivery_hours"]),
    )
    activation = np.zeros((2, slots))
    activation[1, 0] = 0.25
    return {
        "scenario_probability": np.full(2, 0.5),
        "cluster_labels": np.zeros(2, dtype=int),
        "day_ahead_price": np.full((2, slots), 2.0),
        "capacity_price": np.full((2, 1), 10.0),
        "activation_share": activation,
        "activation_price": np.full((2, slots), 20.0),
        "shortfall_price": np.full((2, slots), 50.0),
        "procured_mw": np.array([1.0]),
        "hour_of_slot": np.zeros(slots, dtype=int),
        "arm": arm, "storage": storage,
        "expected": {
            "chi0": {"reserve_mw": 1.0, "objective_eur": float(Fraction(775, 162))},
            "chi1": {"reserve_mw": 0.0, "objective_eur": 0.0},
        },
    }


def check_degenerate_solution(objective_eur: float, reserve_mw: np.ndarray,
                              config: dict) -> dict:
    """Check a planner result against both arithmetic and frozen solve_day."""
    case = degenerate_case(config)
    tolerance = float(config["solver"]["objective_tolerance_eur"])
    reserve_tolerance = float(config["solver"]["exclusivity_tolerance_mw"])
    reference = case["solve_day_reference"]
    reserve = np.asarray(reserve_mw, dtype=float)
    if not np.isfinite(objective_eur) or reserve.size == 0 or not np.isfinite(reserve).all():
        raise ValueError("degenerate solution must contain finite objective and reserve values")
    objective_errors = np.abs([
        objective_eur - case["expected_objective_eur"],
        objective_eur - reference["planned_profit_eur"],
    ])
    reserve_error = float(np.max(np.abs(reserve - case["expected_reserve_mw"])))
    if float(objective_errors.max()) > tolerance:
        raise ValueError(f"degenerate objective errors exceed tolerance: {objective_errors}")
    if reserve_error > reserve_tolerance:
        raise ValueError(f"degenerate reserve error exceeds tolerance: {reserve_error}")
    return {"maximum_objective_error_eur": float(objective_errors.max()),
            "maximum_reserve_error_mw": reserve_error}


def _solve_reduced_cvar(chi: float, config: dict) -> dict:
    beta = float(config["risk"]["cvar_level"])
    coefficients = np.array([10.0, float(Fraction(-35, 81))])
    probabilities = np.full(2, 0.5)
    cost = np.array([
        -(1.0 - chi) * float(probabilities @ coefficients), -chi,
        chi * probabilities[0] / (1.0 - beta),
        chi * probabilities[1] / (1.0 - beta),
    ])
    constraints = np.zeros((2, 4))
    constraints[:, 0] = -coefficients
    constraints[:, 1] = 1.0
    constraints[np.arange(2), 2 + np.arange(2)] = -1.0
    result = linprog(
        cost, A_ub=constraints, b_ub=np.zeros(2),
        bounds=[(0.0, 1.0), (None, None), (0.0, None), (0.0, None)],
        method=config["solver"]["method"],
    )
    if not result.success:
        raise ValueError(f"reduced CVaR LP did not solve: {result.message}")
    return {"reserve_mw": float(result.x[0]), "objective_eur": float(-result.fun)}


def check_two_scenario_solution(chi0: dict, chi1: dict, config: dict) -> dict:
    """Check the primary LP and an independent reduced LP against exact optima."""
    expected = two_scenario_case(config)["expected"]
    independent = {"chi0": _solve_reduced_cvar(0.0, config),
                   "chi1": _solve_reduced_cvar(1.0, config)}
    objective_tolerance = float(config["solver"]["objective_tolerance_eur"])
    reserve_tolerance = float(config["solver"]["exclusivity_tolerance_mw"])
    diagnostics = {}
    for name, observed in (("chi0", chi0), ("chi1", chi1)):
        if not np.isfinite([observed["objective_eur"], observed["reserve_mw"]]).all():
            raise ValueError(f"two-scenario {name} solution must be finite")
        objective_errors = [abs(float(observed["objective_eur"]) - expected[name]["objective_eur"]),
                            abs(independent[name]["objective_eur"] - expected[name]["objective_eur"])]
        reserve_errors = [abs(float(observed["reserve_mw"]) - expected[name]["reserve_mw"]),
                          abs(independent[name]["reserve_mw"] - expected[name]["reserve_mw"])]
        if max(objective_errors) > objective_tolerance or max(reserve_errors) > reserve_tolerance:
            raise ValueError(f"two-scenario {name} check failed: {objective_errors}, {reserve_errors}")
        diagnostics[name] = {"maximum_objective_error_eur": float(max(objective_errors)),
                             "maximum_reserve_error_mw": float(max(reserve_errors))}
    return diagnostics


def check_nonanticipativity(cluster_labels: np.ndarray, scenario_p_ch: np.ndarray,
                            scenario_p_dis: np.ndarray, reserve_0730: np.ndarray,
                            reserve_1200: np.ndarray, config: dict) -> dict:
    """Assert equal within-cluster plans and preservation of the gate reserve."""
    labels = np.asarray(cluster_labels)
    charge = np.asarray(scenario_p_ch, dtype=float)
    discharge = np.asarray(scenario_p_dis, dtype=float)
    if (labels.ndim != 1 or len(labels) == 0 or charge.shape != discharge.shape
            or charge.shape[0] != len(labels)):
        raise ValueError("nonanticipativity arrays have inconsistent shapes")
    if not np.isfinite(charge).all() or not np.isfinite(discharge).all():
        raise ValueError("nonanticipativity plans must be finite")
    within_errors = []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        within_errors.extend([float(np.max(np.abs(charge[positions] - charge[positions[0]]))),
                              float(np.max(np.abs(discharge[positions] - discharge[positions[0]])))])
    first_reserve = np.asarray(reserve_0730, dtype=float)
    second_reserve = np.asarray(reserve_1200, dtype=float)
    if (first_reserve.ndim != 1 or first_reserve.size == 0
            or first_reserve.shape != second_reserve.shape
            or not np.isfinite(first_reserve).all() or not np.isfinite(second_reserve).all()):
        raise ValueError("gate reserve arrays must be finite vectors with equal shapes")
    reserve_error = float(np.max(np.abs(first_reserve - second_reserve)))
    tolerance = float(config["solver"]["exclusivity_tolerance_mw"])
    if max(within_errors) > tolerance or reserve_error > tolerance:
        raise ValueError(f"nonanticipativity failed: within={max(within_errors)}, reserve={reserve_error}")
    return {"maximum_within_cluster_error_mw": float(max(within_errors)),
            "maximum_frozen_reserve_error_mw": reserve_error}


def joint_var_cvar_score(var_k_eur: float, cvar_k_eur: float,
                         realised_k_eur: float, alpha: float) -> float:
    """Evaluate F07 equation (40) for lower-tail profit VaR/CVaR in kEUR."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    indicator = float(realised_k_eur <= var_k_eur)
    logistic = float(1.0 / (1.0 + np.exp(-cvar_k_eur)))
    pinball = (indicator - alpha) * (var_k_eur - realised_k_eur)
    tail = logistic * (indicator * (var_k_eur - realised_k_eur) / alpha
                       + cvar_k_eur - var_k_eur)
    return float(pinball + tail - np.logaddexp(0.0, cvar_k_eur))


def empirical_tail_metrics(profits_k_eur: np.ndarray, alpha: float) -> dict:
    """Compute exact equal-weight lower-tail metrics when alpha*n is integral."""
    values = np.sort(np.asarray(profits_k_eur, dtype=float))
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("profits must be a finite one-dimensional array")
    tail_count = int(round(alpha * len(values)))
    if tail_count < 1 or not np.isclose(tail_count, alpha * len(values)):
        raise ValueError("the hand-check tail must contain an integral positive count")
    var = float(values[tail_count - 1])
    return {"var_k_eur": var, "cvar_k_eur": float(values[:tail_count].mean()),
            "exceedance_rate": float(np.mean(values <= var))}


def check_risk_scoring(score_function, tail_function, config: dict) -> dict:
    """Check VaR/CVaR, exceedance, equation (40), and expected-score ordering."""
    alpha = 1.0 - float(config["risk"]["cvar_level"])
    profits = np.concatenate([np.array([-2.0]), np.ones(19)])
    expected = {"var_k_eur": -2.0, "cvar_k_eur": -2.0, "exceedance_rate": alpha}
    observed = tail_function(profits, alpha)
    tolerance = float(config["export"]["score_tolerance"])
    errors = {key: abs(float(observed[key]) - value) for key, value in expected.items()}
    if max(errors.values()) > tolerance:
        raise ValueError(f"tail metric hand check failed: {errors}")
    true_scores = np.array([score_function(-2.0, -2.0, value, alpha) for value in profits])
    expected_scores = np.array([joint_var_cvar_score(-2.0, -2.0, value, alpha)
                                for value in profits])
    score_error = float(np.max(np.abs(true_scores - expected_scores)))
    shifted = np.array([score_function(-2.0 + alpha, -2.0 + alpha, value, alpha)
                        for value in profits])
    analytic_expected_score = 3.0 * alpha * (1.0 - alpha) - np.logaddexp(0.0, -2.0)
    expectation_error = abs(float(true_scores.mean()) - analytic_expected_score)
    if (score_error > tolerance or expectation_error > tolerance
            or float(true_scores.mean()) > float(shifted.mean()) + tolerance):
        raise ValueError(f"risk score hand check failed: {score_error}, {expectation_error}")
    return {"tail_metric_maximum_error": float(max(errors.values())),
            "score_maximum_error": score_error,
            "expected_score_absolute_error": float(expectation_error),
            "true_expected_score": float(true_scores.mean()),
            "shifted_expected_score": float(shifted.mean())}


def self_check(config: dict) -> dict:
    """Run all checks that do not require the future S3-B planner implementation."""
    degenerate = degenerate_case(config)
    reference = degenerate["solve_day_reference"]
    check_one = check_degenerate_solution(
        reference["planned_profit_eur"], reference["r_up"], config)
    independent = {"chi0": _solve_reduced_cvar(0.0, config),
                   "chi1": _solve_reduced_cvar(1.0, config)}
    check_two = check_two_scenario_solution(independent["chi0"], independent["chi1"], config)
    check_three = check_nonanticipativity(
        np.array([0, 0, 1]), np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([1.0]), np.array([1.0]), config)
    check_ten = check_risk_scoring(joint_var_cvar_score, empirical_tail_metrics, config)
    return {"degenerate_consistency": check_one, "two_scenario": check_two,
            "nonanticipativity": check_three, "risk_scoring": check_ten}


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = _load_config(parser.parse_args().config)
    print(json.dumps(self_check(config), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
