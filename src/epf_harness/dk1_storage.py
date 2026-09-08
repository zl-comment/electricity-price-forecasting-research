"""S2 storage baselines: one feasible region, one settlement engine, three strategies.

The LP plans the delivery day at the two gates. Activation is exogenous and is applied
afterwards by the settlement replay, so a plan that cannot be delivered shows up as a
shortfall rather than being optimised away.
"""

import numpy as np
from scipy.optimize import linprog


def efficiencies(round_trip: float) -> tuple:
    """Split the round-trip loss evenly over charge and discharge."""
    one_way = float(np.sqrt(round_trip))
    return one_way, one_way


def check_boundary_feasibility(arm: dict, storage: dict) -> float:
    """Committing full power in the first or last hour needs stored energy the terminal rule must allow."""
    _, eta_d = efficiencies(storage["round_trip_efficiency"])
    required = arm["power_mw"] * storage["reserve_delivery_hours"] / eta_d
    soc_0 = arm["energy_mwh"] * storage["initial_soc_share"]
    if soc_0 < required:
        raise ValueError(
            f"initial_soc_share {storage['initial_soc_share']} gives {soc_0:.4f} MWh but a full-power "
            f"commitment reserves {required:.4f} MWh; the terminal SOC rule makes the day infeasible"
        )
    return float(required)


def soc_matrix(n_slots: int, eta_c: float, eta_d: float, delta: float, n_hours: int) -> np.ndarray:
    """Row t maps the decision vector to soc[t] - soc_0. Columns are [p_ch, p_dis, r_up]."""
    lower = np.tril(np.ones((n_slots, n_slots)))
    charge = lower * delta * eta_c
    discharge = -lower * delta / eta_d
    return np.hstack([charge, discharge, np.zeros((n_slots, n_hours))])


def hour_selector(hour_of_slot: np.ndarray, n_slots: int, n_hours: int) -> np.ndarray:
    """Row t is the indicator of the hour that slot t belongs to, padded over [p_ch, p_dis]."""
    selector = np.zeros((n_slots, n_hours))
    selector[np.arange(n_slots), hour_of_slot] = 1.0
    return np.hstack([np.zeros((n_slots, 2 * n_slots)), selector])


def day_constraints(soc: np.ndarray, hours: np.ndarray, arm: dict, storage: dict, soc_0: float) -> tuple:
    """Power headroom, SOC band, and the reserve energy reservation, in one inequality block."""
    n_slots = soc.shape[0]
    _, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_max = arm["energy_mwh"] * storage["soc_max_share"]
    soc_min = arm["energy_mwh"] * storage["soc_min_share"]
    discharge = np.zeros_like(soc)
    discharge[np.arange(n_slots), n_slots + np.arange(n_slots)] = 1.0
    a_ub = np.vstack([soc, -soc, discharge + hours, -soc + hours * storage["reserve_delivery_hours"] / eta_d])
    b_ub = np.concatenate([
        np.full(n_slots, soc_max - soc_0),
        np.full(n_slots, soc_0 - soc_min),
        np.full(n_slots, arm["power_mw"]),
        np.full(n_slots, soc_0),
    ])
    return a_ub, b_ub


def day_bounds(arm: dict, n_slots: int, procured_mw: np.ndarray, r_up_fixed: np.ndarray) -> list:
    """Charge and discharge are free within power; the reserve column is fixed or capped."""
    power = arm["power_mw"]
    bounds = [(0.0, power)] * (2 * n_slots)
    if r_up_fixed is None:
        ceiling = np.minimum(power, arm["market_share_cap"] * procured_mw)
        bounds += [(0.0, float(value)) for value in ceiling]
    else:
        bounds += [(float(value), float(value)) for value in r_up_fixed]
    return bounds


def solve_day(prices: np.ndarray, capacity_prices: np.ndarray, procured_mw: np.ndarray,
              hour_of_slot: np.ndarray, arm: dict, storage: dict, solver: dict,
              r_up_fixed: np.ndarray) -> dict:
    """Plan one delivery day. `prices` is what the planner believes, not necessarily what happens."""
    n_slots, n_hours = len(prices), len(capacity_prices)
    delta = storage["slot_hours"]
    eta_c, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_0 = arm["energy_mwh"] * storage["initial_soc_share"]
    soc = soc_matrix(n_slots, eta_c, eta_d, delta, n_hours)
    hours = hour_selector(hour_of_slot, n_slots, n_hours)
    a_ub, b_ub = day_constraints(soc, hours, arm, storage, soc_0)
    cost = np.concatenate([delta * prices, -delta * prices, -capacity_prices])
    result = linprog(
        c=cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=soc[-1:].copy(),
        b_eq=np.zeros(1),
        bounds=day_bounds(arm, n_slots, procured_mw, r_up_fixed),
        method=solver["method"],
    )
    if not result.success:
        raise ValueError(f"LP did not solve: {result.message}")
    return {
        "p_ch": result.x[:n_slots],
        "p_dis": result.x[n_slots:2 * n_slots],
        "r_up": result.x[2 * n_slots:],
        "planned_profit_eur": float(-result.fun),
        "soc_plan": soc_0 + soc @ result.x,
    }


def exclusivity_violations(plan: dict, tolerance: float) -> int:
    """Charge and discharge in the same slot is only optimal at negative prices. Count, do not hide."""
    return int((np.minimum(plan["p_ch"], plan["p_dis"]) > tolerance).sum())


def dispatch_schedule(soc: float, p_ch: float, p_dis: float, delta: float,
                      eta_c: float, eta_d: float, soc_max: float) -> tuple:
    """Deliver the day-ahead schedule as far as stored energy allows.

    Charge and discharge net out inside the slot, which is the accounting the LP used, so a
    plan the LP proved feasible replays with no gap whenever activation does not intervene.
    """
    gain = delta * eta_c * p_ch
    draw = delta * p_dis / eta_d
    actual_ch, actual_dis = p_ch, p_dis
    if soc + gain - draw < 0.0:
        actual_dis = max(soc + gain, 0.0) * eta_d / delta
        draw = delta * actual_dis / eta_d
    if soc + gain - draw > soc_max:
        actual_ch = (soc_max - soc + draw) / (delta * eta_c)
        gain = delta * eta_c * actual_ch
    gap = delta * ((p_dis - actual_dis) + (p_ch - actual_ch))
    return soc + gain - draw, actual_dis, gap


def dispatch_activation(soc: float, need: float, actual_dis: float, delta: float,
                        eta_d: float, power_mw: float) -> tuple:
    """Activation sits on top of the schedule. Power binds first, then stored energy."""
    by_power = min(need, max(power_mw - actual_dis, 0.0) * delta)
    by_energy = min(by_power, max(soc, 0.0) * eta_d)
    return soc - by_energy / eta_d, by_energy, need - by_power, by_power - by_energy


def settle_day(plan: dict, realised: dict, arm: dict, storage: dict) -> dict:
    """Replay the planned day against realised prices and realised activation.

    The plan was made without knowing activation, so activation can drain the battery below
    what the schedule assumed. Both the undelivered schedule and the undelivered reserve are
    recorded rather than being allowed to drive the state of charge negative.
    """
    delta = storage["slot_hours"]
    eta_c, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_max = arm["energy_mwh"] * storage["soc_max_share"]
    soc = arm["energy_mwh"] * storage["initial_soc_share"]
    required = plan["r_up"][realised["hour_of_slot"]] * realised["activation_share"]
    delivered, schedule_gaps, trace = [], [], []
    power_gap = energy_gap = 0.0
    for index, need in enumerate(required):
        soc, actual_dis, gap = dispatch_schedule(
            soc, plan["p_ch"][index], plan["p_dis"][index], delta, eta_c, eta_d, soc_max
        )
        soc, served, missing_power, missing_energy = dispatch_activation(
            soc, need, actual_dis, delta, eta_d, arm["power_mw"]
        )
        power_gap += missing_power
        energy_gap += missing_energy
        delivered.append(served)
        schedule_gaps.append(gap)
        trace.append(soc)
    return activation_ledger(plan, realised, np.array(delivered), required, np.array(trace),
                             np.array(schedule_gaps), power_gap, energy_gap, arm, storage)


def activation_ledger(plan: dict, realised: dict, delivered: np.ndarray, required: np.ndarray,
                      trace: np.ndarray, schedule_gaps: np.ndarray, power_gap: float,
                      energy_gap: float, arm: dict, storage: dict) -> dict:
    """Money and physics of one settled day, kept in separate columns.

    The day-ahead position is settled at the day-ahead price because it was sold at the gate;
    whatever the battery then fails to deliver is bought back at the imbalance price.
    """
    delta = storage["slot_hours"]
    energy_eur = float(delta * np.sum(realised["day_ahead_price"] * (plan["p_dis"] - plan["p_ch"])))
    capacity_eur = float(np.sum(realised["capacity_price"] * plan["r_up"]))
    activation_eur = float(np.sum(delivered * realised["activation_price"]))
    shortfall = required - delivered
    shortfall_eur = float(np.sum(shortfall * realised["shortfall_price"]))
    schedule_gap_eur = float(np.sum(schedule_gaps * realised["shortfall_price"]))
    return {
        "energy_eur": energy_eur,
        "capacity_eur": capacity_eur,
        "activation_eur": activation_eur,
        "shortfall_cost_eur": shortfall_eur,
        "schedule_gap_cost_eur": schedule_gap_eur,
        "total_eur": energy_eur + capacity_eur + activation_eur - shortfall_eur - schedule_gap_eur,
        "committed_mw_hours": float(np.sum(plan["r_up"])),
        "required_activation_mwh": float(np.sum(required)),
        "delivered_activation_mwh": float(np.sum(delivered)),
        "reserve_power_gap_mwh": float(power_gap),
        "reserve_energy_gap_mwh": float(energy_gap),
        "day_ahead_gap_mwh": float(np.sum(schedule_gaps)),
        "slots_with_shortfall": int((shortfall > storage["soc_tolerance_mwh"]).sum()),
        "slots_with_schedule_gap": int((schedule_gaps > storage["soc_tolerance_mwh"]).sum()),
        "min_soc_mwh": float(trace.min()),
        "max_soc_mwh": float(trace.max()),
        "terminal_soc_mwh": float(trace[-1]),
        "terminal_soc_deviation_mwh": float(trace[-1] - arm["energy_mwh"] * storage["initial_soc_share"]),
        "cycles": float(delta * np.sum(plan["p_dis"]) / arm["energy_mwh"]),
    }


def threshold_rule(capacity_prices: np.ndarray, threshold: float, arm: dict,
                   procured_mw: np.ndarray) -> np.ndarray:
    """B1's 07:30 rule. Uses persisted capacity prices only; carries no delivery-day information."""
    ceiling = np.minimum(arm["power_mw"], arm["market_share_cap"] * procured_mw)
    return np.where(capacity_prices >= threshold, ceiling, 0.0)


def hand_check(config: dict) -> dict:
    """A four-slot day whose optimum is arithmetic: charge 0.5 MWh at 10, sell it at 100."""
    case = config["hand_check"]
    arm = {"power_mw": case["power_mw"], "energy_mwh": case["energy_mwh"], "market_share_cap": 1.0}
    storage = {
        "round_trip_efficiency": case["round_trip_efficiency"],
        "reserve_delivery_hours": config["storage"]["reserve_delivery_hours"],
        "slot_hours": config["protocol"]["slot_hours"],
        "initial_soc_share": case["initial_soc_mwh"] / case["energy_mwh"],
        "soc_min_share": 0.0,
        "soc_max_share": 1.0,
    }
    plan = solve_day(
        np.array(case["day_ahead_prices"]), np.array(case["capacity_prices"]), np.array([0.0]),
        np.zeros(len(case["day_ahead_prices"]), dtype=int), arm, storage, config["solver"],
        np.array([0.0]),
    )
    error = abs(plan["planned_profit_eur"] - case["expected_profit_eur"])
    if error > case["tolerance_eur"]:
        raise ValueError(f"Hand-check profit {plan['planned_profit_eur']} != {case['expected_profit_eur']}")
    terminal = abs(plan["soc_plan"][-1] - case["initial_soc_mwh"])
    if terminal > config["solver"]["soc_tolerance_mwh"]:
        raise ValueError(f"Hand-check terminal SOC deviates by {terminal}")
    return {
        "profit_eur": plan["planned_profit_eur"],
        "expected_profit_eur": case["expected_profit_eur"],
        "profit_abs_error": float(error),
        "terminal_soc_abs_error": float(terminal),
        "charged_mwh": float(config["protocol"]["slot_hours"] * plan["p_ch"].sum()),
        "discharged_mwh": float(config["protocol"]["slot_hours"] * plan["p_dis"].sum()),
    }
