"""S2 storage baselines: gate decisions followed by activation-aware settlement."""

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


def activation_selector(activation_share: np.ndarray, hour_of_slot: np.ndarray,
                        n_hours: int) -> np.ndarray:
    """Map hourly reserve commitments to realised activation energy in each slot."""
    selector = np.zeros((len(activation_share), n_hours))
    selector[np.arange(len(activation_share)), hour_of_slot] = activation_share
    return selector


def day_constraints(soc: np.ndarray, hours: np.ndarray, arm: dict, storage: dict, soc_0: float) -> tuple:
    """Power headroom, SOC band, and the reserve energy reservation, in one inequality block."""
    n_slots = soc.shape[0]
    _, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_max = arm["energy_mwh"] * storage["soc_max_share"]
    soc_min = arm["energy_mwh"] * storage["soc_min_share"]
    charge = np.zeros_like(soc)
    charge[np.arange(n_slots), np.arange(n_slots)] = 1.0
    discharge = np.zeros_like(soc)
    discharge[np.arange(n_slots), n_slots + np.arange(n_slots)] = 1.0
    gross_power = charge + discharge + hours
    a_ub = np.vstack([soc, -soc, gross_power,
                      -soc + hours * storage["reserve_delivery_hours"] / eta_d])
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
    soc_plan = soc_0 + soc @ result.x
    return {
        "p_ch": result.x[:n_slots],
        "p_dis": result.x[n_slots:2 * n_slots],
        "r_up": result.x[2 * n_slots:],
        "planned_profit_eur": float(-result.fun),
        "soc_plan": soc_plan,
        "base_terminal_soc_deviation_mwh": float(soc_plan[-1] - soc_0),
    }


def activation_soc_matrix(n_slots: int, n_hours: int, activation: np.ndarray,
                          eta_c: float, eta_d: float, delta: float) -> np.ndarray:
    """Post-slot SOC mapping for [p_ch, p_dis, r_up, p_recovery]."""
    lower = np.tril(np.ones((n_slots, n_slots)))
    return np.hstack([
        lower * delta * eta_c,
        -lower * delta / eta_d,
        -(lower @ activation) / eta_d,
        lower * delta * eta_c,
    ])


def activation_oracle_constraints(soc: np.ndarray, activation: np.ndarray, hours: np.ndarray,
                                  arm: dict, storage: dict, soc_0: float) -> tuple:
    """Physical constraints for a perfect-information activation-aware day."""
    n_slots, n_hours = activation.shape
    eta_c, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_max = arm["energy_mwh"] * storage["soc_max_share"]
    soc_min = arm["energy_mwh"] * storage["soc_min_share"]
    discharge = np.zeros_like(soc)
    discharge[np.arange(n_slots), n_slots + np.arange(n_slots)] = 1.0
    base_charge = np.zeros_like(soc)
    base_charge[np.arange(n_slots), np.arange(n_slots)] = 1.0
    charge = base_charge.copy()
    charge[np.arange(n_slots), 2 * n_slots + n_hours + np.arange(n_slots)] = 1.0
    activation_power = np.hstack([
        np.zeros((n_slots, 2 * n_slots)), activation / storage["slot_hours"],
        np.zeros((n_slots, n_slots)),
    ])
    current_activation = np.hstack([
        np.zeros((n_slots, 2 * n_slots)), activation / eta_d, np.zeros((n_slots, n_slots)),
    ])
    reservation = np.hstack([
        np.zeros((n_slots, 2 * n_slots)), hours * storage["reserve_delivery_hours"] / eta_d,
        np.zeros((n_slots, n_slots)),
    ])
    commitment_power = np.hstack([
        np.zeros((n_slots, 2 * n_slots)), hours, np.zeros((n_slots, n_slots)),
    ])
    base_gross_power = base_charge + discharge + commitment_power
    realised_gross_power = charge + discharge + activation_power
    a_ub = np.vstack([soc, -soc, base_gross_power, realised_gross_power,
                      -soc - current_activation + reservation])
    b_ub = np.concatenate([
        np.full(n_slots, soc_max - soc_0), np.full(n_slots, soc_0 - soc_min),
        np.full(n_slots, arm["power_mw"]), np.full(n_slots, arm["power_mw"]),
        np.full(n_slots, soc_0),
    ])
    return a_ub, b_ub


def solve_activation_oracle(realised: dict, arm: dict, storage: dict, solver: dict) -> dict:
    """Optimise energy, reserve, activation and paid recovery with perfect information."""
    prices, capacity = realised["day_ahead_price"], realised["capacity_price"]
    n_slots, n_hours = len(prices), len(capacity)
    delta = storage["slot_hours"]
    eta_c, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_0 = arm["energy_mwh"] * storage["initial_soc_share"]
    activation = activation_selector(realised["activation_share"], realised["hour_of_slot"], n_hours)
    hours = hour_selector(realised["hour_of_slot"], n_slots, n_hours)[:, 2 * n_slots:]
    soc = activation_soc_matrix(n_slots, n_hours, activation, eta_c, eta_d, delta)
    a_ub, b_ub = activation_oracle_constraints(soc, activation, hours, arm, storage, soc_0)
    activation_value = activation.T @ realised["activation_price"]
    cost = np.concatenate([
        delta * prices, -delta * prices, -capacity - activation_value,
        delta * realised["shortfall_price"],
    ])
    objective_scale = solver["objective_scale_eur"]
    bounds = day_bounds(arm, n_slots, realised["procured_mw"], None)
    bounds += [(0.0, arm["power_mw"])] * n_slots
    base_terminal = np.zeros((1, 3 * n_slots + n_hours))
    base_terminal[0, :2 * n_slots] = soc_matrix(n_slots, eta_c, eta_d, delta, n_hours)[-1, :2 * n_slots]
    terminal = np.vstack([base_terminal, soc[-1:]])
    result = linprog(c=cost / objective_scale, A_ub=a_ub, b_ub=b_ub,
                     A_eq=terminal, b_eq=np.zeros(2),
                     bounds=bounds, method=solver["method"])
    if not result.success:
        raise ValueError(f"Activation-aware Oracle LP did not solve: {result.message}")
    return {
        "p_ch": result.x[:n_slots], "p_dis": result.x[n_slots:2 * n_slots],
        "r_up": result.x[2 * n_slots:2 * n_slots + n_hours],
        "p_recovery": result.x[2 * n_slots + n_hours:],
        "planned_profit_eur": float(-result.fun * objective_scale),
        "soc_plan": soc_0 + soc @ result.x,
        "base_terminal_soc_deviation_mwh": float(base_terminal @ result.x),
    }


def exclusivity_violations(plan: dict, tolerance: float) -> int:
    """Charge and discharge in the same slot is only optimal at negative prices. Count, do not hide."""
    return int((np.minimum(plan["p_ch"], plan["p_dis"]) > tolerance).sum())


def recovery_soc_matrix(n_slots: int, eta_c: float, eta_d: float, delta: float) -> np.ndarray:
    """SOC adjustment from paid recovery power and delivered activation energy."""
    lower = np.tril(np.ones((n_slots, n_slots)))
    return np.hstack([lower * delta * eta_c, -lower / eta_d])


def recovery_constraints(plan: dict, realised: dict, arm: dict, storage: dict,
                         required: np.ndarray, soc: np.ndarray) -> tuple:
    """Keep recovery and delivered activation inside the frozen day plan's physical envelope."""
    n_slots = len(required)
    _, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_max = arm["energy_mwh"] * storage["soc_max_share"]
    soc_min = arm["energy_mwh"] * storage["soc_min_share"]
    current_delivery = np.hstack([np.zeros((n_slots, n_slots)), np.eye(n_slots) / eta_d])
    gross_power = np.hstack([np.eye(n_slots), np.eye(n_slots) / storage["slot_hours"]])
    reserved = plan["r_up"][realised["hour_of_slot"]] * storage["reserve_delivery_hours"] / eta_d
    tolerance = storage["recovery_constraint_tolerance_mwh"]
    a_ub = np.vstack([soc, -soc, -soc - current_delivery, gross_power])
    b_ub = np.concatenate([
        soc_max - plan["soc_plan"] + tolerance,
        plan["soc_plan"] - soc_min + tolerance,
        plan["soc_plan"] - reserved + tolerance,
        arm["power_mw"] - plan["p_ch"] - plan["p_dis"],
    ])
    headroom = np.maximum(arm["power_mw"] - plan["p_ch"] - plan["p_dis"], 0.0)
    bounds = [(0.0, float(value)) for value in headroom]
    bounds += [(0.0, float(value)) for value in np.minimum(required, headroom * storage["slot_hours"])]
    return a_ub, b_ub, bounds


def solve_recovery(plan: dict, realised: dict, arm: dict, storage: dict, solver: dict) -> dict:
    """Maximise delivered activation, then buy the least-cost energy needed to restore terminal SOC."""
    n_slots = len(plan["p_ch"])
    delta = storage["slot_hours"]
    eta_c, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc = recovery_soc_matrix(n_slots, eta_c, eta_d, delta)
    required = plan["r_up"][realised["hour_of_slot"]] * realised["activation_share"]
    a_ub, b_ub, bounds = recovery_constraints(plan, realised, arm, storage, required, soc)
    terminal = np.zeros((1, 2 * n_slots))
    terminal[0] = soc[-1]
    planned_terminal_error = plan["soc_plan"][-1] - arm["energy_mwh"] * storage["initial_soc_share"]
    if abs(planned_terminal_error) > storage["soc_tolerance_mwh"]:
        raise ValueError(f"Frozen plan terminal SOC deviates by {planned_terminal_error:.12f} MWh")
    target = np.zeros(1)
    delivery_cost = np.concatenate([np.zeros(n_slots), -np.ones(n_slots)])
    delivered = linprog(c=delivery_cost, A_ub=a_ub, b_ub=b_ub, A_eq=terminal, b_eq=target,
                         bounds=bounds, method=solver["method"])
    if not delivered.success:
        raise ValueError(f"Activation delivery LP did not solve: {delivered.message}")
    delivered_total = float(delivered.x[n_slots:].sum())
    preserve_delivery = np.zeros((1, 2 * n_slots))
    preserve_delivery[0, n_slots:] = -1.0
    a_ub = np.vstack([a_ub, preserve_delivery])
    b_ub = np.concatenate([b_ub, [-delivered_total + storage["soc_tolerance_mwh"]]])
    recovery_cost = np.concatenate([delta * realised["shortfall_price"], np.zeros(n_slots)])
    result = linprog(c=recovery_cost, A_ub=a_ub, b_ub=b_ub, A_eq=terminal, b_eq=target,
                     bounds=bounds, method=solver["method"])
    if not result.success:
        raise ValueError(f"SOC recovery LP did not solve: {result.message}")
    if result.x.min() < -storage["soc_tolerance_mwh"]:
        raise ValueError(f"SOC recovery LP returned a negative decision: {result.x.min():.12f}")
    solution = np.maximum(result.x, 0.0)
    solution[:n_slots][solution[:n_slots] < storage["soc_tolerance_mwh"] / (n_slots * delta)] = 0.0
    solution[n_slots:][solution[n_slots:] < storage["soc_tolerance_mwh"] / n_slots] = 0.0
    return {
        "p_recovery": solution[:n_slots], "delivered": solution[n_slots:],
        "required": required, "soc_trace": plan["soc_plan"] + soc @ solution,
    }


def settle_day(plan: dict, realised: dict, arm: dict, storage: dict, solver: dict) -> dict:
    """Settle frozen gate decisions with realised activation and paid SOC recovery."""
    required = plan["r_up"][realised["hour_of_slot"]] * realised["activation_share"]
    if "p_recovery" in plan:
        closed = {"p_recovery": plan["p_recovery"], "delivered": required,
                  "required": required, "soc_trace": plan["soc_plan"]}
    else:
        closed = solve_recovery(plan, realised, arm, storage, solver)
    power_limit = np.maximum(arm["power_mw"] - plan["p_dis"], 0.0) * storage["slot_hours"]
    power_gap = float(np.maximum(required - power_limit, 0.0).sum())
    total_gap = required - closed["delivered"]
    energy_gap = float(np.maximum(total_gap - np.maximum(required - power_limit, 0.0), 0.0).sum())
    schedule_gaps = np.zeros(len(required))
    return activation_ledger(plan, realised, closed["delivered"], required, closed["soc_trace"],
                             schedule_gaps, power_gap, energy_gap, closed["p_recovery"], arm, storage)


def activation_ledger(plan: dict, realised: dict, delivered: np.ndarray, required: np.ndarray,
                      trace: np.ndarray, schedule_gaps: np.ndarray, power_gap: float,
                      energy_gap: float, recovery: np.ndarray, arm: dict, storage: dict) -> dict:
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
    recovery_eur = float(delta * np.sum(recovery * realised["shortfall_price"]))
    tolerance = storage["soc_tolerance_mwh"]
    committed_gross_power = plan["p_ch"] + plan["p_dis"] + plan["r_up"][realised["hour_of_slot"]]
    realised_gross_power = plan["p_ch"] + plan["p_dis"] + delivered / delta + recovery
    return {
        "energy_eur": energy_eur,
        "capacity_eur": capacity_eur,
        "activation_eur": activation_eur,
        "shortfall_cost_eur": shortfall_eur,
        "schedule_gap_cost_eur": schedule_gap_eur,
        "recovery_cost_eur": recovery_eur,
        "total_eur": energy_eur + capacity_eur + activation_eur - shortfall_eur - schedule_gap_eur - recovery_eur,
        "committed_mw_hours": float(np.sum(plan["r_up"])),
        "required_activation_mwh": float(np.sum(required)),
        "delivered_activation_mwh": float(np.sum(delivered)),
        "reserve_power_gap_mwh": float(power_gap),
        "reserve_energy_gap_mwh": float(energy_gap),
        "day_ahead_gap_mwh": float(np.sum(schedule_gaps)),
        "recovery_grid_mwh": float(delta * recovery.sum()),
        "max_committed_gross_power_mw": float(committed_gross_power.max()),
        "max_realised_gross_power_mw": float(realised_gross_power.max()),
        "slots_with_shortfall": int((shortfall > tolerance).sum()),
        "slots_with_schedule_gap": int((schedule_gaps > tolerance).sum()),
        "activation_during_charge_slots": int((np.minimum(plan["p_ch"], delivered / delta) > tolerance).sum()),
        "recovery_during_discharge_slots": int((np.minimum(recovery, plan["p_dis"]) > tolerance).sum()),
        "recovery_during_activation_slots": int((np.minimum(recovery, delivered / delta) > tolerance).sum()),
        "recovery_overlap_slots": int((np.minimum(plan["p_dis"] + delivered / delta,
                                                   plan["p_ch"] + recovery) > tolerance).sum()),
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


def activation_hand_check(config: dict) -> dict:
    """Two-hour case: reserve activation in hour one is restored at a known energy cost."""
    case = config["activation_hand_check"]
    n_slots = len(case["day_ahead_prices"])
    arm = {"power_mw": case["power_mw"], "energy_mwh": case["energy_mwh"], "market_share_cap": 1.0}
    storage = {
        "round_trip_efficiency": case["round_trip_efficiency"],
        "reserve_delivery_hours": config["storage"]["reserve_delivery_hours"],
        "slot_hours": config["protocol"]["slot_hours"],
        "initial_soc_share": case["initial_soc_mwh"] / case["energy_mwh"],
        "soc_min_share": 0.0, "soc_max_share": 1.0,
        "soc_tolerance_mwh": config["solver"]["soc_tolerance_mwh"],
        "recovery_constraint_tolerance_mwh": config["solver"]["recovery_constraint_tolerance_mwh"],
    }
    realised = {
        "day_ahead_price": np.array(case["day_ahead_prices"]),
        "capacity_price": np.array(case["capacity_prices"]),
        "procured_mw": np.ones(len(case["capacity_prices"])),
        "hour_of_slot": np.arange(n_slots) // 4,
        "activation_share": np.array(case["activation_share"]),
        "activation_price": np.array(case["activation_prices"]),
        "shortfall_price": np.array(case["imbalance_prices"]),
    }
    plan = solve_activation_oracle(realised, arm, storage, config["solver"])
    settled = settle_day(plan, realised, arm, storage, config["solver"])
    error = abs(settled["total_eur"] - case["expected_profit_eur"])
    terminal = abs(settled["terminal_soc_deviation_mwh"])
    if error > case["tolerance_eur"]:
        raise ValueError(f"Activation hand-check profit {settled['total_eur']} != {case['expected_profit_eur']}")
    if terminal > config["solver"]["soc_tolerance_mwh"]:
        raise ValueError(f"Activation hand-check terminal SOC deviates by {terminal}")
    return {
        "profit_eur": settled["total_eur"], "expected_profit_eur": case["expected_profit_eur"],
        "profit_abs_error": float(error), "terminal_soc_abs_error": float(terminal),
        "committed_mw_hours": settled["committed_mw_hours"],
        "delivered_activation_mwh": settled["delivered_activation_mwh"],
        "recovery_grid_mwh": settled["recovery_grid_mwh"],
    }
