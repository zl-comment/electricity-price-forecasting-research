"""Sparse linear programmes for the frozen S3-B three-stage decision protocol."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans

from epf_harness.dk1_storage import efficiencies


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Layout:
    """Contiguous variable blocks for one sparse stochastic programme."""

    hours: int
    slots: int
    clusters: int
    scenarios: int
    risk: bool
    proxy: bool

    def __post_init__(self) -> None:
        offsets = [self.hours]
        widths = [self.clusters * self.slots] * 3 + [self.scenarios * self.slots] * 3
        for width in widths:
            offsets.append(offsets[-1] + width)
        object.__setattr__(self, "offsets", tuple(offsets))
        object.__setattr__(self, "overlap0", offsets[-1])
        overlap_end = offsets[-1] + int(self.proxy) * self.clusters * self.hours
        object.__setattr__(self, "overlap_end", overlap_end)
        object.__setattr__(self, "eta", overlap_end if self.risk else -1)
        object.__setattr__(self, "u0", overlap_end + int(self.risk))
        object.__setattr__(self, "size", overlap_end + int(self.risk) * (1 + self.scenarios))

    def r(self, hour: int) -> int:
        return hour

    def cluster(self, block: int, cluster: int, slot: int) -> int:
        return self.offsets[block] + cluster * self.slots + slot

    def scenario(self, block: int, scenario: int, slot: int) -> int:
        return self.offsets[3 + block] + scenario * self.slots + slot

    def overlap(self, cluster: int, hour: int) -> int:
        return self.overlap0 + cluster * self.hours + hour


class SparseRows:
    """Accumulate sparse rows without materialising a dense constraint matrix."""

    def __init__(self, columns: int) -> None:
        self.columns = columns
        self.row = []
        self.col = []
        self.data = []
        self.bound = []

    def add(self, coefficients, bound: float) -> None:
        row = len(self.bound)
        for column, value in coefficients:
            if value != 0.0:
                self.row.append(row)
                self.col.append(column)
                self.data.append(float(value))
        self.bound.append(float(bound))

    def matrix(self):
        shape = (len(self.bound), self.columns)
        matrix = coo_matrix((self.data, (self.row, self.col)), shape=shape).tocsr()
        return matrix, np.asarray(self.bound)


def cluster_capacity(capacity_price: np.ndarray, historical_capacity: np.ndarray,
                     config: dict) -> np.ndarray:
    """Cluster capacity paths after scaling each hour by its F(D) population SD."""
    paths = np.asarray(capacity_price, dtype=float)
    history = np.asarray(historical_capacity, dtype=float)
    _require(paths.ndim == 2 and history.ndim == 2, "Capacity paths must be matrices")
    _require(paths.shape[1] == history.shape[1], "Capacity history hour count differs")
    scale = np.std(history, axis=0)
    _require(np.isfinite(paths).all() and np.isfinite(scale).all(), "Capacity scaling is non-finite")
    _require((scale > 0.0).all(), "An F(D) capacity-price standard deviation is zero")
    clusters = min(int(config["tree"]["clusters"]), len(paths))
    model = KMeans(n_clusters=clusters, random_state=int(config["tree"]["seed"]),
                   n_init=int(config["tree"]["n_init"]))
    return model.fit_predict(paths / scale).astype(int)


def _validate_inputs(scenarios: dict, probabilities: np.ndarray,
                     labels: np.ndarray, procured_mw: np.ndarray) -> tuple:
    shares = np.asarray(scenarios["activation_share"], dtype=float)
    day_ahead = np.asarray(scenarios["day_ahead_price"], dtype=float)
    capacity = np.asarray(scenarios["capacity_price"], dtype=float)
    scenario_count, slots = shares.shape
    hours = len(procured_mw)
    expected_slot_shape = (scenario_count, slots)
    _require(day_ahead.shape == expected_slot_shape, "Day-ahead scenario shape differs")
    _require(capacity.shape == (scenario_count, hours), "Capacity scenario shape differs")
    for name in ("activation_price", "shortfall_price"):
        _require(np.asarray(scenarios[name]).shape == expected_slot_shape,
                 f"{name} scenario shape differs")
    _require(len(probabilities) == scenario_count and np.isclose(probabilities.sum(), 1.0),
             "Scenario probabilities are invalid")
    _require(len(labels) == scenario_count and labels.min() == 0,
             "Scenario cluster labels are invalid")
    _require(set(labels.tolist()) == set(range(int(labels.max()) + 1)),
             "Scenario cluster labels are not contiguous")
    _require(np.isfinite(day_ahead).all() and np.isfinite(capacity).all(),
             "Price scenarios contain non-finite values")
    _require(np.isfinite(shares).all() and (shares >= 0.0).all(),
             "Activation-share scenarios are invalid")
    activation_price = np.asarray(scenarios["activation_price"], dtype=float)
    _require(not (np.isnan(activation_price) & (shares > 0.0)).any(),
             "Activation price is missing for a positive activation share")
    _require(np.isfinite(np.asarray(scenarios["shortfall_price"])).all(),
             "Shortfall-price scenarios contain non-finite values")
    return scenario_count, slots, hours, int(labels.max()) + 1


def _bounds(layout: Layout, procured_mw: np.ndarray, arm: dict, storage: dict,
            fixed_reserve: np.ndarray, proxy_forecast_mwh: np.ndarray) -> list:
    power = float(arm["power_mw"])
    soc_min = float(arm["energy_mwh"] * storage["soc_min_share"])
    soc_max = float(arm["energy_mwh"] * storage["soc_max_share"])
    soc_0 = float(arm["energy_mwh"] * storage["initial_soc_share"])
    if fixed_reserve is None:
        ceiling = np.minimum(power, arm["market_share_cap"] * procured_mw)
        bounds = [(0.0, float(value)) for value in ceiling]
    else:
        fixed = np.asarray(fixed_reserve, dtype=float)
        _require(fixed.shape == (layout.hours,), "Fixed reserve shape differs")
        bounds = [(float(value), float(value)) for value in fixed]
    for block in range(3):
        cell = (0.0, power) if block < 2 else (soc_min, soc_max)
        bounds.extend([cell] * (layout.clusters * layout.slots))
    for block in range(3):
        cell = (0.0, power) if block == 0 else ((0.0, power * storage["slot_hours"])
                                                if block == 1 else (soc_min, soc_max))
        bounds.extend([cell] * (layout.scenarios * layout.slots))
    for cluster in range(layout.clusters):
        bounds[layout.cluster(2, cluster, layout.slots - 1)] = (soc_0, soc_0)
    for scenario in range(layout.scenarios):
        bounds[layout.scenario(2, scenario, layout.slots - 1)] = (soc_0, soc_0)
    if layout.proxy:
        bounds.extend((0.0, float(value)) for _ in range(layout.clusters)
                      for value in proxy_forecast_mwh)
    if layout.risk:
        bounds.extend([(None, None)] + [(0.0, None)] * layout.scenarios)
    return bounds


def _base_constraints(layout: Layout, rows_eq: SparseRows, rows_ub: SparseRows,
                      hour_of_slot: np.ndarray, arm: dict, storage: dict) -> None:
    delta = float(storage["slot_hours"])
    eta_c, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_0 = float(arm["energy_mwh"] * storage["initial_soc_share"])
    duration = float(storage["reserve_delivery_hours"])
    for cluster in range(layout.clusters):
        for slot, hour in enumerate(hour_of_slot):
            p_ch = layout.cluster(0, cluster, slot)
            p_dis = layout.cluster(1, cluster, slot)
            soc = layout.cluster(2, cluster, slot)
            previous = [] if slot == 0 else [(layout.cluster(2, cluster, slot - 1), -1.0)]
            rows_eq.add([(soc, 1.0), (p_ch, -delta * eta_c),
                         (p_dis, delta / eta_d), *previous], soc_0 if slot == 0 else 0.0)
            rows_ub.add([(p_ch, 1.0), (p_dis, 1.0), (layout.r(int(hour)), 1.0)],
                        arm["power_mw"])
            rows_ub.add([(layout.r(int(hour)), duration / eta_d), (soc, -1.0)], 0.0)


def _actual_constraints(layout: Layout, rows_eq: SparseRows, rows_ub: SparseRows,
                        scenarios: dict, labels: np.ndarray, hour_of_slot: np.ndarray,
                        arm: dict, storage: dict, delivery_mode: str) -> None:
    delta = float(storage["slot_hours"])
    eta_c, eta_d = efficiencies(storage["round_trip_efficiency"])
    soc_0 = float(arm["energy_mwh"] * storage["initial_soc_share"])
    duration = float(storage["reserve_delivery_hours"])
    shares = np.asarray(scenarios["activation_share"], dtype=float)
    for scenario, cluster in enumerate(labels):
        for slot, hour in enumerate(hour_of_slot):
            p_ch = layout.cluster(0, int(cluster), slot)
            p_dis = layout.cluster(1, int(cluster), slot)
            recovery = layout.scenario(0, scenario, slot)
            delivered = layout.scenario(1, scenario, slot)
            soc = layout.scenario(2, scenario, slot)
            previous = [] if slot == 0 else [(layout.scenario(2, scenario, slot - 1), -1.0)]
            dynamics = [(soc, 1.0), (p_ch, -delta * eta_c), (p_dis, delta / eta_d),
                        (recovery, -delta * eta_c), (delivered, 1.0 / eta_d), *previous]
            rows_eq.add(dynamics, soc_0 if slot == 0 else 0.0)
            rows_ub.add([(p_ch, 1.0), (p_dis, 1.0), (recovery, 1.0),
                         (delivered, 1.0 / delta)], arm["power_mw"])
            rows_ub.add([(layout.r(int(hour)), duration / eta_d), (soc, -1.0),
                         (delivered, -1.0 / eta_d)], 0.0)
            delivery = [(delivered, 1.0),
                        (layout.r(int(hour)), -shares[scenario, slot])]
            if delivery_mode == "full":
                rows_eq.add(delivery, 0.0)
            else:
                _require(delivery_mode in {"economic", "delivery_first"},
                         f"Unknown delivery mode: {delivery_mode}")
                rows_ub.add(delivery, 0.0)


def _proxy_alignment_constraints(layout: Layout, rows_ub: SparseRows,
                                 hour_of_slot: np.ndarray, storage: dict) -> None:
    delta = float(storage["slot_hours"])
    for cluster in range(layout.clusters):
        for hour in range(layout.hours):
            coefficients = [(layout.overlap(cluster, hour), 1.0)]
            coefficients.extend(
                (layout.cluster(0, cluster, int(slot)), -delta)
                for slot in np.flatnonzero(hour_of_slot == hour))
            rows_ub.add(coefficients, 0.0)


def _profit_coefficients(layout: Layout, scenarios: dict, labels: np.ndarray,
                         hour_of_slot: np.ndarray, scenario: int,
                         storage: dict) -> list:
    delta = float(storage["slot_hours"])
    day_ahead = np.asarray(scenarios["day_ahead_price"])[scenario]
    capacity = np.asarray(scenarios["capacity_price"])[scenario]
    shares = np.asarray(scenarios["activation_share"])[scenario]
    activation = np.nan_to_num(np.asarray(scenarios["activation_price"])[scenario], nan=0.0)
    shortfall = np.asarray(scenarios["shortfall_price"])[scenario]
    cluster = int(labels[scenario])
    coefficients = []
    for hour in range(layout.hours):
        slots = np.flatnonzero(hour_of_slot == hour)
        reserve_value = capacity[hour] - float(np.sum(shortfall[slots] * shares[slots]))
        coefficients.append((layout.r(hour), reserve_value))
    for slot in range(layout.slots):
        coefficients.extend([
            (layout.cluster(0, cluster, slot), -delta * day_ahead[slot]),
            (layout.cluster(1, cluster, slot), delta * day_ahead[slot]),
            (layout.scenario(0, scenario, slot), -delta * shortfall[slot]),
            (layout.scenario(1, scenario, slot), activation[slot] + shortfall[slot]),
        ])
    return coefficients


def _objective_and_risk(layout: Layout, rows_ub: SparseRows, scenarios: dict,
                        labels: np.ndarray, hour_of_slot: np.ndarray,
                        probabilities: np.ndarray, chi: float, beta: float,
                        storage: dict) -> tuple:
    objective = np.zeros(layout.size)
    profits = []
    for scenario in range(layout.scenarios):
        coefficients = _profit_coefficients(
            layout, scenarios, labels, hour_of_slot, scenario, storage)
        profits.append(coefficients)
        for column, value in coefficients:
            objective[column] += (1.0 - chi) * probabilities[scenario] * value
        if layout.risk:
            risk_row = [(layout.eta, 1.0), (layout.u0 + scenario, -1.0)]
            risk_row.extend((column, -value) for column, value in coefficients)
            rows_ub.add(risk_row, 0.0)
    if layout.risk:
        objective[layout.eta] = chi
        objective[layout.u0:layout.u0 + layout.scenarios] = -chi * probabilities / (1.0 - beta)
    return objective, profits


def _add_proxy_alignment_value(objective: np.ndarray, layout: Layout,
                               probabilities: np.ndarray, labels: np.ndarray,
                               alignment_credit: float) -> None:
    cluster_probability = np.bincount(
        labels, weights=probabilities, minlength=layout.clusters)
    for cluster in range(layout.clusters):
        start = layout.overlap(cluster, 0)
        objective[start:start + layout.hours] = (
            float(alignment_credit) * cluster_probability[cluster])


def _delivery_objective(layout: Layout, probabilities: np.ndarray) -> np.ndarray:
    objective = np.zeros(layout.size)
    for scenario in range(layout.scenarios):
        start = layout.scenario(1, scenario, 0)
        objective[start:start + layout.slots] = probabilities[scenario]
    return objective


def _minimum_delivery(rows_ub: SparseRows, layout: Layout,
                      probabilities: np.ndarray, minimum: float) -> None:
    coefficients = []
    for scenario in range(layout.scenarios):
        coefficients.extend((layout.scenario(1, scenario, slot), -probabilities[scenario])
                            for slot in range(layout.slots))
    rows_ub.add(coefficients, -float(minimum))


def _solution(layout: Layout, result, objective: np.ndarray, market_objective: np.ndarray,
              profits: list, probabilities: np.ndarray, labels: np.ndarray,
              alignment_credit: float) -> dict:
    scenario_profit = np.asarray([
        sum(value * result.x[column] for column, value in coefficients)
        for coefficients in profits
    ])
    p_ch = result.x[layout.offsets[0]:layout.offsets[1]].reshape(layout.clusters, layout.slots)
    p_dis = result.x[layout.offsets[1]:layout.offsets[2]].reshape(layout.clusters, layout.slots)
    soc = result.x[layout.offsets[2]:layout.offsets[3]].reshape(layout.clusters, layout.slots)
    recovery = result.x[layout.offsets[3]:layout.offsets[4]].reshape(
        layout.scenarios, layout.slots)
    delivered = result.x[layout.offsets[4]:layout.offsets[5]].reshape(
        layout.scenarios, layout.slots)
    if layout.proxy:
        overlap = result.x[layout.overlap0:layout.overlap_end].reshape(
            layout.clusters, layout.hours)
        cluster_probability = np.bincount(
            labels, weights=probabilities, minlength=layout.clusters)
        forecast_alignment = float(cluster_probability @ overlap.sum(axis=1))
    else:
        overlap = np.zeros((layout.clusters, layout.hours))
        forecast_alignment = 0.0
    return {"success": True, "r_up": result.x[:layout.hours], "p_ch_cluster": p_ch,
            "p_dis_cluster": p_dis, "soc_cluster": soc, "recovery": recovery,
            "delivered": delivered, "forecast_overlap_cluster_mwh": overlap,
            "forecast_alignment_mwh": forecast_alignment,
            "alignment_credit_eur": float(alignment_credit) * forecast_alignment,
            "scenario_profit_eur": scenario_profit,
            "weighted_delivery_mwh": float(probabilities @ delivered.sum(axis=1)),
            "expected_profit_eur": float(probabilities @ scenario_profit),
            "market_objective_eur": float(market_objective @ result.x),
            "objective_eur": float(objective @ result.x), "solver_status": int(result.status),
            "solver_iterations": int(result.nit)}


def _clean_reserve(values: np.ndarray, procured_mw: np.ndarray,
                   arm: dict, solver: dict) -> np.ndarray:
    reserve = np.asarray(values, dtype=float)
    ceiling = np.minimum(float(arm["power_mw"]),
                         float(arm["market_share_cap"]) * np.asarray(procured_mw, dtype=float))
    cleaned = np.minimum(np.maximum(reserve, 0.0), ceiling)
    error = float(np.max(np.abs(cleaned - reserve)))
    _require(error <= float(solver["exclusivity_tolerance_mw"]),
             f"Reserve solution violates its bounds by {error}")
    return cleaned


def _solve_linprog(objective: np.ndarray, a_ub, b_ub: np.ndarray, a_eq, b_eq: np.ndarray,
                   bounds: list, solver: dict) -> tuple:
    """Solve the LP; only on HiGHS status 4 re-solve the identical LP with the frozen settings."""
    settings = [{"label": solver["method"], "method": solver["method"], "options": {}}]
    settings += list(solver["status4_retry"])
    for setting in settings:
        options = dict(setting["options"], time_limit=solver["time_limit_seconds"])
        result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds,
                         method=setting["method"], options=options)
        if result.status != 4:
            return result, setting["label"]
    return result, setting["label"]


def solve_stochastic(scenarios: dict, probabilities: np.ndarray, labels: np.ndarray,
                     procured_mw: np.ndarray, hour_of_slot: np.ndarray, arm: dict,
                     storage: dict, solver: dict, risk: dict, chi: float,
                     fixed_reserve: np.ndarray, delivery_mode: str,
                     objective_kind: str, minimum_delivery_mwh,
                     proxy_forecast_mwh=None, alignment_credit: float = 0.0) -> dict:
    """Solve one frozen three-stage or fixed-reserve two-stage stochastic programme."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    scenario_count, slots, hours, clusters = _validate_inputs(
        scenarios, probabilities, labels, np.asarray(procured_mw))
    _require(len(hour_of_slot) == slots and np.max(hour_of_slot) == hours - 1,
             "Slot-to-hour mapping differs")
    _require(0.0 <= chi <= 1.0, "Risk weight lies outside [0, 1]")
    beta = float(risk["cvar_level"])
    _require(0.0 < beta < 1.0, "CVaR level lies outside (0, 1)")
    _require(objective_kind in {"profit", "delivery"}, "Unknown stochastic objective")
    _require(float(alignment_credit) >= 0.0, "Proxy-alignment credit is negative")
    proxy_active = float(alignment_credit) > 0.0
    if proxy_active:
        proxy_forecast_mwh = np.asarray(proxy_forecast_mwh, dtype=float)
        _require(proxy_forecast_mwh.shape == (hours,), "Proxy forecast hour count differs")
        _require(np.isfinite(proxy_forecast_mwh).all()
                 and (proxy_forecast_mwh >= 0.0).all(), "Proxy forecast is invalid")
    else:
        _require(proxy_forecast_mwh is None,
                 "A proxy forecast was supplied without a positive alignment credit")
    layout = Layout(hours, slots, clusters, scenario_count,
                    chi > 0.0 and objective_kind == "profit", proxy_active)
    rows_eq, rows_ub = SparseRows(layout.size), SparseRows(layout.size)
    _base_constraints(layout, rows_eq, rows_ub, hour_of_slot, arm, storage)
    _actual_constraints(layout, rows_eq, rows_ub, scenarios, labels,
                        hour_of_slot, arm, storage, delivery_mode)
    market_objective, profits = _objective_and_risk(
        layout, rows_ub, scenarios, labels, hour_of_slot, probabilities, chi, beta, storage)
    objective = market_objective.copy()
    if layout.proxy:
        _proxy_alignment_constraints(layout, rows_ub, hour_of_slot, storage)
        _add_proxy_alignment_value(
            objective, layout, probabilities, labels, float(alignment_credit))
    if objective_kind == "delivery":
        objective = _delivery_objective(layout, probabilities)
    if minimum_delivery_mwh is not None:
        _minimum_delivery(rows_ub, layout, probabilities, minimum_delivery_mwh)
    a_ub, b_ub = rows_ub.matrix()
    a_eq, b_eq = rows_eq.matrix()
    scale = float(solver["objective_scale_eur"])
    result, setting = _solve_linprog(
        -objective / scale, a_ub, b_ub, a_eq, b_eq,
        _bounds(layout, procured_mw, arm, storage, fixed_reserve,
                proxy_forecast_mwh), solver)
    if not result.success:
        return {"success": False, "status": int(result.status), "message": result.message,
                "solver_setting": setting}
    solution = _solution(layout, result, objective, market_objective, profits,
                         probabilities, labels, float(alignment_credit))
    solution["r_up"] = _clean_reserve(solution["r_up"], procured_mw, arm, solver)
    solution["solver_setting"] = setting
    return {**solution, "labels": labels, "delivery_mode": delivery_mode,
            "objective_kind": objective_kind,
            "proxy_alignment_credit_eur_per_mwh": float(alignment_credit)}


def solve_delivery_first(scenarios: dict, probabilities: np.ndarray, labels: np.ndarray,
                         procured_mw: np.ndarray, hour_of_slot: np.ndarray, arm: dict,
                         storage: dict, solver: dict, risk: dict, chi: float,
                         fixed_reserve: np.ndarray, tolerance_mwh: float,
                         proxy_forecast_mwh=None, alignment_credit: float = 0.0) -> dict:
    """Solve the frozen 12:00 weighted-delivery lexicographic approximation."""
    first = solve_stochastic(scenarios, probabilities, labels, procured_mw, hour_of_slot,
                             arm, storage, solver, risk, chi, fixed_reserve,
                             "delivery_first", "delivery", None)
    if not first["success"]:
        return {**first, "delivery_first_status": first["status"]}
    minimum = first["weighted_delivery_mwh"] - 0.5 * float(tolerance_mwh)
    second = solve_stochastic(scenarios, probabilities, labels, procured_mw, hour_of_slot,
                              arm, storage, solver, risk, chi, fixed_reserve,
                              "delivery_first", "profit", minimum,
                              proxy_forecast_mwh, alignment_credit)
    if not second["success"]:
        return {**second, "delivery_first_status": first["solver_status"],
                "maximum_weighted_delivery_mwh": first["weighted_delivery_mwh"]}
    _require(second["weighted_delivery_mwh"] >= minimum - float(tolerance_mwh),
             "Second solve violates the delivery-first tolerance")
    return {**second, "delivery_first_status": first["solver_status"],
            "profit_status": second["solver_status"],
            "solver_setting": f'{first["solver_setting"]}+{second["solver_setting"]}',
            "maximum_weighted_delivery_mwh": first["weighted_delivery_mwh"],
            "delivery_shortfall_from_maximum_mwh": (
                first["weighted_delivery_mwh"] - second["weighted_delivery_mwh"])}


def deployment_plan(solution: dict, cluster: int = 0) -> dict:
    """Expose one nonanticipative energy plan in the frozen settle_day interface."""
    _require(solution["success"], "Cannot deploy an unsuccessful stochastic solution")
    return {"p_ch": solution["p_ch_cluster"][cluster].copy(),
            "p_dis": solution["p_dis_cluster"][cluster].copy(),
            "r_up": solution["r_up"].copy(),
            "soc_plan": solution["soc_cluster"][cluster].copy(),
            "planned_profit_eur": solution["expected_profit_eur"],
            "forecast_alignment_mwh": solution["forecast_alignment_mwh"],
            "alignment_credit_eur": solution["alignment_credit_eur"],
            "proxy_alignment_credit_eur_per_mwh": solution[
                "proxy_alignment_credit_eur_per_mwh"],
            "base_terminal_soc_deviation_mwh": 0.0,
            "solver_settings": solution["solver_setting"]}
