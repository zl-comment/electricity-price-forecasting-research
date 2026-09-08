"""Feed F01/F08 forecasts into the frozen S2 activation-aware settlement layer."""

from pathlib import Path

import numpy as np
import pandas as pd

from epf_harness.dk1_audit import load_table
from epf_harness.dk1_coupling import add_delivery_day, split_boundaries, usable_delivery_days
from epf_harness.dk1_storage import exclusivity_violations, settle_day, solve_day, threshold_rule


def build_panels(root: Path, config: dict) -> dict:
    """Build the same realised per-day panels as S2."""
    area, timezone = config["data"]["primary_price_area"], config["protocol"]["civil_timezone"]
    prices = load_table(root, config["datasets"]["day_ahead_prices"])
    prices = add_delivery_day(prices[prices["PriceArea"] == area], "TimeUTC", timezone)
    capacity = add_delivery_day(load_table(root, config["datasets"]["afrr_capacity_market"]),
                                "TimeUTC", timezone)
    imbalance = add_delivery_day(load_table(root, config["datasets"]["imbalance_and_activation"]),
                                 "TimeUTC", timezone)
    panels = {}
    for day, quarter in prices.groupby("delivery_day"):
        hourly = capacity[capacity["delivery_day"] == day]
        activation = imbalance[imbalance["delivery_day"] == day]
        n_slots, n_hours = len(quarter), len(hourly)
        if n_slots != 4 * n_hours or len(activation) != n_slots:
            raise ValueError(f"{day}: inconsistent realised panel lengths")
        procured = hourly["UpProcuredMW"].to_numpy()
        panels[day] = {
            "day_ahead_price": quarter["DayAheadPriceEUR"].to_numpy(),
            "capacity_price": hourly["UpPriceEUR"].to_numpy(),
            "procured_mw": procured, "hour_of_slot": np.arange(n_slots) // 4,
            "activation_share": activation["aFRRUpMW"].to_numpy() / np.repeat(procured, 4),
            "activation_price": activation["aFRRVWAUpEUR"].to_numpy(),
            "shortfall_price": activation[config["settlement"]["shortfall_price_column"]].to_numpy(),
        }
    return panels


def resize(series: np.ndarray, length: int) -> np.ndarray:
    """Use S2's frozen DST policy: truncate, or repeat the last value."""
    if len(series) == length:
        return series
    if len(series) > length:
        return series[:length]
    return np.concatenate([series, np.repeat(series[-1], length - len(series))])


def settlement_protocol(panels: dict, config: dict) -> tuple:
    """Return S2's 211 common test days and its look-back capacity threshold."""
    usable = usable_delivery_days(config["window"])
    split = split_boundaries(config, usable)
    available = set(usable) & set(panels)
    lag = config["persistence"]["day_ahead_lag_days"]
    pairs = [(day, (pd.Timestamp(day) - pd.Timedelta(days=lag)).strftime("%Y-%m-%d"))
             for day in sorted(available)]
    test = [day for day, previous in pairs
            if day >= split["first_test_delivery_day"] and previous in available]
    lookback = [day for day in sorted(available) if day < split["first_test_delivery_day"]]
    prices = np.concatenate([panels[day]["capacity_price"] for day in lookback])
    threshold = float(np.quantile(prices, config["reserve_rule"]["threshold_quantile"]))
    acceptance = config["acceptance"]
    if threshold != acceptance["reserve_threshold_eur_per_mw"]:
        raise ValueError(f"Unexpected reserve threshold: {threshold}")
    if len(test) != acceptance["settled_test_days"]:
        raise ValueError(f"Unexpected settlement day count: {len(test)}")
    return test, threshold


def _forecast_vector(forecasts: pd.DataFrame, day: str, target: str) -> np.ndarray:
    rows = forecasts[(forecasts["delivery_day"] == day) & (forecasts["target"] == target)].sort_values("hour")
    values = rows["forecast"].to_numpy(dtype=float)
    if len(values) != 24 or not np.isfinite(values).all() or rows["hour"].tolist() != list(range(24)):
        raise ValueError(f"{day} {target}: expected 24 finite ordered forecasts")
    return values


def forecast_driven_arm(panels: dict, forecasts: pd.DataFrame, threshold: float, arm: dict,
                        storage: dict, solver: dict) -> list:
    """Plan with model prices, then settle actual activation and paid SOC recovery."""
    rows = []
    for day in sorted(forecasts["delivery_day"].unique()):
        panel = panels[day]
        day_ahead = np.repeat(_forecast_vector(forecasts, day, "day_ahead_price"), 4)
        day_ahead = resize(day_ahead, len(panel["day_ahead_price"]))
        capacity = resize(_forecast_vector(forecasts, day, "capacity_price"),
                          len(panel["capacity_price"]))
        fixed = threshold_rule(capacity, threshold, arm, panel["procured_mw"])
        plan = solve_day(day_ahead, capacity, panel["procured_mw"], panel["hour_of_slot"],
                         arm, storage, solver, fixed)
        settled = settle_day(plan, panel, arm, storage, solver)
        rows.append({"delivery_day": day, "planned_profit_eur": plan["planned_profit_eur"],
                     "base_terminal_soc_deviation_mwh": plan["base_terminal_soc_deviation_mwh"],
                     "exclusivity_violations": exclusivity_violations(
                         plan, solver["exclusivity_tolerance_mw"]), **settled})
    return rows


def _hourly_broadcast(panel: dict) -> np.ndarray:
    prices, hours = panel["day_ahead_price"], panel["hour_of_slot"]
    means = np.asarray([prices[hours == hour].mean() for hour in range(len(panel["capacity_price"]))])
    return np.repeat(means, 4)


def _oracle_difference_day(day: str, panel: dict, threshold: float, arm: dict,
                           storage: dict, solver: dict) -> dict:
    fixed = threshold_rule(panel["capacity_price"], threshold, arm, panel["procured_mw"])
    totals = {}
    for name, prices in (("quarter", panel["day_ahead_price"]), ("hourly", _hourly_broadcast(panel))):
        plan = solve_day(prices, panel["capacity_price"], panel["procured_mw"],
                         panel["hour_of_slot"], arm, storage, solver, fixed)
        totals[name] = settle_day(plan, panel, arm, storage, solver)["total_eur"]
    return {"delivery_day": day, "aggregation_quarter_total_eur": totals["quarter"],
            "aggregation_hourly_total_eur": totals["hourly"],
            "aggregation_cost_eur": totals["quarter"] - totals["hourly"]}


def aggregation_cost(panels: dict, threshold: float, arm: dict, storage: dict,
                     solver: dict) -> dict:
    """Measure hourly aggregation by a realised-price oracle difference on common days."""
    daily = [_oracle_difference_day(day, panels[day], threshold, arm, storage, solver)
             for day in sorted(panels)]
    frame = pd.DataFrame(daily)
    quarter = float(frame["aggregation_quarter_total_eur"].sum())
    hourly = float(frame["aggregation_hourly_total_eur"].sum())
    cost = quarter - hourly
    return {"quarter_total_eur": quarter, "hourly_total_eur": hourly,
            "aggregation_cost_eur": cost, "aggregation_cost_share": float(cost / abs(quarter)),
            "daily": frame}
