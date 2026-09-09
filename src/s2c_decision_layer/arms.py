"""Construct and settle the four S2-C arms on the frozen S2 feasible set."""

import numpy as np
import pandas as pd

from epf_harness.dk1_storage import exclusivity_violations, settle_day, solve_day, threshold_rule
from f01_lear_dk1.settlement import resize


ARM_ORDER = ("B2_F01", "B2_F08", "oracle_price_reserve", "perfect_quarter_price")


def _forecast_vector(forecasts: pd.DataFrame, day: str, target: str) -> np.ndarray:
    rows = forecasts[(forecasts["delivery_day"] == day) & (forecasts["target"] == target)]
    rows = rows.sort_values("hour")
    values = rows["forecast"].to_numpy(dtype=float)
    if len(values) != 24 or rows["hour"].tolist() != list(range(24)):
        raise ValueError(f"{day} {target}: expected 24 ordered forecasts")
    if not np.isfinite(values).all():
        raise ValueError(f"{day} {target}: forecasts must be finite")
    return values


def _believed_prices(forecasts: pd.DataFrame, day: str, panel: dict) -> tuple:
    day_ahead = np.repeat(_forecast_vector(forecasts, day, "day_ahead_price"), 4)
    day_ahead = resize(day_ahead, len(panel["day_ahead_price"]))
    capacity = resize(_forecast_vector(forecasts, day, "capacity_price"),
                      len(panel["capacity_price"]))
    return day_ahead, capacity


def _settled_row(name: str, day: str, panel: dict, prices: np.ndarray,
                 capacity_prices: np.ndarray, fixed_reserve: np.ndarray,
                 arm: dict, storage: dict, solver: dict) -> dict:
    plan = solve_day(prices, capacity_prices, panel["procured_mw"], panel["hour_of_slot"],
                     arm, storage, solver, fixed_reserve)
    settled = settle_day(plan, panel, arm, storage, solver)
    return {
        "delivery_day": day,
        "strategy": name,
        "planned_profit_eur": plan["planned_profit_eur"],
        "base_terminal_soc_deviation_mwh": plan["base_terminal_soc_deviation_mwh"],
        "exclusivity_violations": exclusivity_violations(
            plan, solver["exclusivity_tolerance_mw"]),
        **settled,
    }


def solve_arms(panels: dict, settled_days: list, forecasts: dict, threshold: float,
               arm: dict, storage: dict, solver: dict) -> pd.DataFrame:
    """Solve forecast/free-reserve, realised/free-reserve, and realised/threshold arms."""
    rows = []
    for day in settled_days:
        panel = panels[day]
        for name in ("B2_F01", "B2_F08"):
            prices, capacity = _believed_prices(forecasts[name], day, panel)
            rows.append(_settled_row(name, day, panel, prices, capacity, None,
                                     arm, storage, solver))
        rows.append(_settled_row(
            "oracle_price_reserve", day, panel, panel["day_ahead_price"],
            panel["capacity_price"], None, arm, storage, solver))
        fixed = threshold_rule(panel["capacity_price"], threshold, arm, panel["procured_mw"])
        rows.append(_settled_row(
            "perfect_quarter_price", day, panel, panel["day_ahead_price"],
            panel["capacity_price"], fixed, arm, storage, solver))
    result = pd.DataFrame(rows)
    counts = result.groupby("strategy")["delivery_day"].nunique().to_dict()
    expected = {name: len(settled_days) for name in ARM_ORDER}
    if counts != expected:
        raise ValueError(f"Unexpected per-arm day counts: {counts}")
    return result
