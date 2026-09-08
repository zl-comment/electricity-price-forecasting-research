"""S1 coupling checks: does the reserve gate actually cost the energy gate anything?"""

from pathlib import Path

import numpy as np
import pandas as pd

from epf_harness.dk1_audit import load_table

QUARTER_HOURS = 0.25


def add_delivery_day(frame: pd.DataFrame, time_column: str, timezone: str) -> pd.DataFrame:
    """Label every row with its civil delivery day. Conversion is DST-exact."""
    civil = frame[time_column].dt.tz_localize("UTC").dt.tz_convert(timezone)
    return frame.assign(delivery_day=civil.dt.strftime("%Y-%m-%d"))


def usable_delivery_days(window: dict) -> list:
    days = pd.date_range(window["first_delivery_day"], window["last_delivery_day"], freq="D")
    excluded = set(window["excluded_delivery_days"])
    kept = [d.strftime("%Y-%m-%d") for d in days if d.strftime("%Y-%m-%d") not in excluded]
    if len(kept) != len(days) - len(excluded):
        raise ValueError("Excluded delivery days are not all inside the window")
    return kept


def split_boundaries(config: dict, days: list) -> dict:
    split = config["split"]
    lookback = split["calibration_days"] + split["probability_calibration_days"] + split["clearing_days"]
    calendar = pd.date_range(
        config["window"]["first_delivery_day"], config["window"]["last_delivery_day"], freq="D"
    )
    declared = split["first_test_delivery_day"]
    derived = calendar[lookback].strftime("%Y-%m-%d")
    if derived != declared:
        raise ValueError(f"Configured first test day {declared} does not match derived {derived}")
    test = [d for d in days if d >= declared]
    return {
        "lookback_days": int(lookback),
        "first_test_delivery_day": declared,
        "usable_delivery_days": len(days),
        "lookback_delivery_days": len(days) - len(test),
        "test_delivery_days": len(test),
        "test_first": test[0],
        "test_last": test[-1],
    }


def capacity_envelope(arm: dict, storage: dict) -> list:
    """Mechanical crowd-out: what a commitment leaves for the energy gate."""
    power, energy = arm["power_mw"], arm["energy_mwh"]
    duration = storage["reserve_delivery_hours"]
    rows = []
    for level in storage["commitment_levels"]:
        committed = level * power
        rows.append(
            {
                "commitment_level": float(level),
                "committed_mw": float(committed),
                "day_ahead_discharge_headroom_mw": float(power - committed),
                "soc_locked_mwh": float(committed * duration),
                "cyclable_energy_mwh": float(energy - committed * duration),
                "energy_locked_share": float(committed * duration / energy),
            }
        )
    return rows


def greedy_fill(prices: np.ndarray, energy_mwh: float, slot_cap_mwh: float, cheapest: bool) -> float:
    """Cost of moving energy_mwh through the best-priced slots, each capped at slot_cap_mwh."""
    if energy_mwh > slot_cap_mwh * len(prices):
        raise ValueError("Requested energy exceeds what the day's slots can carry")
    order = np.argsort(prices) if cheapest else np.argsort(-prices)
    remaining, total = energy_mwh, 0.0
    for index in order:
        if remaining <= 0:
            break
        taken = min(slot_cap_mwh, remaining)
        total += taken * float(prices[index])
        remaining -= taken
    return total


def daily_energy_bound(prices: np.ndarray, arm: dict, efficiency: float) -> float:
    """Perfect-foresight single-cycle arbitrage bound. Ignores charge-before-discharge order."""
    slot_cap = arm["power_mw"] * QUARTER_HOURS
    discharged = greedy_fill(prices, arm["energy_mwh"], slot_cap, cheapest=False)
    charged = greedy_fill(prices, arm["energy_mwh"] / efficiency, slot_cap, cheapest=True)
    return discharged - charged


def crowd_out_table(prices: pd.DataFrame, capacity: pd.DataFrame, arm: dict, storage: dict) -> pd.DataFrame:
    """Per delivery day: full-day reserve income against the energy the commitment forfeits."""
    reserve = capacity.groupby("delivery_day")["UpPriceEUR"].agg(["sum", "max", "count"])
    rows = []
    for day, group in prices.groupby("delivery_day"):
        if day not in reserve.index:
            raise ValueError(f"Delivery day {day} has day-ahead prices but no capacity prices")
        reserve_eur = float(reserve.loc[day, "sum"]) * arm["power_mw"]
        energy_eur = daily_energy_bound(group["DayAheadPriceEUR"].to_numpy(), arm, storage["round_trip_efficiency"])
        rows.append(
            {
                "delivery_day": day,
                "hours": int(reserve.loc[day, "count"]),
                "quarters": int(len(group)),
                "reserve_full_day_eur": reserve_eur,
                "energy_one_cycle_eur": energy_eur,
                "crowd_out_eur": float(min(reserve_eur, energy_eur)),
                "reserve_dominates": bool(reserve_eur > energy_eur),
                "max_hourly_up_price_eur_per_mw": float(reserve.loc[day, "max"]),
            }
        )
    return pd.DataFrame(rows).sort_values("delivery_day").reset_index(drop=True)


def activation_table(activation: pd.DataFrame, arm: dict) -> pd.DataFrame:
    """Per delivery day, per MW committed: pro-rata activated energy and what it costs."""
    share = activation["aFRRUpMW"] / activation["UpProcuredMW"]
    if bool((share < 0).any()):
        raise ValueError("Negative pro-rata up-activation share")
    frame = activation.assign(
        activated_mwh_per_mw=share,
        activation_revenue_eur_per_mw=share * activation["aFRRVWAUpEUR"],
        forgone_day_ahead_eur_per_mw=share * activation["DayAheadPriceEUR"],
    )
    daily = frame.groupby("delivery_day")[
        ["activated_mwh_per_mw", "activation_revenue_eur_per_mw", "forgone_day_ahead_eur_per_mw"]
    ].sum()
    daily["net_activation_eur_per_mw"] = (
        daily["activation_revenue_eur_per_mw"] - daily["forgone_day_ahead_eur_per_mw"]
    )
    daily["activated_mwh_at_arm_power"] = daily["activated_mwh_per_mw"] * arm["power_mw"]
    daily["soc_throughput_share_of_energy"] = daily["activated_mwh_at_arm_power"] / arm["energy_mwh"]
    return daily.reset_index()


def stress_thresholds(capacity: pd.DataFrame, activation: pd.DataFrame, net_load: pd.DataFrame,
                      lookback_days: set, quantiles: dict) -> dict:
    """Thresholds fixed on the look-back period only. Frozen before any test-period result."""
    reserve = capacity[capacity["delivery_day"].isin(lookback_days)]["UpPriceEUR"]
    hourly_activation = (
        activation[activation["delivery_day"].isin(lookback_days)]
        .assign(hour=lambda f: f["TimeUTC"].dt.floor("1H"))
        .groupby("hour")["aFRRUpMW"]
        .sum()
    )
    load = net_load[net_load["delivery_day"].isin(lookback_days)]["net_load_mwh"]
    if min(len(reserve), len(hourly_activation), len(load)) == 0:
        raise ValueError("Look-back period is empty for at least one stress signal")
    return {
        "lookback_delivery_days": len(lookback_days),
        "reserve_price_spike_eur_per_mw": float(reserve.quantile(quantiles["reserve_price_spike_quantile"])),
        "activation_intensity_mwh_per_hour": float(
            hourly_activation.quantile(quantiles["activation_intensity_quantile"])
        ),
        "net_load_high_mwh": float(load.quantile(quantiles["net_load_high_quantile"])),
        "net_load_low_mwh": float(load.quantile(quantiles["net_load_low_quantile"])),
    }


def build_net_load(root: Path, config: dict) -> pd.DataFrame:
    settings = config["datasets"]["production_consumption_settlement"]
    columns = config["net_load"]
    frame = pd.read_csv(root / settings["csv"], parse_dates=[settings["time_column"]])
    frame = frame[frame["PriceArea"] == config["data"]["primary_price_area"]]
    renewable = frame[columns["wind_columns"] + columns["solar_columns"]].sum(axis=1)
    frame = frame.assign(net_load_mwh=frame[columns["consumption_column"]] - renewable)
    return add_delivery_day(
        frame[[settings["time_column"], "net_load_mwh"]].rename(columns={settings["time_column"]: "TimeUTC"}),
        "TimeUTC",
        config["protocol"]["civil_timezone"],
    )


def load_coupled_frames(root: Path, config: dict) -> dict:
    """Day-ahead prices, hourly capacity and the 15-minute activation join, all on UTC keys."""
    area = config["data"]["primary_price_area"]
    timezone = config["protocol"]["civil_timezone"]
    prices = load_table(root, config["datasets"]["day_ahead_prices"])
    prices = prices[prices["PriceArea"] == area][["TimeUTC", "DayAheadPriceEUR"]]
    capacity = load_table(root, config["datasets"]["afrr_capacity_market"])[
        ["TimeUTC", "UpProcuredMW", "UpPriceEUR"]
    ]
    imbalance = load_table(root, config["datasets"]["imbalance_and_activation"])[
        ["TimeUTC", "aFRRUpMW", "aFRRVWAUpEUR"]
    ]
    hourly = capacity.assign(hour=capacity["TimeUTC"]).drop(columns=["TimeUTC"])
    activation = imbalance.merge(prices, on="TimeUTC", how="inner")
    activation = activation.assign(hour=activation["TimeUTC"].dt.floor("1H")).merge(hourly, on="hour", how="inner")
    if len(activation) != len(imbalance):
        raise ValueError(f"Activation join changed row count: {len(imbalance)} -> {len(activation)}")
    return {
        "prices": add_delivery_day(prices, "TimeUTC", timezone),
        "capacity": add_delivery_day(capacity, "TimeUTC", timezone),
        "activation": add_delivery_day(activation.drop(columns=["hour"]), "TimeUTC", timezone),
        "net_load": build_net_load(root, config),
    }


def restrict_to_days(frames: dict, days: list) -> dict:
    kept = set(days)
    return {name: frame[frame["delivery_day"].isin(kept)].reset_index(drop=True) for name, frame in frames.items()}


def coupling_findings(crowd_out: pd.DataFrame, acceptance: dict) -> list:
    """A finding here means the premise of the research question failed a check."""
    reserve_days = int(crowd_out["reserve_dominates"].sum())
    energy_days = int(len(crowd_out) - reserve_days)
    zero_share = float((crowd_out["crowd_out_eur"] <= 0).mean())
    findings = []
    if reserve_days < acceptance["min_days_with_reserve_dominant"]:
        findings.append(("crowd_out", "reserve_never_dominates", str(reserve_days)))
    if energy_days < acceptance["min_days_with_energy_dominant"]:
        findings.append(("crowd_out", "energy_never_dominates", str(energy_days)))
    if zero_share > acceptance["max_zero_crowd_out_day_share"]:
        findings.append(("crowd_out", "crowd_out_almost_always_zero", f"{zero_share:.4f}"))
    return findings
