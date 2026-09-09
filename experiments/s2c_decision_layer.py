#!/usr/bin/env python3
"""Run the S2-C decomposition of reserve co-optimisation and activation foresight."""

import argparse
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

from f01_lear_dk1.settlement import build_panels, settlement_protocol
from s2c_decision_layer.arms import ARM_ORDER, solve_arms


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


def storage_settings(config: dict) -> dict:
    return dict(
        config["storage"],
        slot_hours=config["protocol"]["slot_hours"],
        soc_tolerance_mwh=config["solver"]["soc_tolerance_mwh"],
        recovery_constraint_tolerance_mwh=config["solver"]["recovery_constraint_tolerance_mwh"],
    )


def load_json(path: str) -> dict:
    return json.loads((REPOSITORY_ROOT / path).read_text(encoding="utf-8"))


def load_forecasts(config: dict, settled_days: list) -> dict:
    paths = {
        "B2_F01": config["comparison"]["f01_forecasts_csv"],
        "B2_F08": config["comparison"]["f08_forecasts_csv"],
    }
    result = {}
    expected = set(settled_days)
    for name, path in paths.items():
        frame = pd.read_csv(REPOSITORY_ROOT / path)
        frame = frame[frame["delivery_day"].isin(settled_days)].copy()
        if set(frame["delivery_day"]) != expected:
            raise ValueError(f"{name}: forecast days differ from settlement days")
        result[name] = frame
    return result


def arm_summary(frame: pd.DataFrame) -> dict:
    sums = ["total_eur", "energy_eur", "capacity_eur", "activation_eur",
            "shortfall_cost_eur", "recovery_cost_eur", "committed_mw_hours"]
    return {
        "days": int(frame["delivery_day"].nunique()),
        **{column: float(frame[column].sum()) for column in sums},
        "exclusivity_violations": int(frame["exclusivity_violations"].sum()),
        "slots_with_shortfall": int(frame["slots_with_shortfall"].sum()),
        "recovery_overlap_slots": int(frame["recovery_overlap_slots"].sum()),
        "terminal_soc_deviation_mwh_max_abs": float(
            frame["terminal_soc_deviation_mwh"].abs().max()),
    }


def reference_totals(config: dict) -> dict:
    s2 = load_json(config["comparison"]["s2_summary_json"])
    f01 = load_json(config["comparison"]["f01_summary_json"])
    f08 = load_json(config["comparison"]["f08_summary_json"])
    return {
        "B1": float(s2["strategies"]["B1"]["total_eur"]["total"]),
        "F01_LEAR": float(f01["settlement"]["total_eur"]["total"]),
        "F08_LightGBM": float(f08["settlement"]["total_eur"]),
        "Oracle": float(s2["strategies"]["Oracle"]["total_eur"]["total"]),
    }


def decomposition(arms: dict, references: dict) -> dict:
    perfect = arms["perfect_quarter_price"]["total_eur"]
    price_reserve = arms["oracle_price_reserve"]["total_eur"]
    oracle = references["Oracle"]
    decision_layer = oracle - perfect
    reserve = price_reserve - perfect
    activation = oracle - price_reserve
    return {
        "decision_layer_eur": decision_layer,
        "reserve_cooptimisation_eur": reserve,
        "activation_foresight_eur": activation,
        "reserve_cooptimisation_share": reserve / decision_layer,
        "activation_foresight_share": activation / decision_layer,
        "definition": ("oracle_price_reserve minus perfect_quarter_price is reserve "
                       "co-optimisation; Oracle minus oracle_price_reserve is perfect activation "
                       "information"),
    }


def comparison(arms: dict, references: dict) -> dict:
    values = {**references, **{name: item["total_eur"] for name, item in arms.items()}}
    oracle = references["Oracle"]
    return {
        "total_eur": values,
        "share_of_oracle": {name: value / oracle for name, value in values.items()},
        "B2_F01_minus_F01_eur": values["B2_F01"] - values["F01_LEAR"],
        "B2_F08_minus_F08_eur": values["B2_F08"] - values["F08_LightGBM"],
        "B2_F01_over_oracle_price_reserve": values["B2_F01"] / values["oracle_price_reserve"],
        "B2_F08_over_oracle_price_reserve": values["B2_F08"] / values["oracle_price_reserve"],
        "B2_order": "B2_F08_above_B2_F01" if values["B2_F08"] > values["B2_F01"]
        else "B2_F01_at_or_above_B2_F08",
    }


def validate(config: dict, arms: dict, references: dict) -> dict:
    expected = config["acceptance"]
    tolerance = config["solver"]["soc_tolerance_mwh"]
    perfect = arms["perfect_quarter_price"]["total_eur"]
    price_reserve = arms["oracle_price_reserve"]["total_eur"]
    if abs(perfect - expected["perfect_quarter_price_total_eur"]) > 1e-6:
        raise ValueError(f"perfect_quarter_price mismatch: {perfect}")
    if max(item["terminal_soc_deviation_mwh_max_abs"] for item in arms.values()) >= tolerance:
        raise ValueError("Terminal SOC tolerance failed")
    if not perfect <= price_reserve <= references["Oracle"]:
        raise ValueError("oracle_price_reserve is outside perfect-price/Oracle bounds")
    if arms["B2_F01"]["total_eur"] > price_reserve or arms["B2_F08"]["total_eur"] > price_reserve:
        raise ValueError("A forecast arm exceeds oracle_price_reserve")
    return {
        "perfect_quarter_price_matches_frozen_result": True,
        "terminal_soc_strictly_below_tolerance": True,
        "oracle_price_reserve_within_bounds": True,
        "forecast_arms_do_not_exceed_oracle_price_reserve": True,
    }


def align_daily_columns(config: dict, daily: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(REPOSITORY_ROOT / config["comparison"]["f01_daily_csv"])
    diagnostics = reference[["delivery_day", "aggregation_quarter_total_eur",
                             "aggregation_hourly_total_eur", "aggregation_cost_eur"]]
    result = daily.merge(diagnostics, on="delivery_day", validate="many_to_one")
    if set(result.columns) != set(reference.columns):
        raise ValueError("S2-C daily columns differ from F01 daily_results.csv")
    return result[reference.columns]


def write_outputs(config: dict, summary: dict, daily: pd.DataFrame) -> Path:
    directory = REPOSITORY_ROOT / config["output"]["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    daily.to_csv(directory / config["output"]["daily_csv"], index=False)
    with (directory / config["output"]["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    return directory


def main() -> None:
    config = yaml.safe_load(parse_arguments().config.read_text(encoding="utf-8"))
    np.random.seed(config["protocol"]["random_seed"])
    panels = build_panels(REPOSITORY_ROOT / config["data"]["root"], config)
    settled_days, threshold = settlement_protocol(panels, config)
    forecasts = load_forecasts(config, settled_days)
    storage = storage_settings(config)
    daily = solve_arms(panels, settled_days, forecasts, threshold,
                       config["storage"]["arms"]["main"], storage, config["solver"])
    grouped = {name: arm_summary(daily[daily["strategy"] == name]) for name in ARM_ORDER}
    references = reference_totals(config)
    summary = {
        "environment_fingerprint": environment_fingerprint(),
        "random_seed": config["protocol"]["random_seed"],
        "settled_test_days": len(settled_days),
        "reserve_threshold_eur_per_mw": threshold,
        "arms": grouped,
        "decomposition": decomposition(grouped, references),
        "comparison": comparison(grouped, references),
        "acceptance": validate(config, grouped, references),
    }
    directory = write_outputs(config, summary, align_daily_columns(config, daily))
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True), flush=True)
    print(f"S2-C results written to {directory}", flush=True)


if __name__ == "__main__":
    main()
