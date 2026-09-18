"""Task-authorised DK1 reconstructions of published decision structures."""

from hashlib import sha256
from pathlib import Path

import numpy as np

from f01_lear_dk1.settlement import resize
from s3_forecast_side.panel import window_days


TARGETS = ["day_ahead_price", "capacity_price", "activation_volume", "activation_price"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hourly(values: np.ndarray, hour_of_slot: np.ndarray, hours: int = 24) -> np.ndarray:
    means = np.asarray([np.mean(values[hour_of_slot == hour])
                        for hour in range(int(hour_of_slot.max()) + 1)])
    return resize(means, hours)


def _panel_paths(panels: dict, days: list) -> dict:
    capacity = np.stack([resize(panels[day]["capacity_price"], 24) for day in days])
    day_ahead = np.stack([_hourly(panels[day]["day_ahead_price"],
                                  panels[day]["hour_of_slot"]) for day in days])
    activation_price = np.stack([_hourly(panels[day]["activation_price"],
                                         panels[day]["hour_of_slot"]) for day in days])
    positive = np.stack([resize((panels[day]["activation_share"] > 0.0).astype(float), 96)
                         for day in days]).reshape(len(days), 24, 4)
    return {"capacity_price": capacity, "day_ahead_price": day_ahead,
            "activation_price": activation_price,
            "activation_probability": positive.mean(axis=(0, 2))}


def x14_day_scenarios(delivery_day: str, panels: dict, forecast: dict,
                      config: dict, seed: int) -> tuple:
    """Generate the authorised wind--battery reconstruction for one DK1 day."""
    fit = window_days(delivery_day, forecast)["fit"]
    _require(set(fit).issubset(panels),
             f"{delivery_day}: wind--battery reconstruction panel is incomplete")
    paths = _panel_paths(panels, fit)
    count = int(config["export"]["saved_scenario_count"])
    generator = np.random.default_rng(int(seed))
    day_indices = generator.integers(0, len(fit), size=count)
    activation_indices = generator.integers(0, len(fit), size=count)
    capacity = np.empty((count, 24))
    for scenario in range(count):
        for hour in range(24):
            capacity[scenario, hour] = generator.choice(paths["capacity_price"][:, hour])
    procured = resize(panels[delivery_day]["procured_mw"], 24)
    alpha = paths["activation_probability"]
    activation_volume = np.repeat((procured * alpha * 0.5)[None, :], count, axis=0)
    values = np.stack([paths["day_ahead_price"][day_indices], capacity,
                       activation_volume, paths["activation_price"][activation_indices]], axis=1)
    diagnostics = {"fit_days": len(fit), "alpha_min": float(alpha.min()),
                   "alpha_max": float(alpha.max()), "day_indices": day_indices,
                   "activation_indices": activation_indices, "paths": paths}
    return values, diagnostics


def check_x14_day(values: np.ndarray, diagnostics: dict) -> dict:
    """Enforce the three provenance checks frozen in task check 11."""
    paths = diagnostics["paths"]
    capacity = values[:, 1]
    membership = [np.isin(capacity[:, hour], paths["capacity_price"][:, hour]).all()
                  for hour in range(24)]
    day_paths = values[:, 0]
    activation_paths = values[:, 3]
    day_match = [np.array_equal(path, paths["day_ahead_price"][index])
                 for path, index in zip(day_paths, diagnostics["day_indices"])]
    activation_match = [np.array_equal(path, paths["activation_price"][index])
                        for path, index in zip(activation_paths,
                                               diagnostics["activation_indices"])]
    _require(all(membership),
             "Wind--battery reconstruction capacity is outside its hourly set")
    _require(all(day_match),
             "Wind--battery reconstruction day-ahead scenario is not a whole historical day")
    _require(all(activation_match),
             "Wind--battery reconstruction activation price is not a whole historical day")
    _require(0.0 <= diagnostics["alpha_min"] <= diagnostics["alpha_max"] <= 1.0,
             "Wind--battery reconstruction activation probability lies outside [0, 1]")
    return {"capacity_membership": True, "whole_day_day_ahead_paths": True,
            "whole_day_activation_price_paths": True,
            "alpha_min": diagnostics["alpha_min"], "alpha_max": diagnostics["alpha_max"]}


def generate_x14_archive(panels: dict, days: list, forecast: dict,
                         config: dict) -> tuple:
    """Generate all wind--battery reconstruction scenarios and diagnostics."""
    arrays, checks = [], []
    base_seed = int(config["export"]["sampling_seed"])
    for index, day in enumerate(days):
        values, diagnostics = x14_day_scenarios(day, panels, forecast, config, base_seed + index)
        arrays.append(values.astype(np.float32))
        checks.append(check_x14_day(values, diagnostics))
    return np.stack(arrays), {"days": len(checks),
                              "alpha_min": min(item["alpha_min"] for item in checks),
                              "alpha_max": max(item["alpha_max"] for item in checks),
                              "all_provenance_checks": all(
                                  all(value is True for key, value in item.items()
                                      if key not in {"alpha_min", "alpha_max"}) for item in checks)}


def write_x14_archive(root: Path, panels: dict, days: list,
                      forecast: dict, config: dict) -> dict:
    """Write the wind--battery reconstruction archive with frozen metadata."""
    values, checks = generate_x14_archive(panels, days, forecast, config)
    directory = root / config["output"]["directory"] / config["output"]["scenarios_0730_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "x14_rebuild.npz"
    config_path = root / "configs/s3_comparison.yaml"
    np.savez_compressed(path, delivery_days=np.asarray(days), targets=np.asarray(TARGETS),
                        values=values, seed=int(config["export"]["sampling_seed"]),
                        information_set="gate_0730", activation_representation="expected_value",
                        evaluated_days=np.asarray(days),
                        source_config_sha256=sha256(config_path.read_bytes()).hexdigest())
    return {"path": str(path.relative_to(root)), "shape": list(values.shape), **checks}


def _x06_hourly_values(point_day_ahead: np.ndarray, point_capacity: np.ndarray,
                       climatology: dict, delivery_day: str,
                       epsilon: float) -> np.ndarray:
    index = climatology["delivery_days"].index(delivery_day)
    source = climatology["values"][index].astype(float)
    volume_position = climatology["targets"].index("activation_volume")
    price_position = climatology["targets"].index("activation_price")
    quantile = np.quantile(source[:, volume_position], 1.0 - epsilon, axis=0)
    selected = int(np.argmin(np.sum((source[:, volume_position] - quantile) ** 2, axis=1)))
    synthetic = source[selected:selected + 1].copy()
    synthetic[:, 0] = point_day_ahead
    synthetic[:, 1] = point_capacity
    synthetic[:, volume_position] = quantile
    synthetic[:, price_position] = source[selected, price_position]
    return synthetic


def x06_deterministic_scenario(point_day_ahead: np.ndarray, point_capacity: np.ndarray,
                               climatology: dict, delivery_day: str,
                               procured_mw: np.ndarray, premium_history,
                               forecast: dict, seed: int, epsilon: float) -> dict:
    """Build the declared DK1 deterministic chance-constraint approximation."""
    synthetic = _x06_hourly_values(
        point_day_ahead, point_capacity, climatology, delivery_day, epsilon)
    archive = {**climatology, "values": np.expand_dims(synthetic, 0),
               "delivery_days": [delivery_day], "evaluated_days": [delivery_day]}
    from s3_comparison.scenario_io import prepare_day_scenarios
    return prepare_day_scenarios(archive, delivery_day, procured_mw, premium_history,
                                 forecast, "gate_0730", seed)
