"""Scoring and significance tests for the frozen S3-A protocol."""

from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def _float_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    assert array.size > 0, f"{name} is empty"
    assert not np.isinf(array).any(), f"{name} contains infinity"
    return array


def _boolean_mask(mask: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(mask)
    assert array.shape == shape, "evaluation mask shape mismatch"
    assert array.dtype == np.bool_, "evaluation mask must be boolean"
    return array


def _valid_vectors(
    actual: np.ndarray, arrays: Sequence[np.ndarray], evaluation_mask: np.ndarray
) -> np.ndarray:
    valid = _boolean_mask(evaluation_mask, actual.shape) & np.isfinite(actual)
    for array in arrays:
        assert array.shape[: actual.ndim] == actual.shape, "forecast shape mismatch"
        axes = tuple(range(actual.ndim, array.ndim))
        finite = np.isfinite(array)
        valid &= finite.all(axis=axes) if axes else finite
    assert valid.any(), "no valid evaluation observations"
    return valid


def _ensemble_inputs(
    actual: np.ndarray, scenarios: np.ndarray, evaluation_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = _float_array(actual, "actual")
    ensemble = _float_array(scenarios, "scenarios")
    assert observed.ndim == 1, "actual must be one-dimensional"
    assert ensemble.ndim == 2, "scenarios must have shape (observation, scenario)"
    valid = _valid_vectors(observed, [ensemble], evaluation_mask)
    return observed[valid], ensemble[valid], valid


def validate_scoring_config(config: Mapping[str, object]) -> Mapping[str, object]:
    """Validate and return the YAML scoring mapping."""
    required = {
        "quantile_levels", "central_interval_levels", "tail_quantile_levels",
        "joint_targets", "joint_blocks", "energy_score_beta", "variogram_power",
        "variogram_weights", "dependence_lags_hours", "upper_tail_quantile",
        "pressure_thresholds", "dm_newey_west_lags", "dm_sidedness",
        "bootstrap_block_days", "bootstrap_repetitions", "bootstrap_confidence_level",
        "bootstrap_seed", "randomized_pit_seed", "hand_check_tolerance",
        "synthetic_check_tolerance",
    }
    assert "scoring" in config, "missing YAML scoring mapping"
    scoring = config["scoring"]
    assert isinstance(scoring, Mapping), "scoring must be a mapping"
    missing = required.difference(scoring)
    assert not missing, f"missing scoring keys: {sorted(missing)}"
    quantiles = _probability_levels(scoring["quantile_levels"], "quantile_levels")
    intervals = _probability_levels(scoring["central_interval_levels"], "central intervals")
    tails = _probability_levels(scoring["tail_quantile_levels"], "tail quantiles")
    assert np.all(np.diff(quantiles) > 0.0), "quantile levels must increase"
    assert np.all(np.diff(intervals) > 0.0), "interval levels must increase"
    assert np.all(np.diff(tails) > 0.0), "tail levels must increase"
    assert scoring["variogram_weights"] == "equal", "variogram weights must be equal"
    assert scoring["dm_sidedness"] == "two_sided", "DM test must be two-sided"
    assert isinstance(scoring["joint_blocks"], Mapping), "joint blocks must be a mapping"
    assert isinstance(scoring["pressure_thresholds"], Mapping), "pressure thresholds must map"
    assert {"capacity_price", "activation_volume"}.issubset(scoring["pressure_thresholds"]), (
        "missing frozen pressure threshold"
    )
    assert int(scoring["dm_newey_west_lags"]) >= 0, "DM lag count must be nonnegative"
    assert int(scoring["bootstrap_block_days"]) > 0, "block length must be positive"
    assert int(scoring["bootstrap_repetitions"]) > 0, "bootstrap repetitions must be positive"
    _assert_probability(float(scoring["bootstrap_confidence_level"]), "confidence level")
    _assert_probability(float(scoring["upper_tail_quantile"]), "upper-tail quantile")
    assert float(scoring["energy_score_beta"]) > 0.0, "energy beta must be positive"
    assert float(scoring["variogram_power"]) > 0.0, "variogram power must be positive"
    assert float(scoring["hand_check_tolerance"]) > 0.0, "hand tolerance must be positive"
    assert float(scoring["synthetic_check_tolerance"]) > 0.0, "synthetic tolerance must be positive"
    return scoring


def _assert_probability(value: float, name: str) -> None:
    assert np.isfinite(value) and 0.0 < value < 1.0, f"{name} must be in (0, 1)"


def _probability_levels(values: object, name: str) -> np.ndarray:
    levels = _float_array(np.asarray(values), name)
    assert levels.ndim == 1, f"{name} must be one-dimensional"
    assert np.isfinite(levels).all(), f"{name} contains missing values"
    assert np.all((levels > 0.0) & (levels < 1.0)), f"{name} must lie in (0, 1)"
    return levels


def point_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    naive_forecast: np.ndarray,
    evaluation_mask: np.ndarray,
) -> Dict[str, float]:
    """Return MAE, RMSE, and rMAE on pairwise-valid observations."""
    observed = _float_array(actual, "actual")
    predicted = _float_array(forecast, "forecast")
    naive = _float_array(naive_forecast, "naive forecast")
    assert observed.ndim == predicted.ndim == naive.ndim == 1, "inputs must be vectors"
    valid = _valid_vectors(observed, [predicted, naive], evaluation_mask)
    errors = predicted[valid] - observed[valid]
    naive_mae = float(np.mean(np.abs(naive[valid] - observed[valid])))
    assert naive_mae > 0.0, "rMAE denominator is zero"
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "rmae": float(np.mean(np.abs(errors)) / naive_mae),
        "n_valid": int(valid.sum()),
    }


def pinball_metrics(
    actual: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: Sequence[float],
    evaluation_mask: np.ndarray,
) -> Dict[str, object]:
    """Return average and per-level pinball losses with the valid count."""
    observed = _float_array(actual, "actual")
    forecasts = _float_array(quantile_forecasts, "quantile forecasts")
    levels = _probability_levels(quantile_levels, "quantile levels")
    assert observed.ndim == 1 and forecasts.ndim == 2, "invalid quantile input dimensions"
    assert forecasts.shape == (observed.size, levels.size), "quantile forecast shape mismatch"
    valid = _valid_vectors(observed, [forecasts], evaluation_mask)
    errors = observed[valid, None] - forecasts[valid]
    losses = np.maximum(levels[None, :] * errors, (levels[None, :] - 1.0) * errors)
    per_level = losses.mean(axis=0)
    return {
        "mean_pinball": float(per_level.mean()),
        "pinball_by_quantile": per_level.tolist(),
        "n_valid": int(valid.sum()),
    }


def sample_crps(
    actual: np.ndarray, scenarios: np.ndarray, evaluation_mask: np.ndarray
) -> Dict[str, float]:
    """Return ensemble CRPS using the S-squared V-statistic."""
    observed, ensemble, _ = _ensemble_inputs(actual, scenarios, evaluation_mask)
    values = np.empty(observed.size, dtype=float)
    for row, (value, draws) in enumerate(zip(observed, ensemble)):
        first = np.mean(np.abs(draws - value))
        second = 0.5 * np.mean(np.abs(draws[:, None] - draws[None, :]))
        values[row] = first - second
    return {"crps": float(values.mean()), "n_valid": int(values.size)}


def central_interval_metrics(
    actual: np.ndarray,
    scenarios: np.ndarray,
    interval_levels: Sequence[float],
    evaluation_mask: np.ndarray,
) -> List[Dict[str, float]]:
    """Return central-interval coverage and width for each configured level."""
    observed, ensemble, _ = _ensemble_inputs(actual, scenarios, evaluation_mask)
    levels = _probability_levels(interval_levels, "central interval levels")
    rows: List[Dict[str, float]] = []
    for level in levels:
        lower_probability = (1.0 - level) / 2.0
        lower = np.quantile(ensemble, lower_probability, axis=1)
        upper = np.quantile(ensemble, 1.0 - lower_probability, axis=1)
        covered = (observed >= lower) & (observed <= upper)
        rows.append({
            "interval_level": float(level),
            "coverage": float(covered.mean()),
            "average_width": float(np.mean(upper - lower)),
            "n_valid": int(observed.size),
        })
    return rows


def randomized_pit_metrics(
    actual: np.ndarray,
    scenarios: np.ndarray,
    seed: int,
    evaluation_mask: np.ndarray,
) -> Dict[str, object]:
    """Return randomized ensemble PIT values and their uniform KS distance."""
    observed, ensemble, _ = _ensemble_inputs(actual, scenarios, evaluation_mask)
    less = np.mean(ensemble < observed[:, None], axis=1)
    equal = np.mean(ensemble == observed[:, None], axis=1)
    random_values = np.random.default_rng(int(seed)).uniform(size=observed.size)
    pit = less + random_values * equal
    assert np.all((pit >= 0.0) & (pit <= 1.0)), "PIT values outside [0, 1]"
    distance = stats.kstest(pit, "uniform").statistic
    return {"pit": pit, "pit_ks_distance": float(distance), "n_valid": int(pit.size)}


def marginal_metrics(
    actual: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: Sequence[float],
    scenarios: np.ndarray,
    interval_levels: Sequence[float],
    pit_seed: int,
    evaluation_mask: np.ndarray,
) -> Dict[str, object]:
    """Return all marginal scores required by the S3-A protocol."""
    pinball = pinball_metrics(actual, quantile_forecasts, quantile_levels, evaluation_mask)
    crps = sample_crps(actual, scenarios, evaluation_mask)
    intervals = central_interval_metrics(actual, scenarios, interval_levels, evaluation_mask)
    pit = randomized_pit_metrics(actual, scenarios, pit_seed, evaluation_mask)
    assert pinball["n_valid"] == crps["n_valid"] == pit["n_valid"], "valid counts differ"
    return {
        "mean_pinball": pinball["mean_pinball"],
        "pinball_by_quantile": pinball["pinball_by_quantile"],
        "sample_crps": crps["crps"],
        "central_intervals": intervals,
        "pit_ks_distance": pit["pit_ks_distance"],
        "n_valid": crps["n_valid"],
    }


def activation_volume_metrics(
    actual: np.ndarray,
    scenarios: np.ndarray,
    zero_value: float,
    evaluation_mask: np.ndarray,
) -> Dict[str, float]:
    """Return zero-event Brier score and positive-observation CRPS."""
    assert np.isfinite(float(zero_value)), "zero-event value must be finite"
    observed, ensemble, valid = _ensemble_inputs(actual, scenarios, evaluation_mask)
    observed_zero = observed == float(zero_value)
    predicted_zero_probability = np.mean(ensemble == float(zero_value), axis=1)
    brier = np.mean((predicted_zero_probability - observed_zero.astype(float)) ** 2)
    positive_mask = np.asarray(evaluation_mask, dtype=bool) & (np.asarray(actual) > zero_value)
    positive_crps = sample_crps(actual, scenarios, positive_mask)
    return {
        "zero_event_brier": float(brier),
        "positive_observation_crps": float(positive_crps["crps"]),
        "n_valid": int(valid.sum()),
        "n_positive": int(positive_crps["n_valid"]),
    }


def tail_exceedance_metrics(
    actual: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: Sequence[float],
    tail_levels: Sequence[float],
    pressure_threshold: Optional[float],
    evaluation_mask: np.ndarray,
) -> List[Dict[str, float]]:
    """Return overall and, where a frozen threshold exists, pressure-hour exceedance."""
    observed = _float_array(actual, "actual")
    forecasts = _float_array(quantile_forecasts, "quantile forecasts")
    levels = _probability_levels(quantile_levels, "quantile levels")
    tails = _probability_levels(tail_levels, "tail levels")
    assert pressure_threshold is None or np.isfinite(float(pressure_threshold)), \
        "pressure threshold must be finite when present"
    assert forecasts.shape == (observed.size, levels.size), "quantile forecast shape mismatch"
    assert all(np.any(levels == tail) for tail in tails), "tail level absent from quantile grid"
    indices = [int(np.flatnonzero(levels == tail)[0]) for tail in tails]
    valid = _valid_vectors(observed, [forecasts], evaluation_mask)
    pressure = valid & (observed >= float(pressure_threshold)) if pressure_threshold is not None \
        else np.zeros_like(valid)
    rows: List[Dict[str, float]] = []
    for tail, index in zip(tails, indices):
        exceeded = observed > forecasts[:, index]
        rows.append({
            "quantile_level": float(tail),
            "exceedance_frequency": float(exceeded[valid].mean()),
            "pressure_exceedance_frequency": float(exceeded[pressure].mean()) if pressure.any() else np.nan,
            "n_valid": int(valid.sum()),
            "n_pressure": int(pressure.sum()),
        })
    return rows


def energy_score(observation: np.ndarray, scenarios: np.ndarray, beta: float) -> float:
    """Return the loss-oriented Energy Score with an S-squared V-statistic."""
    observed = _float_array(observation, "observation")
    ensemble = _float_array(scenarios, "scenarios")
    assert observed.ndim == 1 and ensemble.ndim == 2, "invalid Energy Score dimensions"
    assert ensemble.shape[1] == observed.size, "Energy Score dimension mismatch"
    assert np.isfinite(observed).all() and np.isfinite(ensemble).all(), "missing joint value"
    assert 0.0 < float(beta) <= 2.0, "Energy Score beta must be in (0, 2]"
    distance_to_observation = np.linalg.norm(ensemble - observed[None, :], axis=1) ** beta
    pairwise = np.linalg.norm(ensemble[:, None, :] - ensemble[None, :, :], axis=2) ** beta
    return float(distance_to_observation.mean() - 0.5 * pairwise.mean())


def variogram_score(
    observation: np.ndarray, scenarios: np.ndarray, power: float, weights: str
) -> float:
    """Return equal-weight VS summed over every ordered dimension pair."""
    observed = _float_array(observation, "observation")
    ensemble = _float_array(scenarios, "scenarios")
    assert observed.ndim == 1 and ensemble.ndim == 2, "invalid Variogram Score dimensions"
    assert ensemble.shape[1] == observed.size, "Variogram Score dimension mismatch"
    assert np.isfinite(observed).all() and np.isfinite(ensemble).all(), "missing joint value"
    assert float(power) > 0.0, "variogram power must be positive"
    assert weights == "equal", "only equal variogram weights are permitted"
    observed_variogram = np.abs(observed[:, None] - observed[None, :]) ** power
    scenario_variogram = np.mean(
        np.abs(ensemble[:, :, None] - ensemble[:, None, :]) ** power, axis=0
    )
    return float(np.sum((observed_variogram - scenario_variogram) ** 2))


def _joint_block(
    observation: np.ndarray,
    scenarios: np.ndarray,
    target_scales: np.ndarray,
    target_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    selected_observation = observation[np.asarray(target_indices), :]
    selected_scenarios = scenarios[:, np.asarray(target_indices), :]
    scales = target_scales[np.asarray(target_indices), None]
    assert np.isfinite(scales).all() and np.all(scales > 0.0), "invalid F(D) target scale"
    scaled_observation = (selected_observation / scales).reshape(-1)
    scaled_scenarios = (selected_scenarios / scales[None, :, :]).reshape(scenarios.shape[0], -1)
    valid = np.isfinite(scaled_observation) & np.isfinite(scaled_scenarios).all(axis=0)
    assert valid.any(), "joint score has no valid dimensions"
    return scaled_observation[valid], scaled_scenarios[:, valid], int(valid.sum())


def joint_metrics(
    observation: np.ndarray,
    scenarios: np.ndarray,
    target_scales: np.ndarray,
    target_names: Sequence[str],
    joint_blocks: Mapping[str, Sequence[str]],
    energy_beta: float,
    variogram_power: float,
    variogram_weights: str,
) -> List[Dict[str, object]]:
    """Return standardized daily joint scores for all configured target blocks."""
    observed = _float_array(observation, "joint observation")
    ensemble = _float_array(scenarios, "joint scenarios")
    scales = _float_array(target_scales, "target scales")
    assert observed.ndim == 2 and ensemble.ndim == 3, "invalid joint input dimensions"
    assert ensemble.shape[1:] == observed.shape, "joint scenario shape mismatch"
    assert scales.shape == (observed.shape[0],), "target scale shape mismatch"
    assert len(target_names) == observed.shape[0], "target name count mismatch"
    assert len(set(target_names)) == len(target_names), "duplicate target name"
    lookup = {name: index for index, name in enumerate(target_names)}
    rows: List[Dict[str, object]] = []
    for block_name, block_targets in joint_blocks.items():
        assert block_targets and all(name in lookup for name in block_targets), "invalid joint block"
        block_observed, block_scenarios, n_valid = _joint_block(
            observed, ensemble, scales, [lookup[name] for name in block_targets]
        )
        rows.append({
            "block": block_name,
            "energy_score": energy_score(block_observed, block_scenarios, energy_beta),
            "variogram_score": variogram_score(
                block_observed, block_scenarios, variogram_power, variogram_weights
            ),
            "n_valid_dimensions": n_valid,
        })
    return rows


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> Tuple[float, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    count = int(valid.sum())
    if count < 2:
        return np.nan, count
    left_ranks = stats.rankdata(left[valid])
    right_ranks = stats.rankdata(right[valid])
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return np.nan, count
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1]), count


def _lagged_pair(left: np.ndarray, right: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    assert left.shape == right.shape and left.ndim >= 2, "lagged arrays must match"
    assert 0 <= lag < left.shape[-1], "invalid dependence lag"
    if lag == 0:
        return left.reshape(-1), right.reshape(-1)
    return left[..., :-lag].reshape(-1), right[..., lag:].reshape(-1)


def _correlation_row(
    actual_left: np.ndarray,
    actual_right: np.ndarray,
    scenario_left: np.ndarray,
    scenario_right: np.ndarray,
    target_left: str,
    target_right: str,
    lag: int,
) -> Dict[str, object]:
    observed_pair = _lagged_pair(actual_left, actual_right, lag)
    scenario_pair = _lagged_pair(scenario_left, scenario_right, lag)
    observed_rho, observed_count = _rank_correlation(*observed_pair)
    scenario_rho, scenario_count = _rank_correlation(*scenario_pair)
    difference = scenario_rho - observed_rho
    return {
        "metric": "spearman_rho",
        "target_x": target_left,
        "target_y": target_right,
        "lag_hours_y_after_x": int(lag),
        "observed_value": observed_rho,
        "scenario_value": scenario_rho,
        "difference": difference,
        "n_observed_pairs": observed_count,
        "n_scenario_pairs": scenario_count,
    }


def _upper_tail_probability(left: np.ndarray, right: np.ndarray, level: float) -> Tuple[float, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    assert valid.any(), "upper-tail dependence has no valid pairs"
    left_valid, right_valid = left[valid], right[valid]
    left_threshold = np.quantile(left_valid, level)
    right_threshold = np.quantile(right_valid, level)
    conditioning = left_valid > left_threshold
    assert conditioning.any(), "upper-tail conditioning event is empty"
    probability = np.mean(right_valid[conditioning] > right_threshold)
    return float(probability), int(conditioning.sum())


def _tail_dependence_row(
    actual: np.ndarray,
    scenarios: np.ndarray,
    lookup: Mapping[str, int],
    upper_tail_level: float,
) -> Dict[str, object]:
    day_ahead = lookup["day_ahead_price"]
    capacity = lookup["capacity_price"]
    observed_probability, observed_count = _upper_tail_probability(
        actual[:, day_ahead, :].reshape(-1), actual[:, capacity, :].reshape(-1), upper_tail_level
    )
    scenario_probability, scenario_count = _upper_tail_probability(
        scenarios[:, :, day_ahead, :].reshape(-1),
        scenarios[:, :, capacity, :].reshape(-1),
        upper_tail_level,
    )
    return {
        "metric": "upper_tail_conditional_probability",
        "target_x": "day_ahead_price",
        "target_y": "capacity_price",
        "lag_hours_y_after_x": 0,
        "observed_value": observed_probability,
        "scenario_value": scenario_probability,
        "difference": scenario_probability - observed_probability,
        "n_observed_pairs": observed_count,
        "n_scenario_pairs": scenario_count,
    }


def dependence_metrics(
    actual: np.ndarray,
    scenarios: np.ndarray,
    target_names: Sequence[str],
    joint_targets: Sequence[str],
    dependence_lags: Sequence[int],
    upper_tail_level: float,
) -> List[Dict[str, object]]:
    """Return cross-target, upper-tail, and lag-one dependence errors."""
    observed = _float_array(actual, "dependence observations")
    ensemble = _float_array(scenarios, "dependence scenarios")
    assert observed.ndim == 3 and ensemble.ndim == 4, "invalid dependence dimensions"
    assert ensemble.shape[0] == observed.shape[0], "delivery-day count mismatch"
    assert ensemble.shape[2:] == observed.shape[1:], "dependence target/hour mismatch"
    assert len(target_names) == observed.shape[1], "target name count mismatch"
    lookup = {name: index for index, name in enumerate(target_names)}
    assert len(lookup) == len(target_names), "duplicate target name"
    assert all(name in lookup for name in joint_targets), "joint target absent"
    lags = [int(value) for value in dependence_lags]
    assert 1 in lags, "lag-one dependence is required"
    rows: List[Dict[str, object]] = []
    for left_name, right_name in combinations(joint_targets, 2):
        left, right = lookup[left_name], lookup[right_name]
        for lag in lags:
            rows.append(_correlation_row(
                observed[:, left, :], observed[:, right, :],
                ensemble[:, :, left, :], ensemble[:, :, right, :],
                left_name, right_name, lag,
            ))
    for name in joint_targets:
        index = lookup[name]
        rows.append(_correlation_row(
            observed[:, index, :], observed[:, index, :],
            ensemble[:, :, index, :], ensemble[:, :, index, :], name, name, 1,
        ))
    _assert_probability(float(upper_tail_level), "upper-tail level")
    rows.append(_tail_dependence_row(observed, ensemble, lookup, upper_tail_level))
    return rows


def conditional_metrics(
    actual: np.ndarray,
    scenarios: np.ndarray,
    median_forecast: np.ndarray,
    evaluation_mask: np.ndarray,
) -> Dict[str, float]:
    """Return CRPS and median MAE for a 12:00 day-ahead product."""
    crps = sample_crps(actual, scenarios, evaluation_mask)
    observed = _float_array(actual, "conditional actual")
    median = _float_array(median_forecast, "conditional median")
    assert median.shape == observed.shape, "conditional median shape mismatch"
    valid = _valid_vectors(observed, [median], evaluation_mask)
    return {
        "sample_crps": float(crps["crps"]),
        "median_mae": float(np.mean(np.abs(median[valid] - observed[valid]))),
        "n_valid_crps": int(crps["n_valid"]),
        "n_valid_mae": int(valid.sum()),
    }


def monthly_calibration_metrics(
    delivery_days: Sequence[str],
    actual: np.ndarray,
    scenarios: np.ndarray,
    interval_level: float,
    evaluation_mask: np.ndarray,
) -> List[Dict[str, object]]:
    """Return monthly central-interval coverage deviations and widths."""
    days = np.asarray(delivery_days, dtype="datetime64[D]")
    observed = _float_array(actual, "monthly actual")
    ensemble = _float_array(scenarios, "monthly scenarios")
    assert observed.ndim == 2 and ensemble.ndim == 3, "invalid monthly input dimensions"
    assert ensemble.shape[0] == observed.shape[0] == days.size, "monthly day count mismatch"
    assert ensemble.shape[2] == observed.shape[1], "monthly hour count mismatch"
    mask = _boolean_mask(evaluation_mask, observed.shape)
    _assert_probability(float(interval_level), "monthly interval level")
    assert not np.isnat(days).any(), "delivery day contains NaT"
    months = days.astype("datetime64[M]")
    rows: List[Dict[str, object]] = []
    for month in np.unique(months):
        selected = months == month
        month_actual = observed[selected].reshape(-1)
        month_scenarios = ensemble[selected].transpose(0, 2, 1).reshape(-1, ensemble.shape[1])
        month_mask = mask[selected].reshape(-1)
        result = central_interval_metrics(
            month_actual, month_scenarios, [interval_level], month_mask
        )[0]
        rows.append({
            "month": str(month),
            "interval_level": float(interval_level),
            "coverage": result["coverage"],
            "coverage_deviation": result["coverage"] - float(interval_level),
            "average_width": result["average_width"],
            "n_valid": result["n_valid"],
        })
    return rows


def model_scale_metrics(n_parameters: int, n_effective_samples: int) -> Dict[str, float]:
    """Return trainable parameter count and effective-sample ratio."""
    assert int(n_parameters) > 0, "parameter count must be positive"
    assert int(n_effective_samples) > 0, "effective sample count must be positive"
    return {
        "n_parameters": int(n_parameters),
        "n_effective_samples": int(n_effective_samples),
        "effective_samples_per_parameter": float(n_effective_samples / n_parameters),
    }


def _paired_losses(loss_a: np.ndarray, loss_b: np.ndarray) -> np.ndarray:
    first = _float_array(loss_a, "loss A")
    second = _float_array(loss_b, "loss B")
    assert first.ndim == second.ndim == 1 and first.shape == second.shape, "loss shape mismatch"
    valid = np.isfinite(first) & np.isfinite(second)
    assert valid.all(), "daily loss is missing; block chronology cannot be compressed"
    return first - second


def dm_test(
    loss_a: np.ndarray, loss_b: np.ndarray, newey_west_lags: int, sidedness: str
) -> Dict[str, float]:
    """Return a two-sided daily-loss DM test with Bartlett Newey-West variance."""
    differences = _paired_losses(loss_a, loss_b)
    lag_count = int(newey_west_lags)
    assert sidedness == "two_sided", "only the frozen two-sided DM test is permitted"
    assert 0 <= lag_count < differences.size, "invalid Newey-West lag count"
    centered = differences - differences.mean()
    sample_count = differences.size
    long_run_variance = float(np.dot(centered, centered) / sample_count)
    for lag in range(1, lag_count + 1):
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / sample_count)
        long_run_variance += 2.0 * (1.0 - lag / (lag_count + 1.0)) * autocovariance
    assert np.isfinite(long_run_variance) and long_run_variance > 0.0, "invalid DM long-run variance"
    statistic = float(differences.mean() / np.sqrt(long_run_variance / sample_count))
    probability = float(2.0 * stats.norm.sf(abs(statistic)))
    return {
        "mean_loss_difference_a_minus_b": float(differences.mean()),
        "dm_statistic": statistic,
        "p_value": probability,
        "newey_west_lags": lag_count,
        "n_valid_days": int(sample_count),
    }


def block_bootstrap_mean_difference(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    block_days: int,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> Dict[str, float]:
    """Return a moving-block bootstrap interval for the mean daily loss difference."""
    differences = _paired_losses(loss_a, loss_b)
    block_length = int(block_days)
    repeat_count = int(repetitions)
    confidence = float(confidence_level)
    assert 0 < block_length <= differences.size, "invalid bootstrap block length"
    assert repeat_count > 0, "bootstrap repetitions must be positive"
    _assert_probability(confidence, "bootstrap confidence level")
    blocks_needed = int(np.ceil(differences.size / block_length))
    maximum_start = differences.size - block_length
    offsets = np.arange(block_length)
    generator = np.random.default_rng(int(seed))
    means = np.empty(repeat_count, dtype=float)
    for repetition in range(repeat_count):
        starts = generator.integers(0, maximum_start + 1, size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]).reshape(-1)[: differences.size]
        means[repetition] = differences[indices].mean()
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "mean_loss_difference_a_minus_b": float(differences.mean()),
        "confidence_lower": float(lower),
        "confidence_upper": float(upper),
        "confidence_level": confidence,
        "block_days": block_length,
        "bootstrap_repetitions": repeat_count,
        "bootstrap_seed": int(seed),
        "n_valid_days": int(differences.size),
    }


def significance_metrics(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    newey_west_lags: int,
    sidedness: str,
    block_days: int,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> Dict[str, float]:
    """Return the frozen DM test and seven-day block-bootstrap interval."""
    dm = dm_test(loss_a, loss_b, newey_west_lags, sidedness)
    bootstrap = block_bootstrap_mean_difference(
        loss_a, loss_b, block_days, repetitions, confidence_level, seed
    )
    assert dm["n_valid_days"] == bootstrap["n_valid_days"], "significance counts differ"
    return {**dm, **{key: value for key, value in bootstrap.items() if key not in dm}}


def _uniform_synthetic_checks(scoring: Mapping[str, object]) -> Dict[str, float]:
    sample_count = 401
    actual = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    levels = np.asarray(scoring["quantile_levels"], dtype=float)
    forecasts = np.tile(levels, (sample_count, 1))
    scenarios = np.tile(actual, (sample_count, 1))
    mask = np.ones(sample_count, dtype=bool)
    pinball = pinball_metrics(actual, forecasts, levels, mask)
    expected_pinball = float(np.mean(0.5 * levels * (1.0 - levels)))
    intervals = central_interval_metrics(
        actual, scenarios, scoring["central_interval_levels"], mask
    )
    pit = randomized_pit_metrics(actual, scenarios, int(scoring["randomized_pit_seed"]), mask)
    interval_error = max(abs(row["coverage"] - row["interval_level"]) for row in intervals)
    return {
        "uniform_pinball_error": abs(float(pinball["mean_pinball"]) - expected_pinball),
        "uniform_interval_max_error": float(interval_error),
        "uniform_pit_ks_distance": float(pit["pit_ks_distance"]),
    }


def _significance_determinism_check(scoring: Mapping[str, object]) -> float:
    sample_count = 4 * (int(scoring["dm_newey_west_lags"]) + 1)
    loss_b = np.linspace(0.5, 1.5, sample_count)
    loss_a = loss_b + np.linspace(-0.2, 0.4, sample_count) + 0.03 * np.sin(np.arange(sample_count))
    arguments = (
        loss_a, loss_b, int(scoring["bootstrap_block_days"]),
        int(scoring["bootstrap_repetitions"]), float(scoring["bootstrap_confidence_level"]),
        int(scoring["bootstrap_seed"]),
    )
    first = block_bootstrap_mean_difference(*arguments)
    second = block_bootstrap_mean_difference(*arguments)
    assert first == second, "bootstrap is not deterministic for a fixed seed"
    dm = dm_test(loss_a, loss_b, int(scoring["dm_newey_west_lags"]), scoring["dm_sidedness"])
    assert np.isfinite(dm["dm_statistic"]), "DM hand check is non-finite"
    return float(first["confidence_upper"] - first["confidence_lower"])


def hand_checks(config: Mapping[str, object]) -> Dict[str, float]:
    """Run deterministic hand and synthetic checks required before the full experiment."""
    scoring = validate_scoring_config(config)
    tolerance = float(scoring["hand_check_tolerance"])
    synthetic_tolerance = float(scoring["synthetic_check_tolerance"])
    mask = np.ones(1, dtype=bool)
    degenerate_crps = sample_crps(np.array([2.0]), np.array([[5.0, 5.0]]), mask)["crps"]
    one_dimensional_es = energy_score(np.array([2.0]), np.array([[5.0], [5.0]]), scoring["energy_score_beta"])
    zero_es = energy_score(np.array([2.0]), np.array([[2.0], [2.0]]), scoring["energy_score_beta"])
    variogram = variogram_score(
        np.array([0.0, 2.0]), np.array([[0.0, 1.0], [2.0, 3.0]]),
        scoring["variogram_power"], scoring["variogram_weights"],
    )
    expected_variogram = 2.0 * (np.sqrt(2.0) - 1.0) ** 2
    assert abs(degenerate_crps - 3.0) <= tolerance, "degenerate CRPS hand check failed"
    assert abs(one_dimensional_es - degenerate_crps) <= tolerance, "one-dimensional ES check failed"
    assert abs(zero_es) <= tolerance, "perfect-scenario Energy Score check failed"
    assert abs(variogram - expected_variogram) <= tolerance, "ordered-pair VS check failed"
    synthetic = _uniform_synthetic_checks(scoring)
    assert max(synthetic.values()) <= synthetic_tolerance, "synthetic distribution check failed"
    interval_width = _significance_determinism_check(scoring)
    return {
        "degenerate_crps": float(degenerate_crps),
        "one_dimensional_energy_score": float(one_dimensional_es),
        "perfect_energy_score": float(zero_es),
        "ordered_pair_variogram_score": float(variogram),
        **synthetic,
        "bootstrap_interval_width": interval_width,
    }
