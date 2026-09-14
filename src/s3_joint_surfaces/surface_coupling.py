"""State-conditioned rank coupling from four day-target-hour-quantile surfaces."""

import numpy as np

from src.s3_forecast_side import marginals, scoring
from src.s3_forecast_side.generators import TARGETS, _stream


def probability_cell_weights(levels: np.ndarray, config: dict) -> np.ndarray:
    """Give each requested quantile its midpoint-bounded probability-cell width."""
    assert config["joint_surfaces"]["quantile_weighting"] == "midpoint_probability_cells"
    values = np.asarray(levels, dtype=float)
    assert values.ndim == 1 and values.size > 1 and np.all(np.diff(values) > 0.0)
    edges = np.concatenate([[0.0], (values[:-1] + values[1:]) / 2.0, [1.0]])
    weights = np.diff(edges)
    assert np.all(weights > 0.0) and np.isclose(weights.sum(), 1.0)
    return weights


def surface_distances(fit_surfaces: np.ndarray, current_surface: np.ndarray,
                      target_scales: np.ndarray, target_indices: np.ndarray,
                      levels: np.ndarray, config: dict) -> np.ndarray:
    """Return target-balanced probability-weighted RMS terrain distances."""
    assert config["joint_surfaces"]["distance"] == \
        "fit_window_target_scale_probability_weighted_rms"
    history = np.asarray(fit_surfaces, dtype=float)
    current = np.asarray(current_surface, dtype=float)
    scales = np.asarray(target_scales, dtype=float)
    indices = np.asarray(target_indices, dtype=int)
    assert history.ndim == 4 and current.shape == history.shape[1:]
    assert history.shape[1:] == (len(TARGETS), 24, len(levels))
    assert scales.shape == (len(TARGETS),) and np.all(np.isfinite(scales)) and np.all(scales > 0.0)
    assert indices.size > 0 and np.isfinite(history[:, indices]).all() and np.isfinite(current[indices]).all()
    difference = (history[:, indices] - current[None, indices]) / scales[None, indices, None, None]
    weighted = np.sum(difference ** 2 * probability_cell_weights(levels, config)[None, None, None, :], axis=3)
    distances = np.sqrt(weighted.mean(axis=(1, 2)))
    assert np.isfinite(distances).all() and np.all(distances >= 0.0)
    return distances


def gaussian_neighbor_weights(distances: np.ndarray, neighbor_count: int,
                              config: dict) -> tuple:
    """Return nearest-day indices and normalized Gaussian kernel weights."""
    settings = config["joint_surfaces"]["kernel"]
    assert settings["kind"] == "gaussian" and settings["support"] == "nearest_k_fit_days"
    assert settings["bandwidth"] == "kth_neighbor_distance"
    values = np.asarray(distances, dtype=float)
    count = int(neighbor_count)
    assert values.ndim == 1 and 0 < count <= values.size and np.isfinite(values).all()
    selected = np.argsort(values, kind="mergesort")[:count]
    bandwidth = max(float(values[selected[-1]]), float(settings["minimum_bandwidth"]))
    unnormalized = np.exp(-0.5 * (values[selected] / bandwidth) ** 2)
    weights = unnormalized / unnormalized.sum()
    assert np.isfinite(weights).all() and np.all(weights > 0.0) and np.isclose(weights.sum(), 1.0)
    return selected, weights, bandwidth


def rank_reorder_uniforms(base_uniforms: np.ndarray, templates: np.ndarray) -> tuple:
    """Impose template ranks while preserving every marginal uniform sample exactly."""
    base = np.asarray(base_uniforms, dtype=float)
    ranks = np.asarray(templates, dtype=float)
    assert base.shape == ranks.shape and base.ndim == 2
    assert np.isfinite(base).all() and np.isfinite(ranks).all()
    output = np.empty_like(base)
    for column in range(base.shape[1]):
        order = np.argsort(ranks[:, column], kind="mergesort")
        output[order, column] = np.sort(base[:, column])
    error = float(np.max(np.abs(np.sort(output, axis=0) - np.sort(base, axis=0))))
    assert error == 0.0, "rank coupling changed a marginal sample"
    return output, error


def surface_scenarios(test_quantiles: np.ndarray, fit_surfaces: np.ndarray,
                      fit_uniforms: np.ndarray, target_scales: np.ndarray,
                      target_indices: np.ndarray, levels: np.ndarray, neighbor_count: int,
                      scenario_count: int, seed: int, config: dict) -> tuple:
    """Generate one surface-conditioned, marginal-preserving joint ensemble."""
    settings = config["joint_surfaces"]
    assert settings["coupling"]["method"] == "marginal_preserving_rank_reordering"
    assert settings["coupling"]["missing_rank_policy"] == "seeded_marginal_uniform"
    distances = surface_distances(fit_surfaces, test_quantiles, target_scales,
                                  target_indices, levels, config)
    selected, weights, bandwidth = gaussian_neighbor_weights(distances, neighbor_count, config)
    offsets = settings["stream_offsets"]
    generator = np.random.default_rng(_stream(seed, offsets["template_indices"]))
    sampled = generator.choice(selected, size=int(scenario_count), replace=True, p=weights)
    templates = np.asarray(fit_uniforms[sampled], dtype=float).copy()
    missing = ~np.isfinite(templates)
    missing_generator = np.random.default_rng(_stream(seed, offsets["missing_template_uniforms"]))
    templates[missing] = missing_generator.uniform(size=int(missing.sum()))
    base = np.random.default_rng(_stream(seed, offsets["marginal_uniforms"])).uniform(
        size=(int(scenario_count), len(TARGETS) * 24))
    coupled, error = rank_reorder_uniforms(base, templates)
    rows = test_quantiles.reshape(len(TARGETS) * 24, len(levels))
    values = marginals.evaluate_quantile_grid(rows, levels, coupled.T).T
    diagnostics = {"bandwidth": bandwidth, "effective_neighbors": float(1.0 / np.sum(weights ** 2)),
                   "unique_template_days": int(np.unique(sampled).size),
                   "marginal_preservation_error": error}
    return values.reshape(int(scenario_count), len(TARGETS), 24), diagnostics


def select_neighbor_count(fit_surfaces: np.ndarray, calibration_surfaces: np.ndarray,
                          fit_uniforms: np.ndarray, calibration_actual: np.ndarray,
                          target_scales: np.ndarray, target_indices: np.ndarray,
                          levels: np.ndarray, seed: int, config: dict) -> tuple:
    """Select the sole kernel hyperparameter on C(D) Energy Score only."""
    settings = config["joint_surfaces"]
    selection = settings["selection"]
    assert selection["days"] == "probability_calibration"
    assert selection["metric"] == "energy_score" and selection["block"] == "four_targets_observed"
    candidates = [int(value) for value in settings["neighbor_candidates"]]
    assert candidates == sorted(set(candidates)) and max(candidates) <= fit_surfaces.shape[0]
    rows = []
    for count in candidates:
        losses = []
        for index, surface in enumerate(calibration_surfaces):
            scenarios, _ = surface_scenarios(
                surface, fit_surfaces, fit_uniforms, target_scales, target_indices, levels, count,
                int(selection["scenario_count"]), int(seed) + index, config)
            result = scoring.joint_metrics(
                calibration_actual[index], scenarios, target_scales, TARGETS,
                {selection["block"]: config["scoring"]["joint_blocks"][selection["block"]]},
                config["scoring"]["energy_score_beta"], config["scoring"]["variogram_power"],
                config["scoring"]["variogram_weights"])[0]
            losses.append(result["energy_score"])
        rows.append({"neighbor_count": count, "calibration_energy_score": float(np.mean(losses))})
    chosen = min(rows, key=lambda item: (item["calibration_energy_score"], item["neighbor_count"]))
    return int(chosen["neighbor_count"]), rows
