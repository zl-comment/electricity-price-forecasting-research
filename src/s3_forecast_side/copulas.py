"""Empirical and low-rank Gaussian dependence models for S3-A."""

import numpy as np
from scipy import stats


def normal_scores(uniforms: np.ndarray, config: dict) -> np.ndarray:
    """Map pseudo-observations to clipped Gaussian scores."""
    bounds = np.asarray(config["copula"]["pit_clip"], dtype=float)
    assert bounds.shape == (2,) and 0.0 < bounds[0] < bounds[1] < 1.0, "invalid PIT clip"
    clipped = np.clip(np.asarray(uniforms, dtype=float), bounds[0], bounds[1])
    return stats.norm.ppf(clipped)


def pairwise_correlation(scores: np.ndarray, config: dict) -> tuple:
    """Estimate a pairwise-complete correlation matrix and sample-count matrix."""
    values = np.asarray(scores, dtype=float)
    dimension = values.shape[1]
    minimum = int(config["copula"]["minimum_pair_samples"])
    correlation = np.eye(dimension)
    counts = np.zeros((dimension, dimension), dtype=int)
    for left in range(dimension):
        for right in range(left, dimension):
            valid = np.isfinite(values[:, left]) & np.isfinite(values[:, right])
            count = int(valid.sum())
            assert count >= minimum, f"copula pair {left},{right} has {count} samples"
            coefficient = 1.0 if left == right else np.corrcoef(values[valid, left], values[valid, right])[0, 1]
            assert np.isfinite(coefficient), f"copula pair {left},{right} is degenerate"
            correlation[left, right] = correlation[right, left] = coefficient
            counts[left, right] = counts[right, left] = count
    return correlation, counts


def nearest_correlation(matrix: np.ndarray, config: dict) -> np.ndarray:
    """Project symmetrically to the configured positive-definite correlation cone."""
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    floor = float(config["copula"]["eigenvalue_floor"])
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    scale = np.sqrt(np.diag(projected))
    correlation = projected / np.outer(scale, scale)
    condition = np.linalg.cond(correlation)
    assert condition <= config["copula"]["maximum_condition_number"], "correlation is ill-conditioned"
    return correlation


def principal_axis_factor(correlation: np.ndarray, rank: int, config: dict) -> dict:
    """Fit a static principal-axis low-rank plus diagonal correlation model."""
    matrix = nearest_correlation(correlation, config)
    dimension = matrix.shape[0]
    assert 0 < int(rank) < dimension, "invalid factor rank"
    communalities = np.clip(np.max(np.abs(matrix - np.eye(dimension)), axis=1), 0.0, 1.0)
    tolerance = float(config["copula"]["principal_axis_tolerance"])
    for iteration in range(int(config["copula"]["principal_axis_max_iter"])):
        reduced = matrix.copy()
        np.fill_diagonal(reduced, communalities)
        values, vectors = np.linalg.eigh(reduced)
        keep = np.argsort(values)[-int(rank):]
        loadings = vectors[:, keep] * np.sqrt(np.maximum(values[keep], 0.0))
        updated = np.clip(np.sum(loadings ** 2, axis=1), 0.0, 1.0)
        if np.max(np.abs(updated - communalities)) <= tolerance:
            break
        communalities = updated
    else:
        raise AssertionError("principal-axis factor model did not converge")
    uniqueness = np.maximum(1.0 - updated, float(config["copula"]["uniqueness_floor"]))
    sigma = nearest_correlation(loadings @ loadings.T + np.diag(uniqueness), config)
    return {"rank": int(rank), "loadings": loadings, "uniqueness": uniqueness,
            "sigma": sigma, "iterations": iteration + 1}


def apply_ablation(sigma: np.ndarray, target_count: int, periods: int,
                   kind: str, config: dict) -> np.ndarray:
    """Zero cross-target or within-target cross-hour blocks and restore PD."""
    output = np.asarray(sigma, dtype=float).copy()
    assert output.shape == (target_count * periods,) * 2, "ablation shape mismatch"
    if kind == "no_cross_target":
        for left in range(target_count):
            for right in range(target_count):
                if left != right:
                    output[left * periods:(left + 1) * periods,
                           right * periods:(right + 1) * periods] = 0.0
    elif kind == "no_temporal":
        for target in range(target_count):
            block = slice(target * periods, (target + 1) * periods)
            output[block, block] = np.diag(np.diag(output[block, block]))
    else:
        raise AssertionError(f"unknown ablation: {kind}")
    return nearest_correlation(output, config)


def gaussian_uniform_scenarios(sigma: np.ndarray, count: int, seed: int,
                               config: dict) -> np.ndarray:
    """Draw deterministic Gaussian-Copula uniforms."""
    jitter = float(config["copula"]["cholesky_jitter"])
    covariance = np.asarray(sigma, dtype=float) + np.eye(sigma.shape[0]) * jitter
    factor = np.linalg.cholesky(covariance)
    normals = np.random.default_rng(int(seed)).standard_normal((int(count), sigma.shape[0])) @ factor.T
    return stats.norm.cdf(normals)


def empirical_uniform_scenarios(uniforms: np.ndarray, count: int, seed: int,
                                config: dict) -> np.ndarray:
    """Resample empirical rank rows; missing conditional-price ranks use seeded uniforms."""
    assert config["copula"]["empirical_missing_rank_policy"] == "seeded_marginal_uniform"
    values = np.asarray(uniforms, dtype=float)
    generator = np.random.default_rng(int(seed))
    selected = values[generator.integers(0, values.shape[0], size=int(count))].copy()
    missing = ~np.isfinite(selected)
    selected[missing] = generator.uniform(size=int(missing.sum()))
    return selected


def conditional_gaussian(sigma: np.ndarray, conditioned_indices: np.ndarray,
                         conditioned_scores: np.ndarray, count: int, seed: int,
                         config: dict) -> tuple:
    """Draw from the Gaussian Schur-complement conditional without explicit inversion."""
    dimension = sigma.shape[0]
    conditioned = np.asarray(conditioned_indices, dtype=int)
    active = np.setdiff1d(np.arange(dimension), conditioned)
    cc = sigma[np.ix_(conditioned, conditioned)]
    ac = sigma[np.ix_(active, conditioned)]
    aa = sigma[np.ix_(active, active)]
    mean = ac @ np.linalg.solve(cc, conditioned_scores)
    covariance = aa - ac @ np.linalg.solve(cc, ac.T)
    covariance = nearest_correlation_covariance(covariance, config)
    draws = np.random.default_rng(int(seed)).standard_normal((int(count), active.size))
    samples = mean[None, :] + draws @ np.linalg.cholesky(covariance).T
    return active, stats.norm.cdf(samples)


def nearest_correlation_covariance(matrix: np.ndarray, config: dict) -> np.ndarray:
    """Project a conditional covariance to positive definite without unit rescaling."""
    symmetric = (matrix + matrix.T) / 2.0
    floor = float(config["copula"]["eigenvalue_floor"])
    values, vectors = np.linalg.eigh(symmetric)
    projected = (vectors * np.maximum(values, floor)) @ vectors.T
    condition = np.linalg.cond(projected)
    assert condition <= config["copula"]["maximum_condition_number"], "conditional covariance ill-conditioned"
    return projected


def analog_conditioning(training_capacity: np.ndarray, realized_capacity: np.ndarray,
                        count: int, seed: int, config: dict) -> np.ndarray:
    """Return resampled indices among the configured nearest capacity paths."""
    values = np.asarray(training_capacity, dtype=float)
    realized = np.asarray(realized_capacity, dtype=float)
    assert values.ndim == 2 and values.shape[1] == realized.size, "analog capacity shape mismatch"
    distances = np.linalg.norm(values - realized[None, :], axis=1)
    nearest = np.argsort(distances)[:int(config["copula"]["analog_capacity_days"])]
    return np.random.default_rng(int(seed)).choice(nearest, size=int(count), replace=True)
