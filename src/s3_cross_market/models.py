"""Specification refits, dependence and 12:00 day-ahead products for the S3 cross-market study."""

import numpy as np
from scipy import sparse, stats
from scipy.optimize import linprog

from src.s3_forecast_side import copulas, marginals
from src.s3_forecast_side.generators import (TARGETS, _base_scenarios, _conditional_products,
                                             _fit_dependence, _stream, actual_tensor,
                                             quantile_tensor)

from .features import MODIFIED_TARGETS, spec_config


def _relevant_frame(design, blocks: dict, day: str):
    relevant = design["delivery_day"].isin(blocks["fit"] + blocks["calibration"] + [day])
    return design.loc[relevant].copy().reset_index(drop=True)


def linear_program_quantiles(x: np.ndarray, y: np.ndarray, levels: np.ndarray, config: dict) -> tuple:
    """Solve each estimable linear quantile regression exactly as a HiGHS linear program."""
    assert config["cross_market"]["pooled_solver"]["method"] == "highs"
    assert x.shape[0] == y.size and x.shape[0] > x.shape[1], "insufficient pooled samples"
    count, width = x.shape
    spec = marginals._estimable_quantile_spec(levels, y.size)
    identity = sparse.identity(count, format="csr")
    constraints = sparse.hstack([sparse.csr_matrix(x), identity, -identity], format="csr")
    bounds = [(None, None)] * width + [(0.0, None)] * (2 * count)
    parameters, iterations = [], []
    for level in spec["estimated_levels"]:
        cost = np.concatenate([np.zeros(width), np.full(count, level), np.full(count, 1.0 - level)])
        result = linprog(cost, A_eq=constraints, b_eq=y, bounds=bounds, method="highs")
        assert result.status == 0, f"HiGHS quantile regression failed: {result.message}"
        parameters.append(result.x[:width])
        iterations.append(int(result.nit))
    expanded = marginals._expand_quantile_parameters(np.asarray(parameters), spec)
    return expanded, iterations, spec


def _fit_pooled_target(frame, target: str, levels: np.ndarray, config: dict) -> dict:
    columns = marginals.feature_columns(target, config)
    usable = marginals._complete_rows(frame, columns)
    scaling = marginals._fit_scaling(usable, columns, config, usable)
    x = marginals._scaled_matrix(usable, columns, scaling)
    parameters, iterations, spec = linear_program_quantiles(
        x, usable["value"].to_numpy(dtype=float), levels, config)
    audit = {key: value for key, value in spec.items() if key != "mask"}
    fitted = {"parameters": parameters, "scaling": scaling, "iterations": iterations,
              "quantile_audit": {**audit, "target": target, "group": "pooled", "solver": "highs"}}
    return {"kind": "quantreg", "columns": columns, "models": {"pooled": fitted}, "pooled": True}


def fit_target_grid(frame, target: str, blocks: dict, config: dict, seed: int, origin: str) -> dict:
    """Refit one target on F(D), predict its frame rows, and recalibrate exactly as S3-A does."""
    levels = np.asarray(config["scoring"]["quantile_levels"], dtype=float)
    fit = frame.loc[frame["delivery_day"].isin(blocks["fit"]) & frame["target"].eq(target)]
    if config["targets"][target]["model_granularity"] == "pooled_hours":
        model = _fit_pooled_target(fit, target, levels, config)
    else:
        model = marginals._fit_regular_target(fit, target, levels, config, False, origin)
    selected = frame.loc[frame["target"].eq(target)]
    raw = marginals._predict_regular(model, selected)
    calibration = selected["delivery_day"].isin(blocks["calibration"]).to_numpy()
    pits = marginals.pit_values(selected.loc[calibration, "value"].to_numpy(), raw[calibration],
                                levels, int(seed) + TARGETS.index(target))
    mapping = marginals.recalibration_levels(pits, levels)
    calibrated = marginals.evaluate_quantile_grid(raw, levels, np.broadcast_to(mapping, raw.shape))
    return {"positions": frame.index.get_indexer(selected.index), "raw": raw,
            "calibrated": calibrated, "model": model}


def model_diagnostics(model: dict, limit: int) -> dict:
    units = list(model["models"].values())
    iterations = [value for fitted in units for value in fitted["iterations"]]
    limit_hits = int(np.sum(np.asarray(iterations) >= limit))
    assert limit_hits == 0, "QuantReg reached the iteration limit"
    return {"fit_count": len(iterations), "maximum_iterations": int(max(iterations)),
            "unit_count": len(units),
            "dropped_predictors": [name for fitted in units for name in fitted["scaling"]["dropped"]],
            "support_clipped_entries": int(sum(fitted["scaling"]["clipped_entries"] for fitted in units)),
            "endpoint_replacement_cells":
                int(sum(len(fitted["quantile_audit"]["replaced_levels"]) for fitted in units))}


def spec_bundle(base: dict, design, spec: str, config: dict, seed: int, day: str) -> tuple:
    """Replace the day-ahead and capacity rows of an S3-A bundle with one specification's refit."""
    variant = spec_config(config, spec)
    frame = _relevant_frame(design, base["blocks"], day)
    keys = ["delivery_day", "target", "hour"]
    assert frame[keys].equals(base["frame"][keys]), "specification frame differs from the base bundle"
    raw, calibrated, diagnostics = base["raw"].copy(), base["calibrated"].copy(), {}
    limit = int(config["model"]["quantreg"]["max_iter"])
    for target in MODIFIED_TARGETS:
        fitted = fit_target_grid(frame, target, base["blocks"], variant, seed, f"origin={day}, spec={spec}")
        raw[fitted["positions"]] = fitted["raw"]
        calibrated[fitted["positions"]] = fitted["calibrated"]
        diagnostics[target] = model_diagnostics(fitted["model"], limit)
    return {**base, "frame": frame, "raw": raw, "calibrated": calibrated}, diagnostics


def direct_conditional_grid(base: dict, conditional_design, spec: str, config: dict,
                            seed: int, day: str) -> tuple:
    """Fit the 12:00 day-ahead marginal that reads the realized capacity price as predictors."""
    target = config["cross_market"]["conditional_1200"]["target"]
    variant = spec_config(config, spec, conditional=True)
    frame = _relevant_frame(conditional_design, base["blocks"], day)
    fitted = fit_target_grid(frame, target, base["blocks"], variant, seed,
                             f"origin={day}, spec={spec}, gate_1200")
    rows = frame.loc[frame["target"].eq(target)]
    test = rows["delivery_day"].eq(day).to_numpy()
    assert rows.loc[test, "hour"].tolist() == list(range(24)), "12:00 test rows are out of hour order"
    grid = fitted["calibrated"][test]
    assert np.isfinite(grid).all(), "12:00 day-ahead grid is missing"
    return grid, model_diagnostics(fitted["model"], int(config["model"]["quantreg"]["max_iter"]))


def gaussian_linear_conditional(sigma: np.ndarray, conditioned: np.ndarray, weights: np.ndarray,
                                value: float, count: int, seed: int, config: dict) -> tuple:
    """Draw the Gaussian conditional given one linear combination of the conditioned scores."""
    conditioned = np.asarray(conditioned, dtype=int)
    weights = np.asarray(weights, dtype=float)
    active = np.setdiff1d(np.arange(sigma.shape[0]), conditioned)
    cross = sigma[np.ix_(active, conditioned)] @ weights
    variance = float(weights @ sigma[np.ix_(conditioned, conditioned)] @ weights)
    assert variance > 0.0, "conditioning combination has no variance"
    mean = cross * float(value) / variance
    covariance = copulas.nearest_correlation_covariance(
        sigma[np.ix_(active, active)] - np.outer(cross, cross) / variance, config)
    draws = np.random.default_rng(int(seed)).standard_normal((int(count), active.size))
    return active, stats.norm.cdf(mean[None, :] + draws @ np.linalg.cholesky(covariance).T)


def daily_mean_score_product(sigma: np.ndarray, quantiles: np.ndarray, levels: np.ndarray,
                             realized_capacity: np.ndarray, count: int, seed: int,
                             config: dict) -> np.ndarray:
    """Condition on the mean of the 24 capacity normal scores instead of all 24 separately."""
    scores = marginals.pit_values(realized_capacity, quantiles[TARGETS.index("capacity_price")],
                                  levels, seed)
    bounds = config["copula"]["pit_clip"]
    normal = stats.norm.ppf(np.clip(scores, bounds[0], bounds[1]))
    periods = quantiles.shape[1]
    capacity = TARGETS.index("capacity_price")
    conditioned = np.arange(capacity * periods, (capacity + 1) * periods)
    weights = np.full(periods, 1.0 / periods)
    active, uniforms = gaussian_linear_conditional(sigma, conditioned, weights, float(weights @ normal),
                                                   count, seed, config)
    rows = quantiles.reshape(len(TARGETS) * periods, levels.size)[active]
    values = marginals.evaluate_quantile_grid(rows, levels, uniforms.T).T
    return values.reshape(int(count), len(TARGETS) - 1, periods)


def _spec_item(base: dict, bundle: dict, diagnostics: dict, conditional_design, spec: str,
               config: dict, seed: int, day: str) -> dict:
    dependence = _fit_dependence(bundle, seed, config)
    bundle = {**bundle, "dependence_uniforms": dependence["uniforms"]}
    test_quantiles = quantile_tensor(bundle["frame"], bundle["calibrated"], [day], bundle["levels"])[0]
    grid, direct = direct_conditional_grid(base, conditional_design, spec, config, seed, day)
    realized = actual_tensor(bundle["frame"], [day])[0][TARGETS.index("capacity_price")]
    levels = bundle["levels"]
    scores = marginals.pit_values(realized, test_quantiles[TARGETS.index("capacity_price")], levels, seed)
    saturated = int(np.sum((scores <= levels[0]) | (scores >= levels[-1])))
    return {"bundle": bundle, "dependence": dependence, "test_quantiles": test_quantiles,
            "direct_grid": grid,
            "diagnostics": {"quantreg": diagnostics, "direct_quantreg": direct,
                            "rank": int(dependence["chosen"]["rank"]),
                            "conditioning_saturated_hours": saturated}}


def prepare_origin(design, conditional_design, day: str, config: dict) -> dict:
    """Fit the S3-A base bundle once and every specification's refit, dependence and 12:00 grid."""
    seed = int(config["scoring"]["randomized_pit_seed"])
    base = marginals.marginal_day_bundle(design, day, config, seed)
    limit = int(config["model"]["quantreg"]["max_iter"])
    specs = {}
    for spec in config["cross_market"]["specs"]:
        if spec == config["cross_market"]["reference_spec"]:
            bundle = base
            diagnostics = {target: model_diagnostics(base["models"]["targets"][target], limit)
                           for target in MODIFIED_TARGETS}
        else:
            bundle, diagnostics = spec_bundle(base, design, spec, config, seed, day)
        specs[spec] = _spec_item(base, bundle, diagnostics, conditional_design, spec, config, seed, day)
    return {"base": base, "fit_actual": actual_tensor(base["frame"], base["blocks"]["fit"]),
            "specs": specs, "day": day}


def sample_origin(prepared: dict, seed: int, config: dict) -> tuple:
    """Draw 07:30 scenario arms and 12:00 day-ahead products for every specification."""
    count = int(config["generators"]["scenario_count"])
    settings = config["cross_market"]
    levels, day, fit_actual = prepared["base"]["levels"], prepared["day"], prepared["fit_actual"]
    realized = actual_tensor(prepared["base"]["frame"], [day])[0][TARGETS.index("capacity_price")]
    scenarios, conditional = {}, {}
    for spec, item in prepared["specs"].items():
        base = _base_scenarios(fit_actual, item["test_quantiles"], item["dependence"], count, seed, config)
        for name, source in settings["scenario_sources"].items():
            scenarios[f"{spec}__{name}"] = base[source].astype(np.float32)
        products = _conditional_products(item["bundle"], item["dependence"], item["test_quantiles"],
                                         fit_actual, day, seed, config)
        daily_mean = daily_mean_score_product(
            item["dependence"]["chosen"]["sigma"], item["test_quantiles"], levels, realized, count,
            _stream(seed, settings["stream_offsets"]["gaussian_daily_mean_score"]), config)
        direct = marginals.quantiles_to_scenarios(
            item["direct_grid"], levels, count,
            _stream(seed, settings["stream_offsets"]["direct_quantile_regression"])).T
        drawn = {"unconditional": base["fb2_gate"][:, 0], "gaussian_24_hour": products["fb2_gate"][:, 0],
                 "gaussian_daily_mean_score": daily_mean[:, 0], "analog": products["empirical_copula"][:, 0],
                 "direct_quantile_regression": direct}
        assert list(drawn) == settings["conditional_products"], "conditional product list changed"
        conditional.update({f"{spec}__{name}": values.astype(np.float32) for name, values in drawn.items()})
    return scenarios, conditional
