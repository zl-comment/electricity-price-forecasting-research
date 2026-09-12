"""Frozen marginal models and calibration for S3-A."""

from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from sklearn.linear_model import LogisticRegression
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from .panel import window_days


TARGETS = ("day_ahead_price", "capacity_price", "activation_volume", "activation_price")


def feature_columns(target: str, config: dict, weather: bool = False) -> list:
    """Return the configured numeric design columns for one target."""
    columns = []
    boundaries = config["features"]["activation_period_boundaries"]
    for name in config["features"][target]:
        if name == "four_hour_period_indicators":
            columns.extend(f"period_{left:02d}_{right:02d}"
                           for left, right in zip(boundaries[:-1], boundaries[1:]))
        else:
            columns.append(name)
    if weather:
        columns.extend(config["weather"]["feature_columns"])
    return columns


def _fit_scaling(frame: pd.DataFrame, columns: list, config: dict) -> dict:
    assert config["model"]["quantreg"]["predictor_scaling"] == "fit_window_zscore"
    values = frame[columns].to_numpy(dtype=float)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    assert np.isfinite(center).all() and np.isfinite(scale).all()
    return {"center": center, "scale": scale}


def _scaled_matrix(frame: pd.DataFrame, columns: list, scaling: dict) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "marginal design contains missing values"
    standardized = (values - scaling["center"]) / scaling["scale"]
    return sm.add_constant(standardized, has_constant="add")


def _fit_quantile_parameters(x: np.ndarray, y: np.ndarray, levels: np.ndarray,
                             settings: dict, context: str) -> np.ndarray:
    assert x.shape[0] == y.size and x.shape[0] > x.shape[1], "insufficient QuantReg samples"
    parameters = []
    for level in levels:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IterationLimitWarning)
            result = sm.QuantReg(y, x).fit(q=float(level), max_iter=settings["max_iter"],
                                          p_tol=settings["tolerance"])
        assert result.iterations < settings["max_iter"], \
            f"QuantReg did not converge: {context}, quantile={level}"
        assert np.isfinite(result.params).all(), "QuantReg parameters are non-finite"
        parameters.append(result.params)
    return np.asarray(parameters)


def _complete_rows(frame: pd.DataFrame, columns: list, positive: bool = False) -> pd.DataFrame:
    valid = frame[columns].notna().all(axis=1) & frame["value"].notna()
    if positive:
        valid &= frame["value"].gt(0.0)
    return frame.loc[valid]


def _fit_regular_target(frame: pd.DataFrame, target: str, levels: np.ndarray,
                        config: dict, weather: bool, origin: str) -> dict:
    columns = feature_columns(target, config, weather)
    settings = config["model"]["quantreg"]
    pooled = config["targets"][target]["model_granularity"] == "pooled_hours"
    groups = [("pooled", frame)] if pooled else list(frame.groupby("hour", sort=True))
    models = {}
    for key, group in groups:
        usable = _complete_rows(group, columns)
        scaling = _fit_scaling(usable, columns, config)
        x = _scaled_matrix(usable, columns, scaling)
        context = f"origin={origin}, target={target}, group={key}, weather={weather}"
        parameters = _fit_quantile_parameters(
            x, usable["value"].to_numpy(), levels, settings, context)
        models[key] = {"parameters": parameters, "scaling": scaling}
    return {"kind": "quantreg", "columns": columns, "models": models, "pooled": pooled}


def _logistic_arguments(config: dict) -> dict:
    settings = config["model"]["activation_logistic"]
    assert settings["solver"] == "lbfgs", "lbfgs is required for an unpenalized intercept"
    assert settings["fit_intercept"] is True, "Logistic intercept must be fitted"
    assert settings["intercept_penalty"] == "none", "Logistic intercept must be unpenalized"
    return {"penalty": settings["penalty"], "C": settings["C"], "solver": settings["solver"],
            "max_iter": settings["max_iter"], "tol": settings["tolerance"],
            "random_state": settings["random_state"], "fit_intercept": settings["fit_intercept"]}


def _fit_activation_volume(frame: pd.DataFrame, levels: np.ndarray,
                           config: dict, weather: bool, origin: str) -> dict:
    columns = feature_columns("activation_volume", config, weather)
    models = {}
    for hour, group in frame.groupby("hour", sort=True):
        usable = _complete_rows(group, columns)
        labels = usable["value"].eq(0.0).astype(int).to_numpy()
        classes = np.unique(labels)
        assert not (classes.size == 1 and classes[0] == 1), f"activation hour {hour}: only zero events"
        degenerate = classes.size == 1
        if degenerate:
            assert config["model"]["degenerate_zero_rule"] == "p0_zero_when_no_zero_events"
            logistic = None
        else:
            predictors = usable[columns].to_numpy(dtype=float)
            logistic = LogisticRegression(**_logistic_arguments(config)).fit(predictors, labels)
        positive = _complete_rows(group, columns, positive=True)
        scaling = _fit_scaling(positive, columns, config)
        parameters = _fit_quantile_parameters(_scaled_matrix(positive, columns, scaling),
                                               np.log(positive["value"].to_numpy()), levels,
                                               config["model"]["quantreg"],
                                               f"origin={origin}, target=activation_volume, "
                                               f"group={hour}, weather={weather}")
        context = f"origin={origin}, target=activation_volume, group={hour}, weather={weather}"
        models[int(hour)] = {"zero": logistic,
                             "positive": {"parameters": parameters, "scaling": scaling,
                                          "context": context},
                             "degenerate_zero": degenerate}
    return {"kind": "hurdle", "columns": columns, "models": models, "pooled": False}


def fit_marginals(frame: pd.DataFrame, fit_days: list, config: dict,
                  weather: bool, origin: str) -> dict:
    """Fit all four F(D)-only marginal model collections."""
    levels = np.asarray(config["scoring"]["quantile_levels"], dtype=float)
    fit = frame.loc[frame["delivery_day"].isin(fit_days)]
    models = {}
    for target in TARGETS:
        target_frame = fit.loc[fit["target"].eq(target)]
        if target == "activation_volume":
            models[target] = _fit_activation_volume(target_frame, levels, config, weather, origin)
        else:
            models[target] = _fit_regular_target(target_frame, target, levels, config, weather, origin)
    return {"levels": levels, "targets": models, "fit_days": list(fit_days), "weather": weather}


def _predict_regular(model: dict, frame: pd.DataFrame) -> np.ndarray:
    first = next(iter(model["models"].values()))["parameters"]
    output = np.full((len(frame), first.shape[0]), np.nan)
    groups = [("pooled", frame)] if model["pooled"] else list(frame.groupby("hour", sort=True))
    for key, group in groups:
        valid = group[model["columns"]].notna().all(axis=1)
        indices = group.index[valid]
        if len(indices) == 0:
            continue
        fitted = model["models"][key]
        x = _scaled_matrix(group.loc[indices], model["columns"], fitted["scaling"])
        output[frame.index.get_indexer(indices)] = x @ fitted["parameters"].T
    return np.sort(output, axis=1)


def _positive_quantiles(parameters: np.ndarray, x: np.ndarray,
                        transformed_levels: np.ndarray, levels: np.ndarray,
                        context: str) -> np.ndarray:
    linear = x @ parameters.T
    limits = np.log([np.finfo(float).tiny, np.finfo(float).max])
    assert np.isfinite(linear).all() and linear.min() >= limits[0] and linear.max() <= limits[1], \
        f"activation positive quantile exponent is out of range: {context}"
    grid = np.exp(linear)
    output = np.empty_like(transformed_levels)
    for row in range(grid.shape[0]):
        output[row] = np.interp(transformed_levels[row], levels, np.sort(grid[row]),
                                left=np.min(grid[row]), right=np.max(grid[row]))
    return output


def _mixed_quantiles(positive: dict, x: np.ndarray, zero_probability: np.ndarray,
                     levels: np.ndarray) -> np.ndarray:
    active = levels[None, :] > zero_probability[:, None]
    output = np.zeros(active.shape, dtype=float)
    rows = active.any(axis=1)
    if not rows.any():
        return output
    probabilities = zero_probability[rows]
    transformed = (levels[None, :] - probabilities[:, None]) / (1.0 - probabilities[:, None])
    transformed = np.clip(transformed, levels[0], levels[-1])
    values = _positive_quantiles(positive["parameters"], x[rows], transformed, levels,
                                 positive["context"])
    values[~active[rows]] = 0.0
    output[rows] = values
    return output


def _predict_hurdle(model: dict, frame: pd.DataFrame, levels: np.ndarray) -> np.ndarray:
    output = np.full((len(frame), levels.size), np.nan)
    for hour, group in frame.groupby("hour", sort=True):
        valid = group[model["columns"]].notna().all(axis=1)
        indices = group.index[valid]
        if len(indices) == 0:
            continue
        fitted = model["models"][int(hour)]
        positive = fitted["positive"]
        x = _scaled_matrix(group.loc[indices], model["columns"], positive["scaling"])
        predictors = group.loc[indices, model["columns"]].to_numpy(dtype=float)
        zero_probability = np.zeros(len(indices)) if fitted["degenerate_zero"] else \
            fitted["zero"].predict_proba(predictors)[:, 1]
        assert np.isfinite(zero_probability).all()
        values = _mixed_quantiles(positive, x, zero_probability, levels)
        output[frame.index.get_indexer(indices)] = np.sort(values, axis=1)
    return output


def predict_marginals(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict rearranged raw quantiles for frame rows in their existing order."""
    output = np.full((len(frame), bundle["levels"].size), np.nan)
    for target, model in bundle["targets"].items():
        selected = frame.loc[frame["target"].eq(target)]
        predicted = _predict_hurdle(model, selected, bundle["levels"]) \
            if model["kind"] == "hurdle" else _predict_regular(model, selected)
        output[frame.index.get_indexer(selected.index)] = predicted
    return output


def pit_values(actual: np.ndarray, quantiles: np.ndarray, levels: np.ndarray,
               seed: int) -> np.ndarray:
    """Invert rearranged quantiles, randomizing probability mass at tied values."""
    generator = np.random.default_rng(int(seed))
    output = np.full(actual.shape, np.nan)
    for row, value in enumerate(actual):
        grid = quantiles[row]
        if not np.isfinite(value) or not np.isfinite(grid).all():
            continue
        equal = np.isclose(grid, value, rtol=0.0, atol=1e-12)
        if equal.any():
            bounds = levels[equal]
            output[row] = generator.uniform(bounds.min(), bounds.max())
        else:
            unique, indices = np.unique(grid, return_index=True)
            output[row] = np.interp(value, unique, levels[indices], left=levels[0], right=levels[-1])
    return output


def recalibration_levels(pits: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Return target-level empirical inverse PIT maps from C(D) only."""
    valid = pits[np.isfinite(pits)]
    assert valid.size > 0, "calibration PIT sample is empty"
    return np.quantile(valid, levels, interpolation="linear")


def evaluate_quantile_grid(grid: np.ndarray, grid_levels: np.ndarray,
                           requested: np.ndarray) -> np.ndarray:
    """Evaluate row-specific piecewise-linear quantile functions with endpoint tails."""
    request = np.asarray(requested, dtype=float)
    output = np.empty((grid.shape[0],) + request.shape[1:], dtype=float)
    for row in range(grid.shape[0]):
        output[row] = np.interp(request[row], grid_levels, grid[row],
                                left=grid[row, 0], right=grid[row, -1])
    return output


def recalibrate_predictions(frame: pd.DataFrame, raw: np.ndarray, levels: np.ndarray,
                            calibration_days: list, seed: int) -> tuple:
    """Apply one pooled C(D) empirical PIT map per target."""
    adjusted = np.full_like(raw, np.nan)
    maps = {}
    for offset, target in enumerate(TARGETS):
        selected = frame["target"].eq(target).to_numpy()
        calibration = selected & frame["delivery_day"].isin(calibration_days).to_numpy()
        pits = pit_values(frame.loc[calibration, "value"].to_numpy(), raw[calibration], levels,
                          int(seed) + offset)
        mapping = recalibration_levels(pits, levels)
        maps[target] = mapping
        requested = np.broadcast_to(mapping, (int(selected.sum()), levels.size))
        adjusted[selected] = evaluate_quantile_grid(raw[selected], levels, requested)
    return adjusted, maps


def quantiles_to_scenarios(quantiles: np.ndarray, levels: np.ndarray,
                           scenario_count: int, seed: int) -> np.ndarray:
    """Sample independent row-wise scenarios from calibrated marginal grids."""
    uniforms = np.random.default_rng(int(seed)).uniform(size=(quantiles.shape[0], scenario_count))
    return evaluate_quantile_grid(quantiles, levels, uniforms)


def _lgbm_parameters(config: dict, repository_root: Path) -> dict:
    path = repository_root / config["model"]["fb0_config"]
    reference = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]["params"]
    return dict(reference)


def fit_fb0(frame: pd.DataFrame, fit_days: list, config: dict,
            repository_root: Path, target: str, extra_columns: list = None) -> dict:
    """Fit target/hour LightGBM point models with exactly the frozen F08 parameters."""
    columns = feature_columns(target, config) + list(extra_columns or [])
    selected = frame.loc[frame["delivery_day"].isin(fit_days) & frame["target"].eq(target)]
    pooled = config["targets"][target]["model_granularity"] == "pooled_hours"
    groups = [("pooled", selected)] if pooled else list(selected.groupby("hour", sort=True))
    models = {}
    for key, group in groups:
        usable = _complete_rows(group, columns)
        model = lgb.LGBMRegressor(**_lgbm_parameters(config, repository_root))
        model.fit(usable[columns].to_numpy(dtype=float), usable["value"].to_numpy(dtype=float))
        models[key] = model
    return {"columns": columns, "models": models, "pooled": pooled, "target": target}


def predict_fb0(model: dict, frame: pd.DataFrame, config: dict) -> np.ndarray:
    """Predict one target's LightGBM point values, preserving physical lower bounds."""
    output = np.full(len(frame), np.nan)
    groups = [("pooled", frame)] if model["pooled"] else list(frame.groupby("hour", sort=True))
    for key, group in groups:
        assert group[model["columns"]].notna().all().all(), "FB0 test feature is missing"
        output[frame.index.get_indexer(group.index)] = model["models"][key].predict(
            group[model["columns"]].to_numpy(dtype=float))
    lower = config["targets"][model["target"]]["physical_lower_bound"]
    return np.maximum(output, lower) if lower is not None else output


def marginal_day_bundle(design: pd.DataFrame, delivery_day: str, config: dict,
                        seed: int, weather: bool = False) -> dict:
    """Fit F(D), recalibrate on C(D), and return test and fit marginal objects."""
    blocks = window_days(delivery_day, config)
    relevant = design["delivery_day"].isin(blocks["fit"] + blocks["calibration"] + [delivery_day])
    frame = design.loc[relevant].copy().reset_index(drop=True)
    bundle = fit_marginals(frame, blocks["fit"], config, weather, delivery_day)
    raw = predict_marginals(bundle, frame)
    calibrated, maps = recalibrate_predictions(frame, raw, bundle["levels"],
                                                blocks["calibration"], seed)
    return {"frame": frame, "models": bundle, "raw": raw, "calibrated": calibrated,
            "maps": maps, "blocks": blocks, "levels": bundle["levels"]}
