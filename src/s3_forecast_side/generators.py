"""Scenario-arm assembly for the frozen S3-A experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .copulas import (analog_conditioning, apply_ablation, conditional_gaussian,
                      empirical_uniform_scenarios, gaussian_uniform_scenarios,
                      normal_scores, pairwise_correlation, principal_axis_factor)
from .marginals import (TARGETS, evaluate_quantile_grid, marginal_day_bundle,
                        pit_values, quantiles_to_scenarios)
from .scoring import energy_score


def _stream(seed: int, offset: int) -> int:
    """Derive an independent stream from a day seed without colliding across days or arms."""
    return int(np.random.SeedSequence([int(seed), int(offset)]).generate_state(1)[0])


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(TARGETS)}
    result = frame.copy()
    result["target_order"] = result["target"].map(order)
    return result.sort_values(["delivery_day", "target_order", "hour"]).drop(columns="target_order")


def actual_tensor(frame: pd.DataFrame, days: list) -> np.ndarray:
    """Return day-target-hour labels in the frozen target order."""
    selected = _ordered(frame.loc[frame["delivery_day"].isin(days)])
    expected = len(days) * len(TARGETS) * 24
    assert len(selected) == expected, "actual tensor rows are incomplete"
    assert selected["delivery_day"].drop_duplicates().tolist() == days, "actual days are out of order"
    return selected["value"].to_numpy(dtype=float).reshape(len(days), len(TARGETS), 24)


def quantile_tensor(frame: pd.DataFrame, quantiles: np.ndarray, days: list,
                    levels: np.ndarray) -> np.ndarray:
    """Return day-target-hour-quantile arrays in frozen order."""
    positions = _ordered(frame.assign(_position=np.arange(len(frame)))).loc[
        lambda value: value["delivery_day"].isin(days), "_position"].to_numpy(dtype=int)
    expected = len(days) * len(TARGETS) * 24
    assert positions.size == expected, "quantile tensor rows are incomplete"
    return quantiles[positions].reshape(len(days), len(TARGETS), 24, levels.size)


def _pit_matrix(bundle: dict, seed: int) -> np.ndarray:
    frame, raw, levels = bundle["frame"], bundle["raw"], bundle["levels"]
    fit_days = bundle["blocks"]["fit"]
    rows = _ordered(frame.assign(_position=np.arange(len(frame))))
    rows = rows.loc[rows["delivery_day"].isin(fit_days)]
    output = np.full(len(rows), np.nan)
    for offset, target in enumerate(TARGETS):
        selected = rows["target"].eq(target).to_numpy()
        positions = rows.loc[selected, "_position"].to_numpy(dtype=int)
        output[selected] = pit_values(rows.loc[selected, "value"].to_numpy(), raw[positions], levels,
                                      int(seed) + offset)
    return output.reshape(len(fit_days), len(TARGETS) * 24)


def _scenario_from_uniforms(quantiles: np.ndarray, levels: np.ndarray,
                            uniforms: np.ndarray) -> np.ndarray:
    rows = quantiles.reshape(len(TARGETS) * 24, levels.size)
    requested = uniforms.T
    values = evaluate_quantile_grid(rows, levels, requested).T
    return values.reshape(uniforms.shape[0], len(TARGETS), 24)


def _independent_scenarios(quantiles: np.ndarray, levels: np.ndarray,
                           count: int, seed: int) -> np.ndarray:
    rows = quantiles.reshape(len(TARGETS) * 24, levels.size)
    sampled = quantiles_to_scenarios(rows, levels, count, seed).T
    return sampled.reshape(count, len(TARGETS), 24)


def _empirical_hourly(fit_actual: np.ndarray, count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    output = np.empty((count, len(TARGETS), 24), dtype=float)
    for target in range(len(TARGETS)):
        for hour in range(24):
            values = fit_actual[:, target, hour]
            values = values[np.isfinite(values)]
            assert values.size > 0, "empty climatology cell"
            output[:, target, hour] = generator.choice(values, size=count, replace=True)
    return output


def _conditional_price_draw(values: np.ndarray, generator: np.random.Generator) -> float:
    valid = values[np.isfinite(values)]
    assert valid.size > 0, "activation-price history is empty"
    return float(generator.choice(valid))


def _historical_scenarios(fit_actual: np.ndarray, count: int, seed: int,
                          paired: bool) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    days = fit_actual.shape[0]
    if paired:
        indices = np.broadcast_to(generator.integers(0, days, size=(count, 1)), (count, len(TARGETS)))
    else:
        indices = generator.integers(0, days, size=(count, len(TARGETS)))
    output = np.empty((count, len(TARGETS), 24), dtype=float)
    for scenario in range(count):
        for target in range(len(TARGETS)):
            output[scenario, target] = fit_actual[indices[scenario, target], target]
        for hour in range(24):
            if not np.isfinite(output[scenario, 3, hour]):
                output[scenario, 3, hour] = _conditional_price_draw(fit_actual[:, 3, hour], generator)
    return output


def _target_scales(fit_actual: np.ndarray) -> np.ndarray:
    scales = np.nanstd(fit_actual, axis=(0, 2), ddof=1)
    assert np.isfinite(scales).all() and np.all(scales > 0.0), "invalid F(D) target scales"
    return scales


def _rank_score(bundle: dict, sigma: np.ndarray, seed: int, config: dict) -> float:
    days = bundle["blocks"]["calibration"]
    actual = actual_tensor(bundle["frame"], days)
    quantiles = quantile_tensor(bundle["frame"], bundle["calibrated"], days, bundle["levels"])
    fit_actual = actual_tensor(bundle["frame"], bundle["blocks"]["fit"])
    scales = _target_scales(fit_actual)
    scores = []
    joint = [0, 1, 2]
    count = int(config["generators"]["scenario_count"])
    for index, observation in enumerate(actual):
        uniforms = gaussian_uniform_scenarios(sigma, count, int(seed) + index, config)
        scenarios = _scenario_from_uniforms(quantiles[index], bundle["levels"], uniforms)
        valid = np.isfinite(observation[joint]).reshape(-1)
        observed = (observation[joint] / scales[joint, None]).reshape(-1)[valid]
        draws = (scenarios[:, joint] / scales[None, joint, None]).reshape(count, -1)[:, valid]
        scores.append(energy_score(observed, draws, config["scoring"]["energy_score_beta"]))
    return float(np.mean(scores))


def _fit_dependence(bundle: dict, seed: int, config: dict) -> dict:
    uniforms = _pit_matrix(bundle, seed)
    correlation, counts = pairwise_correlation(normal_scores(uniforms, config), config)
    candidates = []
    for rank in config["copula"]["rank_candidates"]:
        model = principal_axis_factor(correlation, int(rank), config)
        model["calibration_energy_score"] = _rank_score(bundle, model["sigma"], seed, config)
        candidates.append(model)
    chosen = min(candidates, key=lambda item: item["calibration_energy_score"])
    return {"uniforms": uniforms, "correlation": correlation, "counts": counts,
            "candidates": candidates, "chosen": chosen}


def _x09_scenarios(test_quantiles: np.ndarray, levels: np.ndarray, fit_uniforms: np.ndarray,
                   count: int, seed: int, config: dict) -> np.ndarray:
    price_columns = np.r_[0:24, 24:48, 72:96]
    activation_columns = np.arange(48, 72)
    price_uniforms = empirical_uniform_scenarios(fit_uniforms[:, price_columns], count, seed, config)
    generator = np.random.default_rng(_stream(seed, 1))
    indices = generator.integers(0, fit_uniforms.shape[0], size=count)
    activation_uniforms = fit_uniforms[indices][:, activation_columns].copy()
    missing = ~np.isfinite(activation_uniforms)
    activation_uniforms[missing] = generator.uniform(size=int(missing.sum()))
    combined = np.empty((count, len(TARGETS) * 24), dtype=float)
    combined[:, price_columns] = price_uniforms
    combined[:, activation_columns] = activation_uniforms
    return _scenario_from_uniforms(test_quantiles, levels, combined)


def _conditional_from(sigma: np.ndarray, quantiles: np.ndarray, levels: np.ndarray,
                      realized_capacity: np.ndarray, count: int, seed: int,
                      config: dict) -> np.ndarray:
    """Condition one Gaussian copula on the published capacity price and sample 72 dimensions."""
    scores = pit_values(realized_capacity, quantiles[1], levels, seed)
    bounds = config["copula"]["pit_clip"]
    conditioned = stats.norm.ppf(np.clip(scores, bounds[0], bounds[1]))
    active, uniforms = conditional_gaussian(sigma, np.arange(24, 48), conditioned,
                                            count, seed, config)
    rows = quantiles.reshape(len(TARGETS) * 24, levels.size)[active]
    return evaluate_quantile_grid(rows, levels, uniforms.T).T.reshape(count, 3, 24)


def _conditional_products(bundle: dict, dependence: dict, test_quantiles: np.ndarray,
                          fit_actual: np.ndarray, day: str, seed: int, config: dict,
                          weather_quantiles: np.ndarray = None) -> dict:
    actual = actual_tensor(bundle["frame"], [day])[0]
    levels = bundle["levels"]
    count = int(config["generators"]["scenario_count"])
    sigma = dependence["chosen"]["sigma"]
    no_temporal = apply_ablation(sigma, len(TARGETS), 24, "no_temporal", config)
    output = {
        "fb2_gate": _conditional_from(sigma, test_quantiles, levels, actual[1], count,
                                      _stream(seed, 10), config),
        "fb2_no_temporal": _conditional_from(no_temporal, test_quantiles, levels, actual[1],
                                             count, _stream(seed, 11), config),
        "empirical_copula": _analog_product(bundle, test_quantiles, fit_actual, actual[1],
                                            count, _stream(seed, 12), config),
    }
    if weather_quantiles is not None:
        output["fb2_plus_weather"] = _conditional_from(
            sigma, weather_quantiles, levels, actual[1], count, _stream(seed, 13), config) \
            if np.isfinite(weather_quantiles).all() else np.full((count, 3, 24), np.nan)
    return output


def _analog_product(bundle: dict, quantiles: np.ndarray, fit_actual: np.ndarray,
                    realized_capacity: np.ndarray, count: int, seed: int, config: dict) -> np.ndarray:
    indices = analog_conditioning(fit_actual[:, 1], realized_capacity, count, seed, config)
    uniforms = bundle["dependence_uniforms"][indices].copy()
    active_columns = np.r_[0:24, 48:96]
    selected = uniforms[:, active_columns]
    generator = np.random.default_rng(_stream(seed, 1))
    missing = ~np.isfinite(selected)
    selected[missing] = generator.uniform(size=int(missing.sum()))
    rows = quantiles.reshape(len(TARGETS) * 24, bundle["levels"].size)[active_columns]
    return evaluate_quantile_grid(rows, bundle["levels"], selected.T).T.reshape(count, 3, 24)


def _test_p0(bundle: dict, delivery_day: str) -> np.ndarray:
    frame = bundle["frame"]
    selected = frame["delivery_day"].eq(delivery_day) & frame["target"].eq("activation_volume")
    rows = frame.loc[selected].sort_values("hour")
    assert len(rows) == 24 and rows["hour"].tolist() == list(range(24))
    return bundle["zero_probability"][rows.index.to_numpy()]


def prepare_day(design: pd.DataFrame, delivery_day: str, config: dict,
                weather_design: pd.DataFrame = None) -> dict:
    """Fit one rolling origin once, including validation-only rank selection."""
    bundle = marginal_day_bundle(design, delivery_day, config,
                                 config["scoring"]["randomized_pit_seed"])
    fit_days = bundle["blocks"]["fit"]
    fit_actual = actual_tensor(bundle["frame"], fit_days)
    test_quantiles = quantile_tensor(bundle["frame"], bundle["calibrated"],
                                     [delivery_day], bundle["levels"])[0]
    dependence = _fit_dependence(bundle, config["scoring"]["randomized_pit_seed"], config)
    bundle["dependence_uniforms"] = dependence["uniforms"]
    base_columns = set(bundle["frame"].columns)
    assert not base_columns & set(config["weather"]["feature_columns"]), \
        "only the FB2+ upper bound may carry delivery-day weather columns"
    weather_quantiles, weather_p0, weather_diagnostics = None, None, None
    if weather_design is not None:
        assert config["weather"]["allowed_arm"] == "fb2_plus_weather"
        weather_bundle = marginal_day_bundle(weather_design, delivery_day, config,
                                             config["scoring"]["randomized_pit_seed"], True)
        weather_quantiles = quantile_tensor(weather_bundle["frame"], weather_bundle["calibrated"],
                                            [delivery_day], bundle["levels"])[0]
        weather_p0 = _test_p0(weather_bundle, delivery_day)
        weather_diagnostics = weather_bundle["quantreg_diagnostics"]
    return {"bundle": bundle, "test_quantiles": test_quantiles, "fit_actual": fit_actual,
            "target_scales": _target_scales(fit_actual), "dependence": dependence,
            "weather_quantiles": weather_quantiles, "delivery_day": delivery_day,
            "p0": {"fb1_qr": _test_p0(bundle, delivery_day),
                   "fb2_plus_weather": weather_p0},
            "quantreg_diagnostics": {"fb1_qr": bundle["quantreg_diagnostics"],
                                      "fb2_plus_weather": weather_diagnostics}}


def sample_prepared_day(prepared: dict, seed: int, config: dict) -> dict:
    """Sample every runnable arm from a fitted rolling origin for one base seed."""
    count = int(config["generators"]["scenario_count"])
    scenarios = _base_scenarios(prepared["fit_actual"], prepared["test_quantiles"],
                                prepared["dependence"], count, seed, config)
    if prepared["weather_quantiles"] is not None:
        uniforms = gaussian_uniform_scenarios(prepared["dependence"]["chosen"]["sigma"],
                                               count, _stream(seed, 9), config)
        scenarios["fb2_plus_weather"] = _scenario_from_uniforms(
            prepared["weather_quantiles"], prepared["bundle"]["levels"], uniforms)
    conditional = _conditional_products(
        prepared["bundle"], prepared["dependence"], prepared["test_quantiles"],
        prepared["fit_actual"], prepared["delivery_day"], seed, config,
        prepared["weather_quantiles"])
    return {"scenarios": scenarios, "conditional": conditional}


def generate_day(design: pd.DataFrame, delivery_day: str, config: dict,
                 seed: int, weather_design: pd.DataFrame = None) -> dict:
    """Fit and sample one rolling origin; retained as a check-friendly public wrapper."""
    prepared = prepare_day(design, delivery_day, config, weather_design)
    return {**prepared, **sample_prepared_day(prepared, seed, config)}


def _base_scenarios(fit_actual: np.ndarray, quantiles: np.ndarray, dependence: dict,
                    count: int, seed: int, config: dict) -> dict:
    levels = np.asarray(config["scoring"]["quantile_levels"], dtype=float)
    empirical = empirical_uniform_scenarios(dependence["uniforms"], count, _stream(seed, 3), config)
    gaussian = gaussian_uniform_scenarios(dependence["chosen"]["sigma"], count,
                                          _stream(seed, 4), config)
    no_cross = apply_ablation(dependence["chosen"]["sigma"], len(TARGETS), 24,
                              "no_cross_target", config)
    no_temporal = apply_ablation(dependence["chosen"]["sigma"], len(TARGETS), 24,
                                 "no_temporal", config)
    return {
        "climatology": _empirical_hourly(fit_actual, count, _stream(seed, 0)),
        "fb1_qr": _independent_scenarios(quantiles, levels, count, _stream(seed, 1)),
        "hist_paired": _historical_scenarios(fit_actual, count, _stream(seed, 2), True),
        "hist_independent": _historical_scenarios(fit_actual, count, _stream(seed, 21), False),
        "empirical_copula": _scenario_from_uniforms(quantiles, levels, empirical),
        "fb2_gate": _scenario_from_uniforms(quantiles, levels, gaussian),
        "fb2_no_cross_target": _scenario_from_uniforms(
            quantiles, levels, gaussian_uniform_scenarios(no_cross, count, _stream(seed, 5), config)),
        "fb2_no_temporal": _scenario_from_uniforms(
            quantiles, levels, gaussian_uniform_scenarios(no_temporal, count, _stream(seed, 6), config)),
        "x09_block_copula": _x09_scenarios(quantiles, levels, dependence["uniforms"],
                                            count, _stream(seed, 7), config),
    }


def build_weather_design(design: pd.DataFrame, config: dict, repository_root: Path) -> pd.DataFrame:
    """Attach D-day ForecastDayAhead only for the explicit FB2+ upper-bound fit."""
    dataset = config["data"]["datasets"]["wind_solar_forecasts"]
    path = repository_root / config["data"]["root"] / dataset["csv"]
    columns = [dataset["time_column"], dataset["area_column"], dataset["forecast_type_column"],
               config["weather"]["value_column"]]
    raw = pd.read_csv(path, usecols=columns)
    raw = raw.loc[raw[dataset["area_column"]].eq(config["data"]["primary_price_area"])]
    raw["timestamp"] = pd.to_datetime(raw[dataset["time_column"]], utc=True)
    local = raw["timestamp"].dt.tz_convert(config["protocol"]["civil_timezone"])
    raw["delivery_day"] = local.dt.strftime("%Y-%m-%d")
    raw = raw.sort_values(["delivery_day", dataset["forecast_type_column"], "timestamp"])
    parts = []
    for (_, kind), group in raw.groupby(["delivery_day", dataset["forecast_type_column"]], sort=True):
        if kind in config["weather"]["type_mapping"]:
            parts.append(_normalise_weather(group, kind, config))
    weather = pd.concat(parts, ignore_index=True).pivot(index=["delivery_day", "hour"],
                                                        columns="feature", values="value").reset_index()
    _validate_weather_source(weather, config)
    merged = design.merge(weather, on=["delivery_day", "hour"], how="left", validate="many_to_one")
    _validate_weather_missingness(merged, config)
    return merged


def _validate_weather_source(weather: pd.DataFrame, config: dict) -> None:
    columns = config["weather"]["feature_columns"]
    missing = weather.loc[weather[columns].isna().any(axis=1)]
    expected = set(config["weather"]["missing_test_days"]) | set(
        config["weather"]["non_test_all_type_missing_days"])
    assert set(missing["delivery_day"]) == expected, "raw weather missing-day boundary changed"
    mapping = config["weather"]["type_mapping"]
    expected_by_day = {day: [mapping[name] for name in kinds]
                       for day, kinds in config["weather"]["missing_test_days"].items()}
    for day in config["weather"]["non_test_all_type_missing_days"]:
        expected_by_day[day] = columns
    for day, expected_features in expected_by_day.items():
        rows = missing.loc[missing["delivery_day"].eq(day)]
        observed = [column for column in columns if rows[column].isna().all()]
        assert observed == expected_features, f"{day}: raw missing ForecastType changed"


def _validate_weather_missingness(design: pd.DataFrame, config: dict) -> None:
    columns = config["weather"]["feature_columns"]
    missing = design.loc[design[columns].isna().any(axis=1), ["delivery_day", "hour"] + columns]
    missing_days = set(missing["delivery_day"])
    allowed = set(config["weather"]["missing_test_days"]) | set(
        config["weather"]["non_test_all_type_missing_days"])
    expected = allowed & set(design["delivery_day"])
    assert missing_days == expected, f"unexpected FB2+ model missing days: {sorted(missing_days ^ expected)}"
    test_missing = set(config["weather"]["missing_test_days"])
    for day in test_missing:
        rows = missing.loc[missing["delivery_day"].eq(day)]
        assert len(rows) == 24 * len(TARGETS), f"{day}: authorized full-day missingness changed"
        missing_features = [column for column in columns if rows[column].isna().all()]
        expected_types = config["weather"]["missing_test_days"][day]
        mapping = config["weather"]["type_mapping"]
        assert missing_features == [mapping[name] for name in expected_types]


def _normalise_weather(group: pd.DataFrame, kind: str, config: dict) -> pd.DataFrame:
    periods = int(config["protocol"]["periods_per_day"])
    values = group[["delivery_day", config["weather"]["value_column"]]].reset_index(drop=True)
    if len(values) > periods:
        values = values.iloc[:periods].copy()
    elif len(values) < periods:
        assert len(values) > 0, "empty weather day"
        values = pd.concat([values, pd.concat([values.iloc[[-1]]] * (periods - len(values)))],
                           ignore_index=True)
    values["hour"] = np.arange(periods)
    values["feature"] = config["weather"]["type_mapping"][kind]
    values["value"] = values[config["weather"]["value_column"]]
    return values[["delivery_day", "hour", "feature", "value"]]
