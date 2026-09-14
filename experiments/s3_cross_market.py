"""Run the exploratory S3 cross-market marginal study on the frozen S3-A protocol."""

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import platform
from pathlib import Path
import sys
from unittest import TestCase

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.s3_forecast_side import panel, scoring
from src.s3_forecast_side.generators import TARGETS, _target_scales, actual_tensor
from src.s3_cross_market import features, models


WORKER_STATE = {}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load the frozen S3-A config and attach this study's own sections."""
    study = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / study["base_config"]).read_text(encoding="utf-8"))
    assert config["protocol"]["missing_value_policy"] == "preserve"
    assert config["protocol"]["time_split_policy"] == "chronological"
    assert "cross_market" not in config, "base config already carries a cross_market section"
    for key in ["cross_market", "runtime", "output"]:
        config[key] = study[key]
    config["base_config_path"] = study["base_config"]
    return config


def environment_fingerprint() -> dict:
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__}


def build_designs(config: dict) -> tuple:
    hourly_panel = panel.build_hourly_panel(config, ROOT)
    design = panel.build_all_features(hourly_panel, "gate_0730", config)
    cross = features.build_cross_design(design, hourly_panel, config)
    conditional = features.build_conditional_design(cross, hourly_panel, config)
    return hourly_panel, cross, conditional


def _init_worker(config: dict, design: pd.DataFrame, conditional: pd.DataFrame) -> None:
    WORKER_STATE.update({"config": config, "design": design, "conditional": conditional})


def _day_worker(item: tuple) -> dict:
    index, day = item
    config = WORKER_STATE["config"]
    prepared = models.prepare_origin(WORKER_STATE["design"], WORKER_STATE["conditional"], day, config)
    seed = int(config["cross_market"]["sampling_seed"]) + index
    scenarios, conditional = models.sample_origin(prepared, seed, config)
    return {"index": index, "day": day,
            "actual": actual_tensor(prepared["base"]["frame"], [day])[0],
            "scales": _target_scales(prepared["fit_actual"]),
            "quantiles": {spec: item["test_quantiles"] for spec, item in prepared["specs"].items()},
            "direct": {spec: item["direct_grid"] for spec, item in prepared["specs"].items()},
            "scenarios": scenarios, "conditional": conditional,
            "diagnostics": {spec: item["diagnostics"] for spec, item in prepared["specs"].items()}}


def _store(storage: dict, result: dict, day_count: int) -> None:
    index = result["index"]
    storage["actual"][index] = result["actual"]
    storage["scales"][index] = result["scales"]
    for group in ["quantiles", "direct", "scenarios", "conditional"]:
        for key, values in result[group].items():
            storage[group].setdefault(key, np.full((day_count,) + values.shape, np.nan,
                                                   dtype=values.dtype))[index] = values
    storage["diagnostics"].append({"day": result["day"], **result["diagnostics"]})


def run_origins(config: dict, design: pd.DataFrame, conditional: pd.DataFrame, days: list) -> dict:
    storage = {"actual": np.empty((len(days), 4, 24)), "scales": np.empty((len(days), 4)),
               "quantiles": {}, "direct": {}, "scenarios": {}, "conditional": {}, "diagnostics": []}
    context = mp.get_context("fork")
    with context.Pool(int(config["runtime"]["worker_processes"]), initializer=_init_worker,
                      initargs=(config, design, conditional)) as pool:
        for completed, result in enumerate(pool.imap_unordered(_day_worker, enumerate(days)), 1):
            _store(storage, result, len(days))
            if completed % 10 == 0 or completed == len(days):
                print(json.dumps({"completed_origins": completed, "total_origins": len(days)}), flush=True)
    storage["diagnostics"].sort(key=lambda row: row["day"])
    return storage


def _target_mask(actual: np.ndarray, target_index: int) -> np.ndarray:
    mask = np.isfinite(actual[:, target_index])
    if TARGETS[target_index] == "activation_price":
        mask &= actual[:, TARGETS.index("activation_volume")] > 0.0
    return mask


def _flat_scenarios(values: np.ndarray, target_index: int) -> np.ndarray:
    return values[:, :, target_index, :].transpose(0, 2, 1).reshape(-1, values.shape[1])


def marginal_rows(storage: dict, config: dict) -> list:
    rows = []
    levels = config["scoring"]["quantile_levels"]
    seed = int(config["cross_market"]["sampling_seed"])
    for spec in config["cross_market"]["specs"]:
        scenarios = storage["scenarios"][f"{spec}__independent"]
        for index, target in enumerate(TARGETS):
            result = scoring.marginal_metrics(
                storage["actual"][:, index].reshape(-1),
                storage["quantiles"][spec][:, index].reshape(-1, len(levels)), levels,
                _flat_scenarios(scenarios, index), config["scoring"]["central_interval_levels"], seed,
                _target_mask(storage["actual"], index).reshape(-1))
            base = {"spec": spec, "target": target, "mean_pinball": result["mean_pinball"],
                    "sample_crps": result["sample_crps"], "pit_ks_distance": result["pit_ks_distance"],
                    "n_valid": result["n_valid"],
                    "pinball_by_quantile": ";".join(f"{value:.6f}" for value in result["pinball_by_quantile"])}
            rows.extend({**base, **interval} for interval in result["central_intervals"])
    return rows


def daily_pinball_rows(storage: dict, days: list, config: dict) -> list:
    rows = []
    levels = config["scoring"]["quantile_levels"]
    for spec in config["cross_market"]["specs"]:
        for target in features.MODIFIED_TARGETS:
            index = TARGETS.index(target)
            mask = _target_mask(storage["actual"], index)
            for day_index, day in enumerate(days):
                result = scoring.pinball_metrics(storage["actual"][day_index, index],
                                                 storage["quantiles"][spec][day_index, index],
                                                 levels, mask[day_index])
                rows.append({"delivery_day": day, "arm": spec, "target": target,
                             "mean_pinball": result["mean_pinball"], "n_valid": result["n_valid"]})
    return rows


def monthly_rows(storage: dict, days: list, config: dict) -> list:
    rows = []
    for spec in config["cross_market"]["specs"]:
        values = storage["scenarios"][f"{spec}__independent"]
        for target in features.MODIFIED_TARGETS:
            index = TARGETS.index(target)
            results = scoring.monthly_calibration_metrics(
                days, storage["actual"][:, index], values[:, :, index],
                config["scoring"]["monthly_calibration_level"], _target_mask(storage["actual"], index))
            rows.extend({"spec": spec, "target": target, **row} for row in results)
    return rows


def daily_joint_rows(storage: dict, days: list, config: dict) -> list:
    rows = []
    settings = config["scoring"]
    for arm, values in storage["scenarios"].items():
        for day_index, day in enumerate(days):
            metrics = scoring.joint_metrics(storage["actual"][day_index], values[day_index],
                                            storage["scales"][day_index], TARGETS,
                                            settings["joint_blocks"], settings["energy_score_beta"],
                                            settings["variogram_power"], settings["variogram_weights"])
            rows.extend({"delivery_day": day, "delivery_day_index": day_index, "arm": arm, **row}
                        for row in metrics)
    return rows


def dependence_rows(storage: dict, config: dict) -> list:
    rows = []
    settings = config["scoring"]
    for arm, values in storage["scenarios"].items():
        results = scoring.dependence_metrics(storage["actual"], values, TARGETS, settings["joint_targets"],
                                             settings["dependence_lags_hours"],
                                             settings["upper_tail_quantile"])
        rows.extend({"arm": arm, **row} for row in results)
    return rows


def _conditional_grid(storage: dict, spec: str, product: str):
    if product == "direct_quantile_regression":
        return storage["direct"][spec]
    if product == "unconditional":
        return storage["quantiles"][spec][:, TARGETS.index("day_ahead_price")]
    return None


def daily_conditional_rows(storage: dict, days: list, config: dict) -> list:
    rows = []
    actual = storage["actual"][:, TARGETS.index("day_ahead_price")]
    levels = config["scoring"]["quantile_levels"]
    interval = float(config["cross_market"]["conditional_interval_level"])
    for arm, values in storage["conditional"].items():
        spec, product = arm.split("__")
        grid = _conditional_grid(storage, spec, product)
        for day_index, day in enumerate(days):
            ensemble = values[day_index].T
            mask = np.isfinite(actual[day_index])
            result = scoring.conditional_metrics(actual[day_index], ensemble,
                                                 np.median(ensemble, axis=1), mask)
            coverage = scoring.central_interval_metrics(actual[day_index], ensemble, [interval], mask)[0]
            row = {"delivery_day": day, "delivery_day_index": day_index, "arm": arm, "spec": spec,
                   "product": product, **result, "interval_level": interval,
                   "coverage": coverage["coverage"], "average_width": coverage["average_width"]}
            if grid is not None:
                row["mean_pinball"] = scoring.pinball_metrics(actual[day_index], grid[day_index],
                                                              levels, mask)["mean_pinball"]
            rows.append(row)
    return rows


def aggregate_conditional_rows(daily: pd.DataFrame, storage: dict, config: dict) -> list:
    rows = []
    saturation = {spec: sum(record[spec]["conditioning_saturated_hours"] for record in storage["diagnostics"])
                  for spec in config["cross_market"]["specs"]}
    hours = len(storage["diagnostics"]) * 24
    for arm, frame in daily.groupby("arm", sort=True):
        spec = arm.split("__")[0]
        row = {"arm": arm, "spec": spec, "product": frame["product"].iloc[0], "n_days": len(frame),
               "sample_crps": frame["sample_crps"].mean(), "median_mae": frame["median_mae"].mean(),
               "interval_level": frame["interval_level"].iloc[0], "coverage": frame["coverage"].mean(),
               "average_width": frame["average_width"].mean(),
               "conditioning_saturated_share": saturation[spec] / hours}
        if "mean_pinball" in frame and frame["mean_pinball"].notna().all():
            row["mean_pinball"] = frame["mean_pinball"].mean()
        rows.append(row)
    return rows


def _pair(frame: pd.DataFrame, arm: str, comparator: str, metric: str, config: dict) -> dict:
    left = frame.loc[frame["arm"].eq(arm)].sort_values("delivery_day")
    right = frame.loc[frame["arm"].eq(comparator)].sort_values("delivery_day")
    assert left["delivery_day"].tolist() == right["delivery_day"].tolist(), "paired days differ"
    settings = config["scoring"]
    result = scoring.significance_metrics(left[metric].to_numpy(), right[metric].to_numpy(),
                                          settings["dm_newey_west_lags"], settings["dm_sidedness"],
                                          settings["bootstrap_block_days"], settings["bootstrap_repetitions"],
                                          settings["bootstrap_confidence_level"], settings["bootstrap_seed"])
    return {"arm": arm, "comparator": comparator, "metric": metric, **result}


def significance_rows(pinball: pd.DataFrame, joint: pd.DataFrame, conditional: pd.DataFrame,
                      config: dict) -> list:
    settings = config["cross_market"]["significance"]
    rows = []
    for target in settings["marginal_pinball_targets"]:
        frame = pinball.loc[pinball["target"].eq(target)]
        rows.extend({"family": "marginal_pinball", "target": target,
                     **_pair(frame, arm, comparator, "mean_pinball", config)}
                    for arm, comparator in settings["marginal_pinball_pairs"])
    for block in settings["joint_blocks"]:
        frame = joint.loc[joint["block"].eq(block)]
        rows.extend({"family": "joint", "block": block, **_pair(frame, arm, comparator, metric, config)}
                    for arm, comparator in settings["joint_pairs"]
                    for metric in ["energy_score", "variogram_score"])
    for arm in sorted(conditional["arm"].unique()):
        spec = arm.split("__")[0]
        comparators = list(settings["conditional_reference_arms"]) + [f"{spec}__unconditional"]
        for comparator in dict.fromkeys(comparators):
            if comparator != arm:
                rows.append({"family": "conditional_1200", "target": "day_ahead_price",
                             **_pair(conditional, arm, comparator, "sample_crps", config)})
    return rows


def reproduction_check(joint: pd.DataFrame, conditional: pd.DataFrame, config: dict,
                       day_indices: list = None) -> dict:
    """Compare the own_hourly arms with the committed S3-A seed-1 results."""
    settings = config["cross_market"]["reproduction"]
    source = ROOT / settings["source_directory"]
    reference = pd.read_csv(source / "daily_scores.csv")
    reference = reference.loc[reference["seed"].eq(int(config["cross_market"]["sampling_seed"]))]
    output = {}
    for arm, s3a_arm in settings["joint_arms"].items():
        ours = joint.loc[joint["arm"].eq(arm)].set_index(["delivery_day_index", "block"])
        theirs = reference.loc[reference["arm"].eq(s3a_arm)].set_index(["delivery_day_index", "block"])
        theirs = theirs.reindex(ours.index)
        assert theirs["energy_score"].notna().all(), f"{s3a_arm}: missing S3-A daily scores"
        output[arm] = {metric: float(np.max(np.abs(ours[metric] - theirs[metric])))
                       for metric in ["energy_score", "variogram_score"]}
    if day_indices is None:
        table = pd.read_csv(source / "conditional_1200.csv").set_index(["arm", "seed"])
        seed = int(config["cross_market"]["sampling_seed"])
        for arm, s3a_arm in settings["conditional_arms"].items():
            ours = conditional.loc[conditional["arm"].eq(arm), "sample_crps"].mean()
            output[arm] = {"sample_crps": float(abs(ours - table.loc[(s3a_arm, seed), "sample_crps"]))}
    maximum = max(value for item in output.values() for value in item.values())
    assert maximum <= float(settings["tolerance"]), f"S3-A reproduction failed: {output}"
    return {"maximum_absolute_difference": maximum, "tolerance": settings["tolerance"], "arms": output}


def _quantreg_summary(storage: dict, config: dict) -> dict:
    output = {}
    for spec in config["cross_market"]["specs"]:
        records = [record[spec] for record in storage["diagnostics"]]
        units = {f"{target}_0730": [record["quantreg"][target] for record in records]
                 for target in features.MODIFIED_TARGETS}
        units["day_ahead_price_1200_direct"] = [record["direct_quantreg"] for record in records]
        output[spec] = {name: {"fit_count": sum(item["fit_count"] for item in items),
                               "maximum_iterations": max(item["maximum_iterations"] for item in items),
                               "support_clipped_entries": sum(item["support_clipped_entries"] for item in items),
                               "endpoint_replacement_cells":
                                   sum(item["endpoint_replacement_cells"] for item in items),
                               "dropped_predictor_counts": pd.Series(
                                   [name for item in items for name in item["dropped_predictors"]],
                                   dtype=object).value_counts().sort_index().astype(int).to_dict()}
                        for name, items in units.items()}
        output[spec]["rank_selection_counts"] = {
            str(key): int(value) for key, value in
            pd.Series([record["rank"] for record in records]).value_counts().sort_index().items()}
    return output


def _file_hashes(output: Path) -> dict:
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "summary.json"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def write_results(storage: dict, days: list, config: dict) -> None:
    output = ROOT / config["output"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pinball = pd.DataFrame(daily_pinball_rows(storage, days, config))
    joint = pd.DataFrame(daily_joint_rows(storage, days, config))
    conditional = pd.DataFrame(daily_conditional_rows(storage, days, config))
    aggregate_joint = joint.groupby(["arm", "block"], sort=True)[["energy_score", "variogram_score"]].mean()
    tables = {"marginal_metrics_csv": pd.DataFrame(marginal_rows(storage, config)),
              "monthly_calibration_csv": pd.DataFrame(monthly_rows(storage, days, config)),
              "joint_metrics_csv": aggregate_joint.reset_index().assign(n_days=len(days)),
              "dependence_errors_csv": pd.DataFrame(dependence_rows(storage, config)),
              "conditional_1200_csv": pd.DataFrame(aggregate_conditional_rows(conditional, storage, config)),
              "significance_csv": pd.DataFrame(significance_rows(pinball, joint, conditional, config)),
              "daily_marginal_pinball_csv": pinball, "daily_joint_scores_csv": joint,
              "daily_conditional_scores_csv": conditional}
    for key, frame in tables.items():
        frame.to_csv(output / config["output"][key], index=False)
    reproduction = reproduction_check(joint, conditional, config)
    summary = {"environment": environment_fingerprint(), "sampling_seed": config["cross_market"]["sampling_seed"],
               "evaluation_label": config["cross_market"]["evaluation_label"],
               "evaluation_note": config["cross_market"]["evaluation_note"],
               "base_config": config["base_config_path"],
               "base_config_sha256": hashlib.sha256((ROOT / config["base_config_path"]).read_bytes()).hexdigest(),
               "window": config["window"], "test_day_count": len(days),
               "test_day_sha256": hashlib.sha256("\n".join(days).encode()).hexdigest(),
               "specs": config["cross_market"]["specs"],
               "cross_features": config["cross_market"]["cross_features"],
               "conditional_1200": config["cross_market"]["conditional_1200"],
               "diagnostics": _quantreg_summary(storage, config),
               "s3a_reproduction": reproduction,
               "acceptance": {"test_days_match_s2": len(days) == config["split"]["expected_test_days"],
                              "missing_preserved": True, "visibility_asserted": True,
                              "s3a_reproduction_within_tolerance": True},
               "file_sha256": _file_hashes(output)}
    (output / config["output"]["summary_json"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _feature_hand_checks(design: pd.DataFrame, conditional: pd.DataFrame, hourly: pd.DataFrame,
                         day: str, config: dict) -> dict:
    previous = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lookup = hourly.set_index(["delivery_day", "target", "hour"])["value"]
    hour = 17
    capacity = design.loc[design["delivery_day"].eq(day) & design["target"].eq("capacity_price")
                          & design["hour"].eq(hour)].iloc[0]
    prices = np.sort([lookup[(previous, "day_ahead_price", h)] for h in range(24)])
    spread = prices[-2:].mean() - prices[:2].mean()
    assert capacity["cross_day_ahead_previous_day_same_hour"] == lookup[(previous, "day_ahead_price", hour)]
    assert np.isclose(capacity["cross_day_ahead_previous_day_two_hour_spread"], spread, rtol=0.0, atol=1e-9)
    day_ahead = design.loc[design["delivery_day"].eq(day) & design["target"].eq("day_ahead_price")
                           & design["hour"].eq(hour)].iloc[0]
    capacity_previous = [lookup[(previous, "capacity_price", h)] for h in range(24)]
    assert day_ahead["cross_capacity_previous_day_same_hour"] == capacity_previous[hour]
    assert np.isclose(day_ahead["cross_capacity_previous_day_mean"], np.mean(capacity_previous), atol=1e-9)
    late = conditional.loc[conditional["delivery_day"].eq(day) & conditional["target"].eq("day_ahead_price")
                           & conditional["hour"].eq(hour)].iloc[0]
    assert late["realized_capacity_same_hour"] == lookup[(day, "capacity_price", hour)]
    assert "realized_capacity_same_hour" not in design.columns, "07:30 design carries a 12:00 predictor"
    return {"hour": hour, "day_ahead_two_hour_spread": float(spread)}


def _injection_checks(design: pd.DataFrame, conditional: pd.DataFrame, day: str) -> None:
    check = TestCase()
    rows = design.loc[design["delivery_day"].eq(day) & design["target"].eq("capacity_price")].iloc[[0]].copy()
    late = rows.copy()
    late["cross_day_ahead_previous_day_same_hour_available_at"] = late["decision_time_utc"] + pd.Timedelta(minutes=1)
    check.assertRaises(AssertionError, panel.assert_input_visibility, late)
    same_day = rows.copy()
    same_day["cross_day_ahead_previous_day_same_hour_source_day"] = same_day["delivery_day"]
    check.assertRaises(AssertionError, panel.assert_input_visibility, same_day)
    early = conditional.loc[conditional["delivery_day"].eq(day)
                            & conditional["target"].eq("day_ahead_price")].iloc[[0]].copy()
    early["decision_time_utc"] = design.loc[early.index, "decision_time_utc"]
    check.assertRaises(AssertionError, panel.assert_input_visibility, early)


def _linear_conditional_hand_check(config: dict) -> dict:
    sigma = np.asarray([[1.0, 0.3, 0.2, 0.1], [0.3, 1.0, 0.5, 0.4], [0.2, 0.5, 1.0, 0.6],
                        [0.1, 0.4, 0.6, 1.0]])
    conditioned, weights, value = np.asarray([2, 3]), np.asarray([0.5, 0.5]), 0.8
    active, uniforms = models.gaussian_linear_conditional(sigma, conditioned, weights, value, 80000, 7, config)
    samples = scipy.stats.norm.ppf(uniforms)
    cross = sigma[np.ix_(active, conditioned)] @ weights
    variance = weights @ sigma[np.ix_(conditioned, conditioned)] @ weights
    mean_error = float(np.max(np.abs(samples.mean(axis=0) - cross * value / variance)))
    expected = sigma[np.ix_(active, active)] - np.outer(cross, cross) / variance
    covariance_error = float(np.max(np.abs(np.cov(samples, rowvar=False) - expected)))
    assert max(mean_error, covariance_error) <= config["scoring"]["synthetic_check_tolerance"]
    return {"mean_error": mean_error, "covariance_error": covariance_error}


def _reference_origin_checks(design: pd.DataFrame, conditional: pd.DataFrame, day: str,
                             config: dict) -> dict:
    """Refit own_hourly through the study path, then reproduce S3-A day-0 joint scores."""
    seed = int(config["scoring"]["randomized_pit_seed"])
    base = models.marginals.marginal_day_bundle(design, day, config, seed)
    refit, _ = models.spec_bundle(base, design, config["cross_market"]["reference_spec"], config, seed, day)
    assert np.array_equal(refit["raw"], base["raw"], equal_nan=True), "own_hourly raw refit differs"
    assert np.array_equal(refit["calibrated"], base["calibrated"], equal_nan=True)
    days =panel.test_delivery_days(config, ROOT)
    index = days.index(day)
    prepared = models.prepare_origin(design, conditional, day, config)
    scenarios, _ = models.sample_origin(prepared, int(config["cross_market"]["sampling_seed"]) + index, config)
    storage = {"actual": actual_tensor(prepared["base"]["frame"], [day]),
               "scales": _target_scales(prepared["fit_actual"])[None, :],
               "scenarios": {arm: values[None] for arm, values in scenarios.items()
                             if arm in config["cross_market"]["reproduction"]["joint_arms"]}}
    joint = pd.DataFrame(daily_joint_rows(storage, [day], config))
    joint["delivery_day_index"] = index
    reproduction = reproduction_check(joint, None, config, [index])
    return {"reproduction": reproduction,
            "ranks": {spec: item["diagnostics"]["rank"] for spec, item in prepared["specs"].items()},
            "saturated_hours": {spec: item["diagnostics"]["conditioning_saturated_hours"]
                                for spec, item in prepared["specs"].items()}}


def _pooled_solver_checks(design: pd.DataFrame, day: str, config: dict) -> dict:
    """HiGHS must hit the analytic median and never lose to statsmodels on the pinball objective."""
    parameters, _, _ = models.linear_program_quantiles(np.ones((5, 1)), np.arange(1.0, 6.0),
                                                       np.asarray([0.5]), config)
    assert abs(parameters[0, 0] - 3.0) <= 1e-9, "HiGHS median of 1..5 is not 3"
    variant = features.spec_config(config, "own_pooled")
    blocks = panel.window_days(day, config)
    fit = design.loc[design["delivery_day"].isin(blocks["fit"]) & design["target"].eq("capacity_price")]
    columns = models.marginals.feature_columns("capacity_price", variant)
    usable = models.marginals._complete_rows(fit, columns)
    scaling = models.marginals._fit_scaling(usable, columns, variant, usable)
    x = models.marginals._scaled_matrix(usable, columns, scaling)
    y = usable["value"].to_numpy(dtype=float)
    levels = np.asarray(config["scoring"]["quantile_levels"], dtype=float)
    exact, _, spec = models.linear_program_quantiles(x, y, levels, variant)
    settings = variant["model"]["quantreg"]
    irls, _, _ = models.marginals._fit_quantile_parameters(x, y, levels, settings, "pooled solver check")
    gaps = []
    for index, level in enumerate(levels[spec["mask"]]):
        position = np.flatnonzero(spec["mask"])[index]
        losses = [np.mean(np.maximum(level * (y - x @ beta), (level - 1.0) * (y - x @ beta)))
                  for beta in [exact[position], irls[position]]]
        gaps.append(losses[0] - losses[1])
    tolerance = float(config["cross_market"]["pooled_solver"]["agreement_check_tolerance"])
    assert max(gaps) <= tolerance, "HiGHS pinball objective exceeds statsmodels"
    regression = config["cross_market"]["pooled_solver"]
    regression_day = regression["regression_delivery_day"]
    regression_blocks = panel.window_days(regression_day, config)
    regression_frame = models._relevant_frame(design, regression_blocks, regression_day)
    regression_variant = features.spec_config(config, "own_pooled")
    regression_fit = regression_frame.loc[
        regression_frame["delivery_day"].isin(regression_blocks["fit"])
        & regression_frame["target"].eq("capacity_price")]
    regression_columns = models.marginals.feature_columns("capacity_price", regression_variant)
    regression_usable = models.marginals._complete_rows(regression_fit, regression_columns)
    regression_scaling = models.marginals._fit_scaling(
        regression_usable, regression_columns, regression_variant, regression_usable)
    regression_x = models.marginals._scaled_matrix(
        regression_usable, regression_columns, regression_scaling)
    models.linear_program_quantiles(regression_x, regression_usable["value"].to_numpy(dtype=float),
                                    np.asarray(regression["regression_levels"]), config)
    return {"exact_minus_irls_pinball_max": float(max(gaps)),
            "exact_minus_irls_pinball_min": float(min(gaps)), "rows": int(y.size),
            "regression_delivery_day": regression_day,
            "regression_levels": regression["regression_levels"]}


def _function_length_check() -> None:
    paths = [ROOT / "src/s3_cross_market" / name for name in ["features.py", "models.py"]]
    for path in paths + [Path(__file__)]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lengths = [node.end_lineno - node.lineno + 1 for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert max(lengths) <= 60, f"{path}: function exceeds 60 lines"


def run_checks(config: dict) -> None:
    _function_length_check()
    hourly_panel, design, conditional = build_designs(config)
    day = config["split"]["reference_delivery_day"]
    _injection_checks(design, conditional, day)
    results = {"features": _feature_hand_checks(design, conditional, hourly_panel, day, config),
               "linear_conditional": _linear_conditional_hand_check(config),
               "pooled_solver": _pooled_solver_checks(design, day, config),
               "reference_origin": _reference_origin_checks(design, conditional, day, config)}
    print(json.dumps(results, indent=2, sort_keys=True))


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if arguments.check:
        run_checks(config)
        return
    _, design, conditional = build_designs(config)
    days = panel.test_delivery_days(config, ROOT)
    storage = run_origins(config, design, conditional, days)
    write_results(storage, days, config)


if __name__ == "__main__":
    main()
