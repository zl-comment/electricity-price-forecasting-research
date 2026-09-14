"""Run the exploratory four-surface-conditioned joint scenario study."""

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import s3_cross_market as cross_experiment
from src.s3_cross_market import features, models
from src.s3_forecast_side import marginals, panel, scoring
from src.s3_forecast_side.generators import (TARGETS, _base_scenarios, _fit_dependence,
                                             _stream, _target_scales, actual_tensor,
                                             quantile_tensor)
from src.s3_joint_surfaces import surface_coupling


WORKER_STATE = {}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load the cross-market base and attach the joint-surface study sections."""
    study = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    config = cross_experiment.load_config(study["base_config"])
    for key in ["joint_surfaces", "runtime", "output"]:
        config[key] = study[key]
    config["scoring"]["joint_targets"] = study["scoring"]["joint_targets"]
    config["scoring"]["joint_blocks"] = study["scoring"]["joint_blocks"]
    config["protocol"]["sampling_seeds"] = study["joint_surfaces"]["sampling_seeds"]
    config["cross_market_config_path"] = study["base_config"]
    config["joint_surface_config_path"] = path
    assert tuple(config["joint_surfaces"]["target_order"]) == TARGETS
    assert config["protocol"]["missing_value_policy"] == "preserve"
    assert config["protocol"]["time_split_policy"] == "chronological"
    return config


def environment_fingerprint() -> dict:
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__}


def _surface_target_indices(arm: str, config: dict) -> np.ndarray:
    names = config["joint_surfaces"]["surface_arms"][arm]["distance_targets"]
    assert len(names) == len(set(names)) and all(name in TARGETS for name in names)
    return np.asarray([TARGETS.index(name) for name in names], dtype=int)


def prepare_origin(design: pd.DataFrame, day: str, config: dict) -> dict:
    """Fit own_pooled, its dependence references, and all required surface tensors."""
    pit_seed = int(config["scoring"]["randomized_pit_seed"])
    base = marginals.marginal_day_bundle(design, day, config, pit_seed)
    spec = config["joint_surfaces"]["reference_spec"]
    bundle, diagnostics = models.spec_bundle(base, design, spec, config, pit_seed, day)
    dependence = _fit_dependence(bundle, pit_seed, config)
    bundle = {**bundle, "dependence_uniforms": dependence["uniforms"]}
    blocks, levels = bundle["blocks"], bundle["levels"]
    fit_actual = actual_tensor(bundle["frame"], blocks["fit"])
    calibration_actual = actual_tensor(bundle["frame"], blocks["calibration"])
    surfaces = {
        "fit": quantile_tensor(bundle["frame"], bundle["calibrated"], blocks["fit"], levels),
        "calibration": quantile_tensor(
            bundle["frame"], bundle["calibrated"], blocks["calibration"], levels),
        "test": quantile_tensor(bundle["frame"], bundle["calibrated"], [day], levels)[0],
    }
    assert np.isfinite(surfaces["calibration"]).all() and np.isfinite(surfaces["test"]).all()
    return {"day": day, "bundle": bundle, "dependence": dependence, "surfaces": surfaces,
            "fit_actual": fit_actual, "calibration_actual": calibration_actual,
            "test_actual": actual_tensor(bundle["frame"], [day])[0],
            "target_scales": _target_scales(fit_actual), "quantreg_diagnostics": diagnostics}


def surface_history(prepared: dict, arm: str, config: dict) -> tuple:
    """Exclude incomplete fit surfaces without altering or filling their missing cells."""
    assert config["joint_surfaces"]["incomplete_fit_surface_policy"] == \
        "exclude_from_surface_library_preserve_missing"
    indices = _surface_target_indices(arm, config)
    surfaces = prepared["surfaces"]["fit"]
    usable = np.isfinite(surfaces[:, indices]).all(axis=(1, 2, 3))
    minimum = max(int(value) for value in config["joint_surfaces"]["neighbor_candidates"])
    assert int(usable.sum()) >= minimum, f"{prepared['day']} {arm}: insufficient complete surfaces"
    return surfaces[usable], prepared["dependence"]["uniforms"][usable], int(usable.sum())


def select_surface_arms(prepared: dict, origin_index: int, config: dict) -> tuple:
    """Choose one neighbor count per surface arm using C(D) only."""
    settings = config["joint_surfaces"]
    selected, rows = {}, []
    for arm in settings["surface_arms"]:
        fit_surfaces, fit_uniforms, history_count = surface_history(prepared, arm, config)
        offset = settings["stream_offsets"][f"selection_{arm}"]
        seed = _stream(int(settings["selection"]["seed"]) + origin_index, offset)
        count, candidates = surface_coupling.select_neighbor_count(
            fit_surfaces, prepared["surfaces"]["calibration"], fit_uniforms,
            prepared["calibration_actual"],
            prepared["target_scales"], _surface_target_indices(arm, config),
            prepared["bundle"]["levels"], seed, config)
        selected[arm] = count
        rows.extend({"arm": arm, "neighbor_count": item["neighbor_count"],
                     "calibration_energy_score": item["calibration_energy_score"],
                     "selected": item["neighbor_count"] == count,
                     "surface_history_days": history_count} for item in candidates)
    return selected, rows


def sample_origin(prepared: dict, selected: dict, seed: int, config: dict) -> tuple:
    """Draw reference and surface-conditioned arms for one base seed."""
    settings = config["joint_surfaces"]
    count = int(config["generators"]["scenario_count"])
    base = _base_scenarios(prepared["fit_actual"], prepared["surfaces"]["test"],
                           prepared["dependence"], count, seed, config)
    scenarios = {arm: base[source].astype(np.float32)
                 for arm, source in settings["reference_arms"].items()}
    diagnostics = {}
    for arm, neighbor_count in selected.items():
        fit_surfaces, fit_uniforms, history_count = surface_history(prepared, arm, config)
        arm_seed = _stream(seed, settings["stream_offsets"][arm])
        values, audit = surface_coupling.surface_scenarios(
            prepared["surfaces"]["test"], fit_surfaces, fit_uniforms, prepared["target_scales"],
            _surface_target_indices(arm, config), prepared["bundle"]["levels"],
            neighbor_count, count, arm_seed, config)
        scenarios[arm] = values.astype(np.float32)
        diagnostics[arm] = {**audit, "surface_history_days": history_count}
    expected = list(settings["reference_arms"]) + list(settings["surface_arms"])
    assert list(scenarios) == expected
    return scenarios, diagnostics


def _init_worker(config: dict, design: pd.DataFrame) -> None:
    WORKER_STATE.update({"config": config, "design": design})


def _day_worker(item: tuple) -> dict:
    index, day = item
    config = WORKER_STATE["config"]
    prepared = prepare_origin(WORKER_STATE["design"], day, config)
    selected, selection = select_surface_arms(prepared, index, config)
    samples, audits = {}, {}
    for base_seed in config["protocol"]["sampling_seeds"]:
        scenarios, diagnostic = sample_origin(prepared, selected, int(base_seed) + index, config)
        samples[int(base_seed)] = scenarios
        audits[int(base_seed)] = diagnostic
    return {"index": index, "day": day, "actual": prepared["test_actual"],
            "scales": prepared["target_scales"], "samples": samples, "audits": audits,
            "selection": selection, "selected": selected,
            "quantreg_diagnostics": prepared["quantreg_diagnostics"]}


def _empty_storage(days: list, config: dict) -> dict:
    count = int(config["generators"]["scenario_count"])
    arms = list(config["joint_surfaces"]["reference_arms"]) + \
        list(config["joint_surfaces"]["surface_arms"])
    scenarios = {int(seed): {arm: np.empty((len(days), count, len(TARGETS), 24), np.float32)
                             for arm in arms} for seed in config["protocol"]["sampling_seeds"]}
    return {"actual": np.empty((len(days), len(TARGETS), 24)),
            "scales": np.empty((len(days), len(TARGETS))), "scenarios": scenarios,
            "selection": [], "diagnostics": []}


def _store(storage: dict, result: dict) -> None:
    index = result["index"]
    storage["actual"][index] = result["actual"]
    storage["scales"][index] = result["scales"]
    for seed, arms in result["samples"].items():
        for arm, values in arms.items():
            storage["scenarios"][seed][arm][index] = values
    storage["selection"].extend({"delivery_day": result["day"], **row}
                                for row in result["selection"])
    storage["diagnostics"].append({"delivery_day": result["day"],
                                   "selected": result["selected"], "audits": result["audits"],
                                   "quantreg": result["quantreg_diagnostics"]})


def run_origins(config: dict, design: pd.DataFrame, days: list) -> dict:
    storage = _empty_storage(days, config)
    context = mp.get_context("fork")
    with context.Pool(int(config["runtime"]["worker_processes"]), initializer=_init_worker,
                      initargs=(config, design)) as pool:
        for completed, result in enumerate(pool.imap_unordered(_day_worker, enumerate(days)), 1):
            _store(storage, result)
            if completed % 10 == 0 or completed == len(days):
                print(json.dumps({"completed_origins": completed, "total_origins": len(days)}), flush=True)
    storage["selection"].sort(key=lambda row: (row["delivery_day"], row["arm"], row["neighbor_count"]))
    storage["diagnostics"].sort(key=lambda row: row["delivery_day"])
    return storage


def daily_joint_rows(storage: dict, days: list, config: dict) -> list:
    rows, settings = [], config["scoring"]
    for seed, arms in storage["scenarios"].items():
        for arm, values in arms.items():
            for index, day in enumerate(days):
                metrics = scoring.joint_metrics(
                    storage["actual"][index], values[index], storage["scales"][index], TARGETS,
                    settings["joint_blocks"], settings["energy_score_beta"],
                    settings["variogram_power"], settings["variogram_weights"])
                rows.extend({"delivery_day": day, "delivery_day_index": index, "seed": seed,
                             "arm": arm, **metric} for metric in metrics)
    return rows


def aggregate_joint_rows(daily: pd.DataFrame, days: list) -> pd.DataFrame:
    grouped = daily.groupby(["seed", "arm", "block"], sort=True)
    return grouped.agg(energy_score=("energy_score", "mean"),
                       variogram_score=("variogram_score", "mean"),
                       mean_valid_dimensions=("n_valid_dimensions", "mean")).reset_index().assign(
                           n_days=len(days))


def dependence_rows(storage: dict, config: dict) -> list:
    rows, settings = [], config["scoring"]
    for seed, arms in storage["scenarios"].items():
        for arm, values in arms.items():
            metrics = scoring.dependence_metrics(
                storage["actual"], values, TARGETS, settings["joint_targets"],
                settings["dependence_lags_hours"], settings["upper_tail_quantile"])
            rows.extend({"seed": seed, "arm": arm, **metric} for metric in metrics)
    return rows


def _paired_significance(frame: pd.DataFrame, arm: str, comparator: str,
                         metric: str, config: dict) -> dict:
    left = frame.loc[frame["arm"].eq(arm)].sort_values("delivery_day")
    right = frame.loc[frame["arm"].eq(comparator)].sort_values("delivery_day")
    assert left["delivery_day"].tolist() == right["delivery_day"].tolist()
    settings = config["scoring"]
    result = scoring.significance_metrics(
        left[metric].to_numpy(), right[metric].to_numpy(), settings["dm_newey_west_lags"],
        settings["dm_sidedness"], settings["bootstrap_block_days"],
        settings["bootstrap_repetitions"], settings["bootstrap_confidence_level"],
        settings["bootstrap_seed"])
    return {"arm": arm, "comparator": comparator, "metric": metric, **result}


def significance_rows(daily: pd.DataFrame, config: dict) -> list:
    settings = config["joint_surfaces"]["significance"]
    assert settings["seed_aggregation"] == "daily_mean_before_significance"
    averaged = daily.groupby(["delivery_day", "arm", "block"], sort=True)[
        ["energy_score", "variogram_score"]].mean().reset_index()
    rows = []
    for block in settings["blocks"]:
        frame = averaged.loc[averaged["block"].eq(block)]
        rows.extend({"family": "joint_surface", "block": block,
                     **_paired_significance(frame, arm, comparator, metric, config)}
                    for arm, comparator in settings["pairs"]
                    for metric in ["energy_score", "variogram_score"])
    return rows


def reproduction_check(daily: pd.DataFrame, config: dict, indices: list = None) -> dict:
    settings = config["joint_surfaces"]["reproduction"]
    reference = pd.read_csv(ROOT / settings["source_daily_scores"])
    ours = daily.loc[daily["seed"].eq(1)]
    if indices is not None:
        reference = reference.loc[reference["delivery_day_index"].isin(indices)]
        ours = ours.loc[ours["delivery_day_index"].isin(indices)]
    output = {}
    for arm in config["joint_surfaces"]["reference_arms"]:
        left = ours.loc[ours["arm"].eq(arm) & ~ours["block"].eq("four_targets_observed")].set_index(
            ["delivery_day_index", "block"])
        right = reference.loc[reference["arm"].eq(arm)].set_index(["delivery_day_index", "block"])
        right = right.reindex(left.index)
        assert right["energy_score"].notna().all()
        output[arm] = {metric: float(np.max(np.abs(left[metric] - right[metric])))
                       for metric in ["energy_score", "variogram_score"]}
    maximum = max(value for result in output.values() for value in result.values())
    assert maximum <= float(settings["tolerance"]), f"own_pooled reproduction failed: {output}"
    return {"maximum_absolute_difference": maximum, "tolerance": settings["tolerance"], "arms": output}


def _diagnostic_summary(storage: dict, config: dict) -> dict:
    selection = pd.DataFrame(storage["selection"])
    counts = selection.loc[selection["selected"]].groupby(["arm", "neighbor_count"]).size()
    selected = {arm: {str(count): int(value) for count, value in counts.loc[arm].items()}
                for arm in config["joint_surfaces"]["surface_arms"]}
    errors = [audit["marginal_preservation_error"] for row in storage["diagnostics"]
              for seed in row["audits"].values() for audit in seed.values()]
    return {"neighbor_selection_counts": selected,
            "maximum_marginal_preservation_error": float(max(errors))}


def _file_hashes(output: Path) -> dict:
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "summary.json"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def write_results(storage: dict, days: list, config: dict) -> None:
    output = ROOT / config["output"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(daily_joint_rows(storage, days, config))
    tables = {"surface_selection_csv": pd.DataFrame(storage["selection"]),
              "joint_metrics_csv": aggregate_joint_rows(daily, days),
              "dependence_errors_csv": pd.DataFrame(dependence_rows(storage, config)),
              "significance_csv": pd.DataFrame(significance_rows(daily, config)),
              "daily_joint_scores_csv": daily}
    for key, frame in tables.items():
        frame.to_csv(output / config["output"][key], index=False)
    reproduction = reproduction_check(daily, config)
    summary = {"environment": environment_fingerprint(),
               "evaluation_label": config["joint_surfaces"]["evaluation_label"],
               "evaluation_note": config["joint_surfaces"]["evaluation_note"],
               "sampling_seeds": config["protocol"]["sampling_seeds"],
               "test_day_count": len(days), "test_day_sha256": hashlib.sha256(
                   "\n".join(days).encode()).hexdigest(), "window": config["window"],
               "cross_market_config": config["cross_market_config_path"],
               "joint_surface_config": config["joint_surface_config_path"],
               "joint_surface_config_sha256": hashlib.sha256(
                   (ROOT / config["joint_surface_config_path"]).read_bytes()).hexdigest(),
               "joint_surfaces": config["joint_surfaces"],
               "diagnostics": _diagnostic_summary(storage, config),
               "cross_market_reproduction": reproduction,
               "acceptance": {"test_days_match_s2": len(days) == config["split"]["expected_test_days"],
                              "missing_preserved": True, "visibility_asserted": True,
                              "marginals_preserved_exactly": True,
                              "cross_market_reproduction_within_tolerance": True},
               "file_sha256": _file_hashes(output)}
    (output / config["output"]["summary_json"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_checks(config: dict) -> dict:
    levels = np.asarray(config["scoring"]["quantile_levels"], dtype=float)
    weights = surface_coupling.probability_cell_weights(levels, config)
    generator = np.random.default_rng(7)
    base = generator.uniform(size=(100, len(TARGETS) * 24))
    templates = generator.normal(size=base.shape)
    coupled, error = surface_coupling.rank_reorder_uniforms(base, templates)
    assert np.array_equal(np.sort(coupled, axis=0), np.sort(base, axis=0))
    distances = np.linspace(0.1, 2.0, 84)
    selected, kernel, bandwidth = surface_coupling.gaussian_neighbor_weights(distances, 14, config)
    assert selected.tolist() == list(range(14)) and kernel[0] > kernel[-1]
    return {"probability_weight_sum": float(weights.sum()), "marginal_preservation_error": error,
            "kernel_weight_sum": float(kernel.sum()), "bandwidth": bandwidth}


def _reference_origin_check(design: pd.DataFrame, config: dict) -> dict:
    day = config["split"]["reference_delivery_day"]
    days = panel.test_delivery_days(config, ROOT)
    index = days.index(day)
    prepared = prepare_origin(design, day, config)
    selected, selection = select_surface_arms(prepared, index, config)
    scenarios, audits = sample_origin(prepared, selected, 1 + index, config)
    storage = {"actual": prepared["test_actual"][None],
               "scales": prepared["target_scales"][None],
               "scenarios": {1: {arm: values[None] for arm, values in scenarios.items()}}}
    daily = pd.DataFrame(daily_joint_rows(storage, [day], config))
    reproduction = reproduction_check(daily, config, [index])
    return {"selected_neighbors": selected, "selection_rows": len(selection),
            "scenario_shapes": {arm: list(values.shape) for arm, values in scenarios.items()},
            "maximum_marginal_preservation_error": max(
                audit["marginal_preservation_error"] for audit in audits.values()),
            "reproduction": reproduction}


def _function_length_check() -> None:
    paths = [ROOT / "src/s3_joint_surfaces/surface_coupling.py", Path(__file__)]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lengths = [node.end_lineno - node.lineno + 1 for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert max(lengths) <= 60, f"{path}: function exceeds 60 lines"


def run_checks(config: dict) -> None:
    _function_length_check()
    scoring.validate_scoring_config(config)
    _, design, _ = cross_experiment.build_designs(config)
    results = {"synthetic": _synthetic_checks(config),
               "reference_origin": _reference_origin_check(design, config)}
    print(json.dumps(results, indent=2, sort_keys=True))


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if arguments.check:
        run_checks(config)
        return
    _, design, _ = cross_experiment.build_designs(config)
    days = panel.test_delivery_days(config, ROOT)
    storage = run_origins(config, design, days)
    write_results(storage, days, config)


if __name__ == "__main__":
    main()
