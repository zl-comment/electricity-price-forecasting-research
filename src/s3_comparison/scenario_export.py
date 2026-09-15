"""Export the frozen own-pooled S3-B scenarios without changing upstream results."""

import copy
import hashlib
import json
import multiprocessing as mp
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml

from src.s3_cross_market import features, models
from src.s3_forecast_side import copulas, marginals, panel, scoring
from src.s3_forecast_side.generators import (TARGETS, _conditional_products, _target_scales,
                                             _stream, actual_tensor)


ROOT = Path(__file__).resolve().parents[2]
WORKER_STATE = {}


def load_config(path: str) -> dict:
    study = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    cross_path = ROOT / study["cross_market_config"]
    cross = yaml.safe_load(cross_path.read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / study["base_config"]).read_text(encoding="utf-8"))
    assert cross["base_config"] == study["base_config"]
    config["cross_market"] = cross["cross_market"]
    config["runtime"] = study["runtime"]
    for key in ["storage_config", "storage", "solver", "risk", "tree", "export", "arms",
                "published_rules", "output", "evaluation"]:
        config[key] = study[key]
    config["comparison_config_path"] = path
    return config


def _capacity_enriched(design: pd.DataFrame, hourly: pd.DataFrame, config: dict) -> pd.DataFrame:
    capacity = hourly.loc[hourly["target"].eq("capacity_price"),
                          ["delivery_day", "hour", "value"]].copy()
    capacity = capacity.rename(columns={"value": "realized_capacity_same_hour"})
    capacity["realized_capacity_daily_mean"] = capacity.groupby("delivery_day")[
        "realized_capacity_same_hour"].transform("mean")
    enriched = design.merge(capacity, on=["delivery_day", "hour"], how="left",
                             validate="many_to_one")
    columns = config["model"]["fb0_capacity_features"]
    assert enriched.loc[enriched["target"].eq("day_ahead_price"), columns].notna().all().all()
    return enriched


def build_inputs(config: dict) -> tuple:
    hourly = panel.build_hourly_panel(config, ROOT)
    design = panel.build_all_features(hourly, "gate_0730", config)
    cross = features.build_cross_design(design, hourly, config)
    conditional = features.build_conditional_design(cross, hourly, config)
    enriched = _capacity_enriched(design, hourly, config)
    days = panel.test_delivery_days(config, ROOT)
    assert len(days) == config["split"]["expected_test_days"]
    return hourly, cross, conditional, enriched, days


def _init_worker(config: dict, design: pd.DataFrame, conditional: pd.DataFrame,
                 enriched: pd.DataFrame) -> None:
    WORKER_STATE.update({"config": config, "design": design, "conditional": conditional,
                         "enriched": enriched})


def _point_1200(prepared: dict, day: str) -> np.ndarray:
    config = WORKER_STATE["config"]
    columns = config["model"]["fb0_capacity_features"]
    model = marginals.fit_fb0(WORKER_STATE["enriched"], prepared["base"]["blocks"]["fit"],
                              config, ROOT, "day_ahead_price", columns)
    rows = WORKER_STATE["enriched"].loc[
        WORKER_STATE["enriched"]["delivery_day"].eq(day)
        & WORKER_STATE["enriched"]["target"].eq("day_ahead_price")].sort_values("hour")
    return marginals.predict_fb0(model, rows, config)


def _shared_uniform_products(selected: dict, scenarios: dict, seed: int, config: dict) -> dict:
    count = int(config["export"]["generated_scenario_count"])
    levels = np.asarray(config["scoring"]["quantile_levels"], dtype=float)
    uniforms = {
        "gaussian": copulas.gaussian_uniform_scenarios(
            selected["dependence"]["chosen"]["sigma"], count, _stream(seed, 4), config),
        "empirical": copulas.empirical_uniform_scenarios(
            selected["dependence"]["uniforms"], count, _stream(seed, 3), config)}
    grid = selected["test_quantiles"].reshape(len(TARGETS) * 24, levels.size)
    for name, values in uniforms.items():
        rebuilt = marginals.evaluate_quantile_grid(grid, levels, values.T).T.reshape(count, 4, 24)
        source = scenarios[f"own_pooled__{name}"]
        assert np.array_equal(rebuilt.astype(np.float32), source, equal_nan=True), \
            f"{name}: regenerated uniforms do not reproduce 07:30 scenarios"
    return {name: marginals.evaluate_quantile_grid(
        selected["direct_grid"], levels, values[:, :24].T).T
            for name, values in uniforms.items()}


def _day_worker(item: tuple) -> dict:
    index, day = item
    config = WORKER_STATE["config"]
    prepared = models.prepare_origin(WORKER_STATE["design"], WORKER_STATE["conditional"], day, config)
    seed = int(config["export"]["sampling_seed"]) + index
    scenarios, conditional = models.sample_origin(prepared, seed, config)
    selected = prepared["specs"]["own_pooled"]
    full = _conditional_products(selected["bundle"], selected["dependence"],
                                 selected["test_quantiles"], prepared["fit_actual"], day,
                                 seed, config)
    direct_uniform = _shared_uniform_products(selected, scenarios, seed, config)
    return {"index": index, "day": day, "actual": actual_tensor(prepared["base"]["frame"], [day])[0],
            "scales": _target_scales(prepared["fit_actual"]),
            "quantiles": selected["test_quantiles"], "direct_grid": selected["direct_grid"],
            "scenarios": scenarios, "conditional": conditional,
            "conditional_full": {key: value.astype(np.float32) for key, value in full.items()},
            "direct_uniform": direct_uniform,
            "point_1200": _point_1200(prepared, day)}


def _empty_storage(days: list, config: dict) -> dict:
    count = int(config["export"]["generated_scenario_count"])
    levels = len(config["scoring"]["quantile_levels"])
    return {"actual": np.empty((len(days), 4, 24)), "scales": np.empty((len(days), 4)),
            "quantiles": np.empty((len(days), 4, 24, levels)),
            "direct_grid": np.empty((len(days), 24, levels)),
            "point_1200": np.empty((len(days), 24)), "scenarios": {}, "conditional": {},
            "conditional_full": {}, "direct_uniform": {}}


def _store(storage: dict, result: dict, days: list) -> None:
    index = result["index"]
    assert result["day"] == days[index]
    for key in ["actual", "scales", "quantiles", "direct_grid", "point_1200"]:
        storage[key][index] = result[key]
    for group in ["scenarios", "conditional", "conditional_full", "direct_uniform"]:
        for name, values in result[group].items():
            shape = (len(days),) + values.shape
            storage[group].setdefault(name, np.empty(shape, dtype=values.dtype))[index] = values


def generate(config: dict, design: pd.DataFrame, conditional: pd.DataFrame,
             enriched: pd.DataFrame, days: list) -> dict:
    restricted = copy.deepcopy(config)
    restricted["cross_market"]["specs"] = {
        "own_pooled": config["cross_market"]["specs"]["own_pooled"]}
    restricted["cross_market"]["scenario_sources"][
        "gaussian_no_cross_target"] = "fb2_no_cross_target"
    assert restricted["generators"]["scenario_count"] == config["export"]["generated_scenario_count"]
    storage = _empty_storage(days, restricted)
    context = mp.get_context("fork")
    with context.Pool(int(config["runtime"]["worker_processes"]), initializer=_init_worker,
                      initargs=(restricted, design, conditional, enriched)) as pool:
        for completed, result in enumerate(pool.imap_unordered(_day_worker, enumerate(days)), 1):
            _store(storage, result, days)
            if completed % 10 == 0 or completed == len(days):
                print(json.dumps({"completed_exports": completed, "total": len(days)}), flush=True)
    return storage


def assembled_products(storage: dict) -> dict:
    gaussian = storage["conditional_full"]["fb2_gate"].copy()
    gaussian[:, :, 0] = storage["conditional"]["own_pooled__gaussian_24_hour"]
    analog = storage["conditional_full"]["empirical_copula"].copy()
    analog[:, :, 0] = storage["conditional"]["own_pooled__analog"]
    products = {"pooled_gaussian__gaussian_24_hour": gaussian,
                "pooled_empirical__analog": analog}
    for prefix, source in [("pooled_gaussian", "own_pooled__gaussian"),
                           ("pooled_empirical", "own_pooled__empirical")]:
        base = storage["scenarios"][source]
        uniform_source = prefix.removeprefix("pooled_")
        values = base[:, :, [0, 2, 3], :].astype(float)
        values[:, :, 0, :] = storage["direct_uniform"][uniform_source]
        products[f"{prefix}__direct_qr"] = values
    return products


def _save_npz(path: Path, values: np.ndarray, days: list, targets: list,
               information_set: str, source_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, delivery_days=np.asarray(days), targets=np.asarray(targets), values=values,
                        seed=1, information_set=information_set, activation_representation="path",
                        evaluated_days=np.asarray(days), source_config_sha256=source_hash)


def write_exports(storage: dict, days: list, config: dict) -> None:
    output = ROOT / config["output"]["directory"]
    count = int(config["export"]["saved_scenario_count"])
    source_path = ROOT / config["export"]["source_config"]
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    for target, source in config["export"]["scenario_0730_sources"].items():
        path = output / config["output"]["scenarios_0730_directory"] / f"{target}.npz"
        _save_npz(path, storage["scenarios"][source][:, :count].astype(np.float32), days,
                  TARGETS, "gate_0730", source_hash)
    for target, values in assembled_products(storage).items():
        path = output / config["output"]["scenarios_1200_directory"] / f"{target}.npz"
        _save_npz(path, values[:, :count].astype(np.float32), days,
                  ["day_ahead_price", "activation_volume", "activation_price"],
                  "gate_1200", source_hash)
    _write_point(storage, days, output, config)


def _write_point(storage: dict, days: list, output: Path, config: dict) -> None:
    rows = [(day, hour, storage["point_1200"][day_index, hour],
             storage["actual"][day_index, TARGETS.index("day_ahead_price"), hour])
            for day_index, day in enumerate(days) for hour in range(24)]
    directory = output / config["output"]["point_1200_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["delivery_day", "hour", "forecast", "actual"]).assign(
        target="day_ahead_price")[
            ["delivery_day", "target", "hour", "forecast", "actual"]].to_csv(
                directory / f'{config["export"]["point_1200_id"]}.csv', index=False)


def _joint_check(storage: dict, days: list, config: dict) -> dict:
    reference = pd.read_csv(ROOT / "p1_paper/results/s3_cross_market/daily_joint_scores.csv")
    source_map = {key: value for key, value in config["export"]["scenario_0730_sources"].items()
                  if key != "pooled_gaussian_no_cross_target"}
    differences = []
    for exported, source in source_map.items():
        expected = reference.loc[reference["arm"].eq(source)].set_index(["delivery_day", "block"])
        for day_index, day in enumerate(days):
            rows = scoring.joint_metrics(storage["actual"][day_index], storage["scenarios"][source][day_index],
                                         storage["scales"][day_index], TARGETS,
                                         config["scoring"]["joint_blocks"],
                                         config["scoring"]["energy_score_beta"],
                                         config["scoring"]["variogram_power"],
                                         config["scoring"]["variogram_weights"])
            for row in rows:
                truth = expected.loc[(day, row["block"]), :]
                differences.extend(abs(row[name] - truth[name]) for name in
                                   ["energy_score", "variogram_score"])
    return {"maximum_absolute_difference": float(max(differences)), "arms": list(source_map)}


def _conditional_check(storage: dict, products: dict, days: list) -> dict:
    reference = pd.read_csv(ROOT / "p1_paper/results/s3_cross_market/daily_conditional_scores.csv")
    candidates = {
        "own_pooled__unconditional": storage["scenarios"]["own_pooled__gaussian"][:, :, 0],
        "own_pooled__gaussian_24_hour": products["pooled_gaussian__gaussian_24_hour"][:, :, 0],
        "own_pooled__analog": products["pooled_empirical__analog"][:, :, 0],
        "own_pooled__direct_quantile_regression":
            storage["conditional"]["own_pooled__direct_quantile_regression"]}
    differences = []
    actual = storage["actual"][:, TARGETS.index("day_ahead_price")]
    for source, values in candidates.items():
        expected = reference.loc[reference["arm"].eq(source)].set_index("delivery_day")
        for day_index, day in enumerate(days):
            mask = np.isfinite(actual[day_index])
            result = scoring.conditional_metrics(actual[day_index], values[day_index].T,
                                                 np.median(values[day_index].T, axis=1), mask)
            differences.append(abs(result["sample_crps"] - expected.loc[day, "sample_crps"]))
    return {"maximum_absolute_difference": float(max(differences)), "products": list(candidates)}


def _point_check(storage: dict, config: dict) -> dict:
    reference = pd.read_csv(ROOT / "p1_paper/results/s3_forecast_side/point_metrics.csv")
    row = reference.loc[reference["arm"].eq(config["export"]["point_1200_id"])
                        & reference["evaluation_set"].eq("main_211")
                        & reference["target"].eq("day_ahead_price")].iloc[0]
    actual = storage["actual"][:, TARGETS.index("day_ahead_price")]
    mae_difference = abs(np.mean(np.abs(storage["point_1200"] - actual)) - row["mae"])
    return {"point_mae_difference": float(mae_difference)}


def _ordered_pair_diagnostics(left: np.ndarray, right: np.ndarray) -> dict:
    assert left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
    inversions, tie_groups, correlations = 0, 0, []
    for day in range(left.shape[0]):
        for hour in range(left.shape[2]):
            x, y = left[day, :, hour], right[day, :, hour]
            order = np.lexsort((y, x))
            x, y = x[order], y[order]
            starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, x.size]
            previous = np.empty(0)
            for start, stop in zip(starts[:-1], starts[1:]):
                current = y[start:stop]
                tie_groups += int(stop - start > 1)
                if previous.size and previous.max() > current.min():
                    inversions += int((previous[:, None] > current[None, :]).sum())
                previous = np.r_[previous, current]
            correlations.append(pd.Series(x).corr(pd.Series(y), method="spearman"))
    return {"inversion_pair_count": inversions, "tie_group_count": tie_groups,
            "minimum_spearman": float(np.nanmin(correlations))}


def _direct_checks(storage: dict, products: dict) -> dict:
    output = {}
    for label, prefix, source in [("gaussian", "pooled_gaussian", "own_pooled__gaussian"),
                                  ("empirical", "pooled_empirical", "own_pooled__empirical")]:
        base = storage["scenarios"][source]
        direct = products[f"{prefix}__direct_qr"]
        assert np.array_equal(direct[:, :, 1:], base[:, :, [2, 3]], equal_nan=True)
        output[label] = _ordered_pair_diagnostics(base[:, :, 0], direct[:, :, 0])
        assert output[label]["inversion_pair_count"] == 0, output
    return output


def validate(storage: dict, days: list, config: dict) -> dict:
    products = assembled_products(storage)
    checks = {"joint_scores": _joint_check(storage, days, config),
              "conditional_crps": _conditional_check(storage, products, days),
              "point": _point_check(storage, config),
              "direct_qr_generated_100": _direct_checks(storage, products)}
    tolerance = float(config["export"]["score_tolerance"])
    assert checks["joint_scores"]["maximum_absolute_difference"] <= tolerance, checks
    assert checks["conditional_crps"]["maximum_absolute_difference"] <= tolerance, checks
    assert checks["point"]["point_mae_difference"] <= tolerance, checks
    return checks


def _read_values(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return archive["values"]


def readback_checks(config: dict) -> dict:
    output = ROOT / config["output"]["directory"]
    early = output / config["output"]["scenarios_0730_directory"]
    late = output / config["output"]["scenarios_1200_directory"]
    checks = {}
    for label, prefix in [("gaussian", "pooled_gaussian"),
                          ("empirical", "pooled_empirical")]:
        base = _read_values(early / f"{prefix}.npz")
        direct = _read_values(late / f"{prefix}__direct_qr.npz")
        assert np.array_equal(direct[:, :, 1:], base[:, :, [2, 3]], equal_nan=True)
        checks[label] = _ordered_pair_diagnostics(base[:, :, 0], direct[:, :, 0])
        assert checks[label]["inversion_pair_count"] == 0, checks
    return checks


def _environment_fingerprint() -> dict:
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__}


def _file_hashes(output: Path) -> dict:
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "summary.json"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def write_wave0_summary(checks: dict, days: list, config: dict) -> None:
    output = ROOT / config["output"]["directory"]
    config_path = ROOT / config["comparison_config_path"]
    source_path = ROOT / config["export"]["source_config"]
    summary = {
        "stage": "wave0_export", "environment": _environment_fingerprint(),
        "sampling_seed": config["export"]["sampling_seed"], "test_day_count": len(days),
        "test_day_sha256": hashlib.sha256("\n".join(days).encode()).hexdigest(),
        "comparison_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "export_checks": checks,
        "acceptance": {"joint_score_reproduction": True, "conditional_crps_reproduction": True,
                       "point_mae_reproduction": True, "uniform_reproduction": True,
                       "direct_qr_zero_inversions_generated_100": True,
                       "direct_qr_zero_inversions_exported_50": True,
                       "direct_qr_activation_dimensions_equal": True},
        "file_sha256": _file_hashes(output)}
    (output / config["output"]["summary_json"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def run(path: str) -> dict:
    config = load_config(path)
    _, design, conditional, enriched, days = build_inputs(config)
    storage = generate(config, design, conditional, enriched, days)
    checks = validate(storage, days, config)
    write_exports(storage, days, config)
    checks["direct_qr_exported_50"] = readback_checks(config)
    write_wave0_summary(checks, days, config)
    maximum_size = max(path.stat().st_size for path in
                       (ROOT / config["output"]["directory"]).rglob("*") if path.is_file())
    assert maximum_size <= 20 * 1024 * 1024
    return {"checks": checks, "maximum_file_bytes": maximum_size, "test_days": len(days)}
