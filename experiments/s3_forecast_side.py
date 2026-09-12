"""Run the frozen DK1 S3-A probabilistic forecast-side experiment."""

import argparse
import ast
import hashlib
import inspect
import json
import multiprocessing as mp
import platform
from pathlib import Path
import sys

import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.s3_forecast_side import copulas, marginals, panel, scoring
from src.s3_forecast_side.generators import (TARGETS, actual_tensor, build_weather_design,
                                             prepare_day, sample_prepared_day)


WORKER_STATE = {}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    config = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert config["protocol"]["missing_value_policy"] == "preserve"
    assert config["protocol"]["time_split_policy"] == "chronological"
    return config


def environment_fingerprint() -> dict:
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__, "lightgbm": lightgbm.__version__}


def _init_worker(config: dict, design: pd.DataFrame, weather: pd.DataFrame,
                 enriched: pd.DataFrame, hourly_panel: pd.DataFrame) -> None:
    WORKER_STATE.update({"config": config, "design": design, "weather": weather,
                         "enriched": enriched, "panel": hourly_panel})


def _latest_visible(hourly: pd.DataFrame, day: str, target: str, config: dict) -> np.ndarray:
    decision = panel.decision_time_utc(day, "gate_0730", config)
    rows = hourly.loc[hourly["target"].eq(target) & hourly["delivery_day"].lt(day)
                      & hourly["available_at"].le(decision)]
    output = np.empty(24)
    for hour in range(24):
        candidates = rows.loc[rows["hour"].eq(hour)].sort_values("available_at")
        valid = candidates.dropna(subset=["value"])
        assert not valid.empty, f"{day} {target} hour {hour}: no persistence value"
        output[hour] = valid.iloc[-1]["value"]
    return output


def _point_models(day: str, blocks: dict) -> dict:
    config, design, enriched = WORKER_STATE["config"], WORKER_STATE["design"], WORKER_STATE["enriched"]
    test = design.loc[design["delivery_day"].eq(day)]
    forecasts = {"persistence": np.stack([
        _latest_visible(WORKER_STATE["panel"], day, target, config) for target in TARGETS])}
    fb0 = np.empty((len(TARGETS), 24))
    for index, target in enumerate(TARGETS):
        model = marginals.fit_fb0(design, blocks["fit"], config, ROOT, target)
        rows = test.loc[test["target"].eq(target)].sort_values("hour")
        fb0[index] = marginals.predict_fb0(model, rows, config)
    forecasts["fb0_lgbm"] = fb0
    capacity_columns = config["model"]["fb0_capacity_features"]
    model = marginals.fit_fb0(enriched, blocks["fit"], config, ROOT,
                              "day_ahead_price", capacity_columns)
    rows = enriched.loc[enriched["delivery_day"].eq(day)
                        & enriched["target"].eq("day_ahead_price")].sort_values("hour")
    forecasts["fb0_lgbm_cap1200"] = marginals.predict_fb0(model, rows, config)
    return forecasts


def _day_worker(item: tuple) -> dict:
    index, day = item
    config = WORKER_STATE["config"]
    prepared = prepare_day(WORKER_STATE["design"], day, config, WORKER_STATE["weather"])
    samples = {}
    for base_seed in config["protocol"]["sampling_seeds"]:
        seed = int(base_seed) + index
        samples[int(base_seed)] = sample_prepared_day(prepared, seed, config)
    points = _point_models(day, prepared["bundle"]["blocks"])
    candidates = prepared["dependence"]["candidates"]
    return {"index": index, "day": day,
            "actual": actual_tensor(prepared["bundle"]["frame"], [day])[0],
            "scales": prepared["target_scales"], "points": points, "samples": samples,
            "p0": prepared["p0"], "quantreg_diagnostics": prepared["quantreg_diagnostics"],
            "rank": prepared["dependence"]["chosen"]["rank"],
            "rank_scores": {str(row["rank"]): row["calibration_energy_score"] for row in candidates},
            "minimum_pair_samples": int(prepared["dependence"]["counts"].min())}


def _capacity_enriched(design: pd.DataFrame, hourly_panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    capacity = hourly_panel.loc[hourly_panel["target"].eq("capacity_price"),
                                ["delivery_day", "hour", "value"]].copy()
    capacity = capacity.rename(columns={"value": "realized_capacity_same_hour"})
    capacity["realized_capacity_daily_mean"] = capacity.groupby("delivery_day")[
        "realized_capacity_same_hour"].transform("mean")
    enriched = design.merge(capacity, on=["delivery_day", "hour"], how="left", validate="many_to_one")
    columns = config["model"]["fb0_capacity_features"]
    assert enriched.loc[enriched["target"].eq("day_ahead_price"), columns].notna().all().all()
    return enriched


def _empty_storage(days: list, config: dict) -> dict:
    day_count = len(days)
    scenario_count = int(config["generators"]["scenario_count"])
    arms = config["generators"]["scenario_arms"]
    seeds = config["protocol"]["sampling_seeds"]
    scenarios = {int(seed): {arm: np.empty((day_count, scenario_count, 4, 24), np.float32)
                             for arm in arms} for seed in seeds}
    conditional = {int(seed): {arm: np.empty((day_count, scenario_count, 3, 24), np.float32)
                               for arm in config["generators"]["conditional_arms"]} for seed in seeds}
    return {"actual": np.empty((day_count, 4, 24)), "scales": np.empty((day_count, 4)),
            "scenarios": scenarios, "conditional": conditional, "points": {},
            "p0": {"fb1_qr": np.full((day_count, 24), np.nan),
                   "fb2_plus_weather": np.full((day_count, 24), np.nan)},
            "diagnostics": []}


def _store_result(storage: dict, result: dict, config: dict) -> None:
    index = result["index"]
    storage["actual"][index] = result["actual"]
    storage["scales"][index] = result["scales"]
    for arm, values in result["p0"].items():
        if values is not None:
            storage["p0"][arm][index] = values
    for arm, values in result["points"].items():
        shape = (storage["actual"].shape[0],) + values.shape
        storage["points"].setdefault(arm, np.empty(shape))[index] = values
    for seed, sampled in result["samples"].items():
        for arm, values in sampled["scenarios"].items():
            storage["scenarios"][seed][arm][index] = values
        for arm, values in sampled["conditional"].items():
            storage["conditional"][seed][arm][index] = values
    storage["diagnostics"].append({key: result[key] for key in
                                   ["day", "rank", "rank_scores", "minimum_pair_samples",
                                    "quantreg_diagnostics"]})


def run_origins(config: dict, hourly_panel: pd.DataFrame, design: pd.DataFrame,
                weather: pd.DataFrame, enriched: pd.DataFrame, days: list) -> dict:
    storage = _empty_storage(days, config)
    processes = int(config["runtime"]["worker_processes"])
    context = mp.get_context("fork")
    initializer = (config, design, weather, enriched, hourly_panel)
    with context.Pool(processes, initializer=_init_worker, initargs=initializer) as pool:
        for completed, result in enumerate(pool.imap_unordered(_day_worker, enumerate(days)), 1):
            _store_result(storage, result, config)
            if completed % 10 == 0 or completed == len(days):
                print(json.dumps({"completed_origins": completed, "total_origins": len(days)}), flush=True)
    storage["diagnostics"].sort(key=lambda row: row["day"])
    return storage


def _load_external_points(storage: dict, days: list) -> None:
    for arm, directory in [("f01_lear", "f01_lear_dk1"), ("f08_lightgbm", "f08_lightgbm_dk1")]:
        frame = pd.read_csv(ROOT / "p1_paper/results" / directory / "forecasts.csv")
        frame["target"] = pd.Categorical(frame["target"], TARGETS, ordered=True)
        frame = frame.loc[frame["delivery_day"].isin(days)].sort_values(["delivery_day", "target", "hour"])
        assert frame["delivery_day"].drop_duplicates().tolist() == days
        storage["points"][arm] = frame["forecast"].to_numpy().reshape(len(days), 4, 24)


def _lag_seven_naive(hourly_panel: pd.DataFrame, days: list) -> np.ndarray:
    lookup = hourly_panel.set_index(["delivery_day", "target", "hour"])["value"]
    output = np.full((len(days), 4, 24), np.nan)
    for day_index, day in enumerate(days):
        lag_day = (pd.Timestamp(day) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        for target_index, target in enumerate(TARGETS):
            for hour in range(24):
                key = (lag_day, target, hour)
                if key in lookup.index:
                    output[day_index, target_index, hour] = lookup.loc[key]
    return output


def _degenerate_mask(audit: pd.DataFrame, days: list) -> np.ndarray:
    cells = audit.loc[audit["audit_scope"].eq("activation_zero_cell")]
    pivot = cells.pivot(index="delivery_day", columns="hour", values="degenerate_zero_cell")
    pivot = pivot.reindex(index=days, columns=range(24))
    assert pivot.notna().all().all()
    return pivot.to_numpy(dtype=bool)


def _weather_audit_rows(weather: pd.DataFrame, days: list, config: dict) -> pd.DataFrame:
    columns = config["weather"]["feature_columns"]
    known = set(config["weather"]["missing_test_days"]) | set(
        config["weather"]["non_test_all_type_missing_days"])
    rows = []
    for day in days:
        blocks = panel.window_days(day, config)
        for target in TARGETS:
            row = {"delivery_day": day, "snapshot": "gate_0730", "target": target,
                   "audit_scope": "weather_window", "visibility_assertion": True}
            for block in ["fit", "calibration"]:
                selected = weather.loc[weather["delivery_day"].isin(blocks[block])
                                       & weather["target"].eq(target)]
                missing = int(selected[columns].isna().any(axis=1).sum())
                calendar_days = known & set(blocks[f"{block}_calendar"])
                raw_missing = len(calendar_days) * config["protocol"]["periods_per_day"]
                row[f"{block}_weather_candidate_rows"] = len(selected)
                row[f"{block}_weather_missing_rows"] = missing
                row[f"{block}_weather_raw_calendar_missing_rows"] = raw_missing
                row[f"{block}_weather_excluded_missing_rows"] = raw_missing - missing
            test = weather.loc[weather["delivery_day"].eq(day) & weather["target"].eq(target)]
            row["test_weather_missing_rows"] = int(test[columns].isna().any(axis=1).sum())
            row["test_weather_evaluated"] = row["test_weather_missing_rows"] == 0
            rows.append(row)
    return pd.DataFrame(rows)


def _augment_weather_audit(audit: pd.DataFrame, weather: pd.DataFrame,
                           days: list, config: dict) -> pd.DataFrame:
    combined = pd.concat([audit, _weather_audit_rows(weather, days, config)], ignore_index=True)
    columns = ["delivery_day", "audit_scope", "snapshot", "target", "hour"]
    combined = combined.sort_values(columns, na_position="last").reset_index(drop=True)
    path = ROOT / config["data"]["panel_audit_csv"]
    combined.to_csv(path, index=False)
    return combined


def _quantreg_audit_rows(storage: dict) -> pd.DataFrame:
    rows = []
    for diagnostic in storage["diagnostics"]:
        for arm, collection in diagnostic["quantreg_diagnostics"].items():
            for unit in collection["units"]:
                group = unit["group"]
                replaced = unit["replaced_levels"].tolist()
                estimated = unit["estimated_levels"].tolist()
                rows.append({"delivery_day": diagnostic["day"], "snapshot": "gate_0730",
                             "target": unit["target"], "hour": group if group != "pooled" else np.nan,
                             "audit_scope": "quantreg_estimable_interval", "arm": arm,
                             "weather": unit["weather"], "positive_part": unit["positive_part"],
                             "quantreg_n": unit["sample_count"],
                             "estimable_quantile_lower": unit["lower"],
                             "estimable_quantile_upper": unit["upper"],
                             "estimated_quantiles": ";".join(str(value) for value in estimated),
                             "endpoint_replaced_quantiles": ";".join(str(value) for value in replaced),
                             "endpoint_replacement_count": len(replaced),
                             "visibility_assertion": True})
    return pd.DataFrame(rows)


def _augment_quantreg_audit(audit: pd.DataFrame, storage: dict, config: dict) -> pd.DataFrame:
    combined = pd.concat([audit, _quantreg_audit_rows(storage)], ignore_index=True)
    columns = ["delivery_day", "audit_scope", "snapshot", "target", "hour", "arm"]
    combined = combined.sort_values(columns, na_position="last").reset_index(drop=True)
    combined.to_csv(ROOT / config["data"]["panel_audit_csv"], index=False)
    return combined


def _target_mask(actual: np.ndarray, target_index: int) -> np.ndarray:
    mask = np.isfinite(actual[:, target_index])
    if TARGETS[target_index] == "activation_price":
        mask &= actual[:, 2] > 0.0
    return mask


def _weather_day_mask(days: list, config: dict) -> np.ndarray:
    missing = set(config["weather"]["missing_test_days"])
    mask = np.asarray([day not in missing for day in days], dtype=bool)
    assert int(mask.sum()) == config["weather"]["evaluated_test_days"]
    return mask


def _evaluation_sets(arm: str, days: list, config: dict) -> list:
    main = np.ones(len(days), dtype=bool)
    common = _weather_day_mask(days, config)
    if arm == "fb2_plus_weather":
        return [("common_209", common)]
    return [("main_211", main), ("common_209", common)]


def _scenario_observations(values: np.ndarray, target_index: int) -> np.ndarray:
    return values[:, :, target_index, :].transpose(0, 2, 1).reshape(-1, values.shape[1])


def point_metric_rows(storage: dict, lag_naive: np.ndarray, days: list, config: dict) -> list:
    rows = []
    forecasts = dict(storage["points"])
    main_seed = int(config["protocol"]["sampling_seeds"][0])
    for arm, scenarios in storage["scenarios"][main_seed].items():
        forecasts[arm] = np.median(scenarios, axis=1)
    for arm, values in forecasts.items():
        for evaluation_set, day_mask in _evaluation_sets(arm, days, config):
            for target_index, target in enumerate(TARGETS):
                actual = storage["actual"][:, target_index].reshape(-1)
                mask = _target_mask(storage["actual"], target_index) & day_mask[:, None]
                forecast = values[:, target_index].reshape(-1) if values.ndim == 3 else values.reshape(-1)
                result = scoring.point_metrics(actual, forecast, lag_naive[:, target_index].reshape(-1),
                                               mask.reshape(-1))
                rows.append({"arm": arm, "evaluation_set": evaluation_set,
                             "n_days": int(day_mask.sum()), "target": target, **result})
    return rows


def probability_metric_rows(storage: dict, degenerate: np.ndarray, days: list,
                            config: dict) -> tuple:
    marginal_rows, tail_rows = [], []
    for seed, arms in storage["scenarios"].items():
        for arm, values in arms.items():
            for evaluation_set, day_mask in _evaluation_sets(arm, days, config):
                _probability_arm_rows(marginal_rows, tail_rows, storage, values, degenerate,
                                      day_mask, evaluation_set, arm, seed, config)
    return marginal_rows, tail_rows


def _probability_arm_rows(marginal_rows: list, tail_rows: list, storage: dict,
                          values: np.ndarray, degenerate: np.ndarray, day_mask: np.ndarray,
                          evaluation_set: str, arm: str, seed: int, config: dict) -> None:
    levels = np.asarray(config["scoring"]["quantile_levels"])
    for target_index, target in enumerate(TARGETS):
        observations = storage["actual"][:, target_index].reshape(-1)
        scenarios = _scenario_observations(values, target_index)
        quantiles = np.nanquantile(scenarios, levels, axis=1).T
        base_mask = (_target_mask(storage["actual"], target_index) & day_mask[:, None]).reshape(-1)
        scopes = [("all_cells", base_mask)]
        if target == "activation_volume":
            scopes.append(("nondegenerate_cells", base_mask & ~degenerate.reshape(-1)))
        _append_marginal_rows(marginal_rows, observations, scenarios, quantiles, scopes,
                              target, arm, seed, config, evaluation_set, int(day_mask.sum()))
        if target in config["scoring"]["pressure_thresholds"]:
            results = scoring.tail_exceedance_metrics(
                observations, quantiles, levels, config["scoring"]["tail_quantile_levels"],
                config["scoring"]["pressure_thresholds"][target], base_mask)
            resolution = 1.0 / config["diagnostics"]["fit_window_empirical_resolution_days"]
            extrapolated = ";".join(str(value) for value in
                                    config["diagnostics"]["extrapolated_quantile_levels"])
            tail_rows.extend({"arm": arm, "seed": seed, "target": target,
                              "evaluation_set": evaluation_set, "n_days": int(day_mask.sum()),
                              "fit_window_empirical_resolution": resolution,
                              "extrapolated_quantile_levels": extrapolated,
                              "extrapolation_note":
                                  "outside_per_unit_estimable_interval_uses_endpoint_extension",
                              "activation_positive_q99_note":
                                  "q99_often_equals_q97_5_endpoint_extension_not_estimate"
                                  if target == "activation_volume" else "not_applicable",
                              **row}
                             for row in results)


def _append_marginal_rows(rows: list, actual: np.ndarray, scenarios: np.ndarray,
                          quantiles: np.ndarray, scopes: list, target: str,
                          arm: str, seed: int, config: dict, evaluation_set: str,
                          n_days: int) -> None:
    levels = config["scoring"]["quantile_levels"]
    for scope, mask in scopes:
        result = scoring.marginal_metrics(actual, quantiles, levels, scenarios,
                                          config["scoring"]["central_interval_levels"], seed, mask)
        base = {key: value for key, value in result.items()
                if key not in ["central_intervals", "pinball_by_quantile"]}
        activation = scoring.activation_volume_metrics(actual, scenarios, 0.0, mask) \
            if target == "activation_volume" else {}
        for interval in result["central_intervals"]:
            rows.append({"arm": arm, "seed": seed, "target": target, "scope": scope,
                         "evaluation_set": evaluation_set, "n_days": n_days,
                         **base, **activation, **interval})


def daily_joint_rows(storage: dict, config: dict) -> list:
    rows = []
    settings = config["scoring"]
    for seed, arms in storage["scenarios"].items():
        for arm, values in arms.items():
            for day_index, observation in enumerate(storage["actual"]):
                if not np.isfinite(values[day_index]).all():
                    assert arm == "fb2_plus_weather", "unexpected scenario missingness"
                    continue
                metrics = scoring.joint_metrics(observation, values[day_index],
                                                storage["scales"][day_index], TARGETS,
                                                settings["joint_blocks"], settings["energy_score_beta"],
                                                settings["variogram_power"], settings["variogram_weights"])
                rows.extend({"delivery_day_index": day_index, "arm": arm, "seed": seed, **row}
                            for row in metrics)
    return rows


def aggregate_joint_rows(daily: pd.DataFrame, days: list, config: dict) -> list:
    rows = []
    for arm in sorted(daily["arm"].unique()):
        for evaluation_set, day_mask in _evaluation_sets(arm, days, config):
            indices = set(np.flatnonzero(day_mask).tolist())
            selected = daily.loc[daily["arm"].eq(arm) & daily["delivery_day_index"].isin(indices)]
            for (seed, block), frame in selected.groupby(["seed", "block"], sort=True):
                assert len(frame) == int(day_mask.sum())
                rows.append({"arm": arm, "seed": seed, "block": block,
                             "evaluation_set": evaluation_set, "n_days": len(frame),
                             "energy_score_mean": frame["energy_score"].mean(),
                             "variogram_score_mean": frame["variogram_score"].mean()})
    return rows


def dependence_rows(storage: dict, days: list, config: dict) -> list:
    rows = []
    settings = config["scoring"]
    for seed, arms in storage["scenarios"].items():
        for arm, values in arms.items():
            for evaluation_set, day_mask in _evaluation_sets(arm, days, config):
                results = scoring.dependence_metrics(storage["actual"][day_mask], values[day_mask], TARGETS,
                                                     settings["joint_targets"],
                                                     settings["dependence_lags_hours"],
                                                     settings["upper_tail_quantile"])
                rows.extend({"arm": arm, "seed": seed, "evaluation_set": evaluation_set,
                             "n_days": int(day_mask.sum()), **result} for result in results)
    return rows


def _conditional_sources(storage: dict, seed: int) -> dict:
    sources = {"fb2_gate_conditional": storage["conditional"][seed]["fb2_gate"],
               "empirical_copula_conditional": storage["conditional"][seed]["empirical_copula"]}
    indices = [0, 2, 3]
    sources["fb2_gate_unconditional"] = storage["scenarios"][seed]["fb2_gate"][:, :, indices]
    sources["fb1_qr"] = storage["scenarios"][seed]["fb1_qr"][:, :, indices]
    for arm in ["fb0_lgbm_cap1200", "f08_lightgbm"]:
        point = storage["points"][arm]
        day_ahead = point if point.ndim == 2 else point[:, 0]
        sources[arm] = np.repeat(day_ahead[:, None, None, :], 100, axis=1)
    return sources


def conditional_rows(storage: dict, config: dict) -> tuple:
    aggregate, daily = [], []
    actual = storage["actual"][:, 0]
    mask = np.isfinite(actual)
    for seed in config["protocol"]["sampling_seeds"]:
        for arm, values in _conditional_sources(storage, int(seed)).items():
            day_ahead = values[:, :, 0] if values.ndim == 4 else values[:, :, 0]
            for day_index in range(actual.shape[0]):
                result = scoring.conditional_metrics(actual[day_index], day_ahead[day_index],
                                                     np.median(day_ahead[day_index], axis=0),
                                                     mask[day_index])
                daily.append({"delivery_day_index": day_index, "arm": arm,
                              "seed": seed, **result})
            frame = pd.DataFrame([row for row in daily if row["arm"] == arm and row["seed"] == seed])
            aggregate.append({"arm": arm, "seed": seed, "sample_crps": frame["sample_crps"].mean(),
                              "median_mae": frame["median_mae"].mean(), "n_days": len(frame)})
    return aggregate, daily


def significance_rows(daily_joint: pd.DataFrame, conditional_daily: pd.DataFrame,
                      days: list, config: dict) -> list:
    rows = []
    main_seed = config["protocol"]["sampling_seeds"][0]
    base = daily_joint.loc[(daily_joint["seed"] == main_seed) & daily_joint["block"].eq("all")]
    comparisons = ["fb1_qr", "empirical_copula", "x07_static_copula",
                   "x14_hist_independent", "x09_block_copula"]
    for comparator in comparisons:
        row = _significance_pair(base, "fb2_gate", comparator, "energy_score", config)
        rows.append({"evaluation_set": "main_211", "n_days": len(days), **row})
    common_indices = set(np.flatnonzero(_weather_day_mask(days, config)).tolist())
    common = base.loc[base["delivery_day_index"].isin(common_indices)]
    row = _significance_pair(common, "fb2_plus_weather", "fb2_gate",
                             "energy_score", config)
    rows.append({"evaluation_set": "common_209", "n_days": len(common_indices), **row})
    conditional = conditional_daily.loc[conditional_daily["seed"] == main_seed]
    row = _significance_pair(conditional, "fb2_gate_conditional",
                             "fb2_gate_unconditional", "sample_crps", config)
    rows.append({"evaluation_set": "main_211", "n_days": len(days), **row})
    return rows


def _significance_pair(frame: pd.DataFrame, arm: str, comparator: str,
                       metric: str, config: dict) -> dict:
    if comparator not in set(frame["arm"]):
        return {"arm": arm, "comparator": comparator, "metric": metric,
                "status": "skipped_undetermined_published_method"}
    left = frame.loc[frame["arm"].eq(arm)].sort_values("delivery_day_index")[metric].to_numpy()
    right = frame.loc[frame["arm"].eq(comparator)].sort_values("delivery_day_index")[metric].to_numpy()
    settings = config["scoring"]
    result = scoring.significance_metrics(left, right, settings["dm_newey_west_lags"],
                                          settings["dm_sidedness"], settings["bootstrap_block_days"],
                                          settings["bootstrap_repetitions"],
                                          settings["bootstrap_confidence_level"], settings["bootstrap_seed"])
    return {"arm": arm, "comparator": comparator, "metric": metric, "status": "computed", **result}


def monthly_rows(storage: dict, days: list, config: dict) -> list:
    rows = []
    seed = int(config["protocol"]["sampling_seeds"][0])
    for arm, values in storage["scenarios"][seed].items():
        for evaluation_set, day_mask in _evaluation_sets(arm, days, config):
            selected_days = np.asarray(days)[day_mask].tolist()
            for index, target in enumerate(TARGETS):
                results = scoring.monthly_calibration_metrics(
                    selected_days, storage["actual"][day_mask, index], values[day_mask, :, index], 0.90,
                    _target_mask(storage["actual"], index)[day_mask])
                rows.extend({"arm": arm, "target": target, "evaluation_set": evaluation_set,
                             "n_days": int(day_mask.sum()), **row} for row in results
                            if "2026-02" <= row["month"] <= "2026-08")
    return rows


def _write_point_files(storage: dict, days: list, config: dict, output: Path) -> None:
    directory = output / config["output"]["point_0730_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    forecasts = dict(storage["points"])
    seed = int(config["protocol"]["sampling_seeds"][0])
    forecasts.update({arm: np.median(values, axis=1) for arm, values in storage["scenarios"][seed].items()})
    for arm, values in forecasts.items():
        if values.ndim == 2:
            continue
        rows = []
        for day_index, day in enumerate(days):
            for target_index, target in enumerate(TARGETS):
                for hour in range(24):
                    rows.append((day, target, hour, values[day_index, target_index, hour],
                                 storage["actual"][day_index, target_index, hour]))
        pd.DataFrame(rows, columns=["delivery_day", "target", "hour", "forecast", "actual"]).to_csv(
            directory / f"{arm}.csv", index=False)


def _write_scenario_files(storage: dict, days: list, config: dict, output: Path) -> None:
    count = int(config["generators"]["export_scenario_count"])
    seed = int(config["protocol"]["sampling_seeds"][0])
    directory = output / config["output"]["scenarios_0730_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    for arm, values in storage["scenarios"][seed].items():
        information = "beyond_gate_upper_bound" if arm == "fb2_plus_weather" else "gate_0730"
        evaluated = np.asarray(days)[_weather_day_mask(days, config)] if arm == "fb2_plus_weather" \
            else np.asarray(days)
        np.savez_compressed(directory / f"{arm}.npz", delivery_days=np.asarray(days), targets=np.asarray(TARGETS),
                            values=values[:, :count].astype(np.float32), seed=seed,
                            information_set=information, activation_representation="path",
                            evaluated_days=evaluated)
    conditional_dir = output / config["output"]["scenarios_1200_directory"]
    conditional_dir.mkdir(parents=True, exist_ok=True)
    names = np.asarray(["day_ahead_price", "activation_volume", "activation_price"])
    for arm, values in storage["conditional"][seed].items():
        np.savez_compressed(conditional_dir / f"{arm}.npz", delivery_days=np.asarray(days), targets=names,
                            values=values[:, :count].astype(np.float32), seed=seed,
                            information_set="gate_1200", activation_representation="path",
                            conditioning="realized_capacity_price")


def _file_hashes(output: Path) -> dict:
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "summary.json"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def _deviation_summary(audit: pd.DataFrame) -> dict:
    cells = audit.loc[audit["audit_scope"].eq("activation_zero_cell")]
    degenerate = cells.loc[cells["degenerate_zero_cell"].astype(bool)]
    distribution = cells["fit_zero_events"].astype(int).value_counts().sort_index()
    assert degenerate["delivery_actual_positive"].astype(bool).all()
    return {"rule": "p0_zero_when_no_zero_events", "total_cells": len(degenerate),
            "test_days": degenerate["delivery_day"].nunique(),
            "hours": sorted(degenerate["hour"].astype(int).unique().tolist()),
            "delivery_day_actual_activation_all_positive": True,
            "fit_zero_event_count_distribution": {str(key): int(value) for key, value in distribution.items()}}


def _p0_summary(storage: dict, days: list, config: dict) -> dict:
    output = {}
    levels = config["diagnostics"]["p0_summary_quantiles"]
    low = config["diagnostics"]["p0_low_threshold"]
    high = config["diagnostics"]["p0_high_threshold"]
    for arm, matrix in storage["p0"].items():
        mask = _weather_day_mask(days, config) if arm == "fb2_plus_weather" \
            else np.ones(len(days), dtype=bool)
        values = matrix[mask].reshape(-1)
        assert np.isfinite(values).all() and np.all((values >= 0.0) & (values <= 1.0))
        quantiles = np.quantile(values, levels)
        output[arm] = {"evaluated_cells": len(values),
                       "quantiles": {str(level): float(value)
                                     for level, value in zip(levels, quantiles)},
                       "p0_ge_0_995_cells": int(np.sum(values >= high)),
                       "p0_le_0_005_cells": int(np.sum(values <= low)),
                       "p0_exactly_one_cells": int(np.sum(values == 1.0)),
                       "p0_exactly_zero_cells": int(np.sum(values == 0.0))}
    return output


def _quantreg_summary(storage: dict, config: dict) -> dict:
    records = [diagnostic for row in storage["diagnostics"]
               for diagnostic in row["quantreg_diagnostics"].values()
               if diagnostic is not None]
    settings = config["model"]["quantreg"]
    limit_hits = sum(record["iteration_limit_count"] for record in records)
    assert limit_hits == 0, "a completed run cannot contain QuantReg iteration-limit hits"
    units = [unit for record in records for unit in record["units"]]
    replaced = [unit for unit in units if len(unit["replaced_levels"]) > 0]
    levels = sorted({float(level) for unit in replaced for level in unit["replaced_levels"]})
    activation = [unit for unit in units if unit["positive_part"]]
    q99_to_q975 = [unit for unit in activation if 0.99 in unit["replaced_levels"]
                   and unit["estimated_levels"][-1] == 0.975]
    return {"configured_tolerance": settings["tolerance"],
            "statsmodels_default_tolerance": settings["statsmodels_default_tolerance"],
            "configured_max_iter": settings["max_iter"],
            "statsmodels_default_max_iter": settings["statsmodels_default_max_iter"],
            "fit_count": sum(record["fit_count"] for record in records),
            "maximum_observed_iterations": max(record["maximum_iterations"] for record in records),
            "iteration_limit_cell_count": limit_hits,
            "unit_count": len(units),
            "unit_count_with_endpoint_replacement": len(replaced),
            "endpoint_replacement_cell_count": sum(len(unit["replaced_levels"])
                                                   for unit in replaced),
            "endpoint_replaced_quantile_levels": levels,
            "activation_positive_unit_count": len(activation),
            "activation_positive_q99_equals_q97_5_unit_count": len(q99_to_q975)}


def _p0_deviations(summary: dict) -> dict:
    return {arm: {"p0_ge_0_995_cells": values["p0_ge_0_995_cells"],
                  "p0_le_0_005_cells": values["p0_le_0_005_cells"],
                  "note": "both extreme-probability tails contain test cells"}
            for arm, values in summary.items()
            if values["p0_ge_0_995_cells"] > 0 and values["p0_le_0_005_cells"] > 0}


def write_results(storage: dict, days: list, hourly_panel: pd.DataFrame,
                  audit: pd.DataFrame, config: dict) -> None:
    output = ROOT / config["output"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    lag_naive = _lag_seven_naive(hourly_panel, days)
    degenerate = _degenerate_mask(audit, days)
    point_rows = point_metric_rows(storage, lag_naive, days, config)
    marginal_rows, tail_rows = probability_metric_rows(storage, degenerate, days, config)
    daily_joint = pd.DataFrame(daily_joint_rows(storage, config))
    conditional, conditional_daily = conditional_rows(storage, config)
    tables = {"point_metrics_csv": point_rows, "marginal_metrics_csv": marginal_rows,
              "tail_metrics_csv": tail_rows,
              "joint_metrics_csv": aggregate_joint_rows(daily_joint, days, config),
              "dependence_errors_csv": dependence_rows(storage, days, config),
              "conditional_1200_csv": conditional,
              "significance_csv": significance_rows(
                  daily_joint, pd.DataFrame(conditional_daily), days, config),
              "monthly_calibration_csv": monthly_rows(storage, days, config)}
    for key, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / config["output"][key], index=False)
    daily_joint.assign(delivery_day=daily_joint["delivery_day_index"].map(dict(enumerate(days)))).to_csv(
        output / config["output"]["daily_scores_csv"], index=False)
    _write_point_files(storage, days, config, output)
    _write_scenario_files(storage, days, config, output)
    _write_summary(storage, days, audit, config, output)


def _write_summary(storage: dict, days: list, audit: pd.DataFrame,
                   config: dict, output: Path) -> None:
    test_hash = hashlib.sha256("\n".join(days).encode()).hexdigest()
    skipped = {key: value["skipped_reason"] for key, value in config["published_generators"].items()
               if not value["run"]}
    ranks = pd.Series([row["rank"] for row in storage["diagnostics"]]).value_counts().sort_index()
    weather = config["weather"]
    weather_days = sorted(weather["missing_test_days"])
    coverage = {"fb2_plus_weather": {
        "evaluation_day_count": weather["evaluated_test_days"],
        "total_test_day_count": len(days),
        "missing_days": [{"delivery_day": day,
                          "missing_forecast_types": weather["missing_test_days"][day]}
                         for day in weather_days]}}
    p0_summary = _p0_summary(storage, days, config)
    deviations = {"activation_zero_boundary": _deviation_summary(audit),
                  "quantreg_solver": _quantreg_summary(storage, config)}
    p0_deviations = _p0_deviations(p0_summary)
    if p0_deviations:
        deviations["p0_extreme_probabilities"] = p0_deviations
    summary = {"environment": environment_fingerprint(), "sampling_seeds": config["protocol"]["sampling_seeds"],
               "window": config["window"], "test_day_count": len(days), "test_day_sha256": test_hash,
               "arm_day_coverage": coverage,
               "arms": config["generators"]["arms"], "skipped_arms": skipped,
               "published_generator_deviations": config["published_generators"],
               "p0_summary": p0_summary, "deviations": deviations,
               "logistic_intercept_unpenalized": True,
               "rank_selection_counts": {str(key): int(value) for key, value in ranks.items()},
               "minimum_pair_samples": min(row["minimum_pair_samples"] for row in storage["diagnostics"]),
               "acceptance": {"test_days_match_s2": len(days) == config["split"]["expected_test_days"],
                              "missing_preserved": True, "test_used_for_selection": False,
                              "information_sets_equal_except_fb2_plus": True},
               "file_sha256": _file_hashes(output), "incomplete": []}
    (output / config["output"]["summary_json"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _conditional_hand_check(config: dict) -> dict:
    sigma = np.asarray([[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]])
    conditioned = np.asarray([1])
    value = np.asarray([0.7])
    active, uniforms = copulas.conditional_gaussian(sigma, conditioned, value, 50000, 11, config)
    samples = scipy.stats.norm.ppf(uniforms)
    expected_mean = sigma[np.ix_(active, conditioned)] @ np.linalg.solve(
        sigma[np.ix_(conditioned, conditioned)], value)
    expected_cov = sigma[np.ix_(active, active)] - sigma[np.ix_(active, conditioned)] @ np.linalg.solve(
        sigma[np.ix_(conditioned, conditioned)], sigma[np.ix_(conditioned, active)])
    mean_error = float(np.max(np.abs(samples.mean(axis=0) - expected_mean)))
    covariance_error = float(np.max(np.abs(np.cov(samples, rowvar=False) - expected_cov)))
    assert max(mean_error, covariance_error) <= config["scoring"]["synthetic_check_tolerance"]
    return {"conditional_mean_error": mean_error, "conditional_covariance_error": covariance_error}


def _function_length_check() -> None:
    paths = [ROOT / "src/s3_forecast_side" / name for name in
             ["panel.py", "scoring.py", "marginals.py", "copulas.py", "generators.py"]]
    paths.append(Path(__file__))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lengths = [node.end_lineno - node.lineno + 1 for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert max(lengths) <= 60, f"{path}: function exceeds 60 lines"


def _logistic_intercept_check(config: dict) -> None:
    settings = config["model"]["activation_logistic"]
    assert settings["solver"] == "lbfgs" and settings["intercept_penalty"] == "none"
    source = inspect.getsource(sklearn.linear_model._logistic._logistic_loss_and_grad)
    assert "grad[-1] = z0.sum()" in source
    assert "alpha * w" in source


def run_checks(config: dict) -> None:
    results = {"scoring": scoring.hand_checks(config),
               "panel": panel.panel_checks(config, ROOT),
               "conditional": _conditional_hand_check(config)}
    _function_length_check()
    _logistic_intercept_check(config)
    hourly_panel = panel.build_hourly_panel(config, ROOT)
    design = panel.build_all_features(hourly_panel, "gate_0730", config)
    weather = build_weather_design(design, config, ROOT)
    days = panel.test_delivery_days(config, ROOT)
    assert int(_weather_day_mask(days, config).sum()) == config["weather"]["evaluated_test_days"]
    prepared = prepare_day(design, config["split"]["reference_delivery_day"], config, weather)
    assert prepared["dependence"]["chosen"]["rank"] in config["copula"]["rank_candidates"]
    reference = yaml.safe_load((ROOT / config["model"]["fb0_config"]).read_text())["model"]["params"]
    assert marginals._lgbm_parameters(config, ROOT) == reference
    print(json.dumps({"checks": results, "reference_rank": prepared["dependence"]["chosen"]["rank"]},
                     indent=2, sort_keys=True))


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if arguments.check:
        run_checks(config)
        return
    hourly_panel = panel.build_hourly_panel(config, ROOT)
    design = panel.build_all_features(hourly_panel, "gate_0730", config)
    weather = build_weather_design(design, config, ROOT)
    enriched = _capacity_enriched(design, hourly_panel, config)
    days = panel.test_delivery_days(config, ROOT)
    audit = panel.write_panel_audit(hourly_panel, config, ROOT)
    audit = _augment_weather_audit(audit, weather, days, config)
    storage = run_origins(config, hourly_panel, design, weather, enriched, days)
    audit = _augment_quantreg_audit(audit, storage, config)
    _load_external_points(storage, days)
    write_results(storage, days, hourly_panel, audit, config)


if __name__ == "__main__":
    main()
