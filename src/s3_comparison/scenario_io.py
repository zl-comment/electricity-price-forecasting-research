"""Load and transform the frozen S3-B scenarios on one common protocol."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from s3_forecast_side.panel import decision_time_utc, test_delivery_days, window_days


TARGETS_0730 = ["day_ahead_price", "capacity_price", "activation_volume", "activation_price"]
TARGETS_1200 = ["day_ahead_price", "activation_volume", "activation_price"]
INHERITED_0730 = ["hist_paired", "hist_independent", "climatology", "x09_block_copula",
                  "fb2_gate", "fb2_plus_weather"]
INHERITED_1200 = ["fb2_gate", "fb2_plus_weather"]
EXPORTED_1200 = ["pooled_gaussian__gaussian_24_hour", "pooled_empirical__analog",
                 "pooled_gaussian__direct_qr", "pooled_empirical__direct_qr"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _scalar(archive: np.lib.npyio.NpzFile, key: str):
    value = archive[key]
    _require(value.shape == (), f"{key}: expected scalar metadata")
    return value.item()


def load_scenario_archive(path: Path, expected_days: list, expected_targets: list,
                          information_set: str, evaluated_days: list,
                          scenario_count: int, expected_hours: int, expected_seed: int,
                          source_hash: str = None) -> dict:
    """Return one archive after exact day, shape, metadata, and missing-value checks."""
    required = {"delivery_days", "targets", "values", "seed", "information_set",
                "activation_representation", "evaluated_days"}
    with np.load(path, allow_pickle=False) as archive:
        _require(required.issubset(archive.files), f"{path}: missing archive fields")
        days = archive["delivery_days"].astype(str).tolist()
        targets = archive["targets"].astype(str).tolist()
        evaluated = archive["evaluated_days"].astype(str).tolist()
        values = archive["values"].copy()
        metadata = {"seed": int(_scalar(archive, "seed")),
                    "information_set": str(_scalar(archive, "information_set")),
                    "activation_representation": str(_scalar(archive, "activation_representation"))}
        if source_hash is not None:
            _require("source_config_sha256" in archive.files, f"{path}: missing source hash")
            _require(str(_scalar(archive, "source_config_sha256")) == source_hash,
                     f"{path}: source config hash changed")
    _require(days == expected_days, f"{path}: delivery days differ day by day")
    _require(targets == expected_targets, f"{path}: target order differs")
    shape = (len(expected_days), scenario_count, len(expected_targets), expected_hours)
    _require(values.shape == shape, f"{path}: expected {shape}, got {values.shape}")
    _require(values.dtype == np.float32, f"{path}: values are not float32")
    _require(metadata["seed"] == expected_seed, f"{path}: sampling seed changed")
    _require(metadata["information_set"] == information_set, f"{path}: wrong information set")
    _require(evaluated == evaluated_days, f"{path}: evaluated days differ day by day")
    _require(metadata["activation_representation"] in {"path", "expected_value"},
             f"{path}: unknown activation representation")
    _require(not np.isinf(values).any(), f"{path}: infinite scenario value")
    indices = [days.index(day) for day in evaluated]
    non_price = [index for index, target in enumerate(targets) if target != "activation_price"]
    _require(np.isfinite(values[np.ix_(indices, range(scenario_count), non_price, range(24))]).all(),
             f"{path}: non-activation-price value missing on evaluated days")
    return {"path": path, "delivery_days": days, "targets": targets, "values": values,
            "evaluated_days": evaluated, **metadata}


def _manifest(root: Path, config: dict, forecast: dict, days: list) -> list:
    comparison = root / config["output"]["directory"]
    upstream = root / forecast["output"]["directory"]
    full, weather = days, [day for day in days if day not in forecast["weather"]["missing_test_days"]]
    count = int(config["export"]["saved_scenario_count"])
    hours = int(forecast["protocol"]["periods_per_day"])
    export_seed = int(config["export"]["sampling_seed"])
    inherited_seed = int(forecast["protocol"]["sampling_seeds"][0])
    source_path = root / config["export"]["source_config"]
    source_hash = sha256(source_path.read_bytes()).hexdigest()
    rows = []
    for name in config["export"]["scenario_0730_sources"]:
        rows.append((comparison / config["output"]["scenarios_0730_directory"] / f"{name}.npz",
                     TARGETS_0730, "gate_0730", full, count, hours, export_seed, source_hash))
    for name in EXPORTED_1200:
        rows.append((comparison / config["output"]["scenarios_1200_directory"] / f"{name}.npz",
                     TARGETS_1200, "gate_1200", full, count, hours, export_seed, source_hash))
    for name in INHERITED_0730:
        info, evaluated = ("beyond_gate_upper_bound", weather) if name == "fb2_plus_weather" \
            else ("gate_0730", full)
        rows.append((upstream / forecast["output"]["scenarios_0730_directory"] / f"{name}.npz",
                     TARGETS_0730, info, evaluated, count, hours, inherited_seed, None))
    for name in INHERITED_1200:
        info, evaluated = ("beyond_gate_upper_bound", weather) if name == "fb2_plus_weather" \
            else ("gate_1200", full)
        rows.append((upstream / forecast["output"]["scenarios_1200_directory"] / f"{name}.npz",
                     TARGETS_1200, info, evaluated, count, hours, inherited_seed, None))
    return rows


def load_common_scenarios(repository_root: Path, config: dict, forecast: dict,
                          expected_days: list) -> dict:
    """Load every Wave-0 and inherited archive listed by the frozen S3-B task."""
    loaded = {}
    lower = forecast["targets"]["activation_volume"]["physical_lower_bound"]
    for path, targets, information, evaluated, count, hours, seed, source_hash in _manifest(
            repository_root, config, forecast, expected_days):
        key = str(path.relative_to(repository_root))
        loaded[key] = load_scenario_archive(path, expected_days, targets, information,
                                             evaluated, count, hours, seed, source_hash)
        volume = loaded[key]["values"][:, :, targets.index("activation_volume")]
        _require(not (volume < lower).any(), f"{path}: activation volume below configured bound")
    return loaded


def _delivery_labels(stamps: pd.Series, timezone: str) -> pd.Series:
    return stamps.dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")


def load_premium_history(repository_root: Path, config: dict, forecast: dict,
                         storage: dict) -> pd.DataFrame:
    """Return observed UTC-keyed imbalance premia; missing values remain missing."""
    root = repository_root / forecast["data"]["root"]
    area = forecast["data"]["primary_price_area"]
    price_spec = forecast["data"]["datasets"][forecast["targets"]["day_ahead_price"]["dataset"]]
    act_spec = forecast["data"]["datasets"][forecast["targets"]["activation_volume"]["dataset"]]
    price = pd.read_csv(root / price_spec["csv"], usecols=[price_spec["time_column"],
                                                          price_spec["area_column"],
                                                          forecast["targets"]["day_ahead_price"]["column"]])
    activation_columns = [act_spec["time_column"], act_spec["area_column"],
                          forecast["targets"]["activation_volume"]["column"],
                          storage["settlement"]["shortfall_price_column"]]
    activation = pd.read_csv(root / act_spec["csv"], usecols=activation_columns)
    price = price.loc[price[price_spec["area_column"]].eq(area)].copy()
    activation = activation.loc[activation[act_spec["area_column"]].eq(area)].copy()
    price["timestamp_utc"] = pd.to_datetime(price[price_spec["time_column"]], utc=True)
    activation["timestamp_utc"] = pd.to_datetime(activation[act_spec["time_column"]], utc=True)
    price = price[["timestamp_utc", forecast["targets"]["day_ahead_price"]["column"]]]
    activation = activation[["timestamp_utc", forecast["targets"]["activation_volume"]["column"],
                             storage["settlement"]["shortfall_price_column"]]]
    joined = activation.merge(price, on="timestamp_utc", how="inner", validate="one_to_one")
    _require(len(joined) == len(activation), "UTC price join changed imbalance row count")
    timezone = forecast["protocol"]["civil_timezone"]
    joined["delivery_day"] = _delivery_labels(joined["timestamp_utc"], timezone)
    return _premium_columns(joined, config, forecast, storage)


def _publication_availability(delivery_days: pd.Series, spec: dict) -> pd.Series:
    _require(spec["kind"] == "next_danish_bank_day_publication",
             "Unknown premium publication rule")
    weekdays = set(int(value) for value in spec["working_weekdays"])
    holidays = set(spec["holidays"])
    _require(weekdays == {0, 1, 2, 3, 4}, "Danish working weekdays changed")
    mapping = {}
    for label in sorted(delivery_days.unique()):
        candidate = pd.Timestamp(label) + pd.DateOffset(days=1)
        while candidate.weekday() not in weekdays or candidate.strftime("%Y-%m-%d") in holidays:
            candidate += pd.DateOffset(days=1)
        local = pd.Timestamp(
            f'{candidate.strftime("%Y-%m-%d")} {spec["publication_civil_time"]}',
            tz=spec["civil_timezone"],
        )
        mapping[label] = local.tz_convert("UTC")
    available = delivery_days.map(mapping)
    _require(available.notna().all(), "Premium publication timestamp is missing")
    return pd.to_datetime(available, utc=True)


def _premium_columns(frame: pd.DataFrame, config: dict, forecast: dict,
                     storage: dict) -> pd.DataFrame:
    volume = forecast["targets"]["activation_volume"]["column"]
    price = forecast["targets"]["day_ahead_price"]["column"]
    shortfall = storage["settlement"]["shortfall_price_column"]
    quarters = int(forecast["protocol"]["quarters_per_hour"])
    ordered = frame.sort_values(["delivery_day", "timestamp_utc"]).reset_index(drop=True)
    ordered["quarter_index"] = ordered.groupby("delivery_day").cumcount()
    ordered["hour"] = ordered["quarter_index"] // quarters
    counts = ordered.groupby(["delivery_day", "hour"]).size()
    _require(counts.eq(quarters).all(), "Historical UTC hours do not contain exact quarter counts")
    states = ordered.groupby(["delivery_day", "hour"])[volume].agg(
        activation_valid=lambda values: values.notna().all(), activation_sum="sum").reset_index()
    ordered = ordered.merge(states, on=["delivery_day", "hour"], validate="many_to_one")
    lower = forecast["targets"]["activation_volume"]["physical_lower_bound"]
    ordered["activation_positive"] = np.where(
        ordered["activation_valid"], ordered["activation_sum"] > lower, np.nan)
    ordered["premium"] = ordered[shortfall] - ordered[price]
    ordered["available_at"] = _publication_availability(
        ordered["delivery_day"], config["premium_availability"])
    boundaries = forecast["features"]["activation_period_boundaries"]
    ordered["period"] = pd.cut(ordered["hour"], boundaries, right=False, labels=False)
    return ordered[["delivery_day", "timestamp_utc", "available_at", "hour", "period",
                    "activation_positive", "premium"]]


def premium_pools(history: pd.DataFrame, delivery_day: str, snapshot: str,
                  forecast: dict) -> dict:
    """Build F(D) four-hour/state pools and assert every retained input was available."""
    blocks = window_days(delivery_day, forecast)
    selected = history.loc[history["delivery_day"].isin(blocks["fit"])].copy()
    decision = decision_time_utc(delivery_day, snapshot, forecast)
    _require(selected["available_at"].le(decision).all(),
             f"{delivery_day}: premium input available after {snapshot}")
    boundaries = forecast["features"]["activation_period_boundaries"]
    pools = {}
    for period in range(len(boundaries) - 1):
        for state in (False, True):
            rows = selected.loc[selected["period"].eq(period)
                                & selected["activation_positive"].eq(state)]
            values = rows["premium"].dropna().to_numpy(dtype=float)
            _require(len(values) > 0, f"{delivery_day} period {period} state {state}: empty premium pool")
            pools[(period, state)] = values
    return pools


def _resize_hours(values: np.ndarray, hours: int, policy: str) -> np.ndarray:
    _require(values.ndim == 2, "Hourly scenario values must be scenario by hour")
    if values.shape[1] == hours:
        return values.copy()
    _require(policy == "truncate_or_repeat_last", f"Unknown DST policy: {policy}")
    if values.shape[1] > hours:
        return values[:, :hours].copy()
    repeat = np.repeat(values[:, -1:], hours - values.shape[1], axis=1)
    return np.concatenate([values, repeat], axis=1)


def hourly_to_quarters(values: np.ndarray, hours: int, quantity: str,
                       forecast: dict) -> np.ndarray:
    """Broadcast hourly prices or split hourly activation MWh across quarter slots."""
    resized = _resize_hours(np.asarray(values, dtype=float), hours,
                            forecast["protocol"]["dst_policy"])
    quarters = int(forecast["protocol"]["quarters_per_hour"])
    expanded = np.repeat(resized, quarters, axis=1)
    if quantity == "activation_energy":
        return expanded / quarters
    _require(quantity == "price", f"Unknown hourly quantity: {quantity}")
    return expanded


def sample_imbalance_prices(day_ahead: np.ndarray, activation_hourly: np.ndarray,
                            pools: dict, seed: int, forecast: dict) -> np.ndarray:
    """Sample F(D) premia by four-hour period and hourly activation state."""
    _require(seed is not None, "A sampling seed is required")
    scenarios, hours = activation_hourly.shape
    quarters = int(forecast["protocol"]["quarters_per_hour"])
    _require(day_ahead.shape == (scenarios, hours * quarters), "Day-ahead quarter shape differs")
    boundaries = forecast["features"]["activation_period_boundaries"]
    lower = forecast["targets"]["activation_volume"]["physical_lower_bound"]
    _require(np.isfinite(activation_hourly).all(), "Activation state contains missing values")
    rng, premium = np.random.default_rng(seed), np.empty_like(day_ahead)
    for scenario in range(scenarios):
        for hour in range(hours):
            period = np.searchsorted(boundaries, hour, side="right") - 1
            state = bool(activation_hourly[scenario, hour] > lower)
            pool = pools[(period, state)]
            start = hour * quarters
            premium[scenario, start:start + quarters] = rng.choice(pool, size=quarters, replace=True)
    return day_ahead + premium


def prepare_day_scenarios(archive: dict, delivery_day: str, procured_mw: np.ndarray,
                          premium_history: pd.DataFrame, forecast: dict,
                          snapshot: str, seed: int) -> dict:
    """Return one day's common quarter-hour scenario arrays and sampled imbalance prices."""
    _require(delivery_day in archive["evaluated_days"], f"{delivery_day}: archive is not evaluated")
    index = archive["delivery_days"].index(delivery_day)
    source = archive["values"][index].astype(float)
    mapped = {target: source[:, position] for position, target in enumerate(archive["targets"])}
    hours = len(procured_mw)
    _require(np.isfinite(procured_mw).all() and (procured_mw > 0).all(),
             f"{delivery_day}: procurement must be finite and positive")
    day_ahead = hourly_to_quarters(mapped["day_ahead_price"], hours, "price", forecast)
    activation_hourly = _resize_hours(mapped["activation_volume"], hours,
                                      forecast["protocol"]["dst_policy"])
    activation = hourly_to_quarters(mapped["activation_volume"], hours,
                                    "activation_energy", forecast)
    activation_price = hourly_to_quarters(mapped["activation_price"], hours, "price", forecast)
    pools = premium_pools(premium_history, delivery_day, snapshot, forecast)
    result = {"day_ahead_price": day_ahead,
              "activation_volume": activation,
              "activation_share": activation / np.repeat(procured_mw, forecast["protocol"]["quarters_per_hour"]),
              "activation_price": activation_price,
              "shortfall_price": sample_imbalance_prices(day_ahead, activation_hourly, pools, seed, forecast),
              "activation_representation": archive["activation_representation"]}
    if "capacity_price" in mapped:
        result["capacity_price"] = _resize_hours(mapped["capacity_price"], hours,
                                                  forecast["protocol"]["dst_policy"])
    return result


def required_activation_mwh(activation_share: np.ndarray, reserve_up_mw: np.ndarray,
                            representation: str, forecast: dict) -> np.ndarray:
    """Map reserve MW to pathwise or deterministic expected activation MWh."""
    quarters = int(forecast["protocol"]["quarters_per_hour"])
    reserve = np.repeat(np.asarray(reserve_up_mw, dtype=float), quarters)
    shares = np.asarray(activation_share, dtype=float)
    _require(shares.ndim == 2 and shares.shape[1] == len(reserve), "Activation-share shape differs")
    _require(np.isfinite(shares).all() and (shares >= 0).all(), "Activation shares are invalid")
    _require(np.isfinite(reserve).all() and (reserve >= 0).all(), "Reserve commitment is invalid")
    if representation == "path":
        return shares * reserve
    _require(representation == "expected_value", f"Unknown activation representation: {representation}")
    _require(np.array_equal(shares, np.repeat(shares[:1], len(shares), axis=0), equal_nan=True),
             "Expected-value activation varies across scenarios")
    return shares[0] * reserve


def validate_activation_prices(activation_price: np.ndarray, required: np.ndarray) -> None:
    """Assert activation-price NaNs occur only where required activation equals zero."""
    prices = np.asarray(activation_price, dtype=float)
    requirement = np.asarray(required, dtype=float)
    if requirement.ndim == 1:
        requirement = np.repeat(requirement[None, :], prices.shape[0], axis=0)
    _require(prices.shape == requirement.shape, "Activation-price and required shapes differ")
    _require(np.isfinite(requirement).all() and (requirement >= 0).all(), "Required activation is invalid")
    _require(not np.isinf(prices).any(), "Activation price is infinite")
    _require(not (np.isnan(prices) & (requirement > 0)).any(),
             "Activation price is missing where required activation is positive")


def _file_audit(loaded: dict, root: Path) -> list:
    rows = []
    for item in loaded.values():
        values = item["values"]
        selected = [item["delivery_days"].index(day) for day in item["evaluated_days"]]
        evaluated = values[selected]
        rows.append({"scope": "scenario_file", "path": str(item["path"].relative_to(root)),
                     "delivery_day": "", "period": "", "activation_state": "",
                     "information_set": item["information_set"],
                     "activation_representation": item["activation_representation"],
                     "delivery_days": len(item["delivery_days"]),
                     "evaluated_days": len(item["evaluated_days"]), "scenarios": values.shape[1],
                     "valid_n": int(np.isfinite(evaluated).sum()),
                     "missing_n": int(np.isnan(evaluated).sum()), "availability_ok": True})
    return rows


def _pool_audit(history: pd.DataFrame, days: list, forecast: dict) -> list:
    rows, boundaries = [], forecast["features"]["activation_period_boundaries"]
    for day in days:
        blocks = window_days(day, forecast)
        selected = history.loc[history["delivery_day"].isin(blocks["fit"])]
        decision = decision_time_utc(day, "gate_0730", forecast)
        for period in range(len(boundaries) - 1):
            for state in (False, True):
                cells = selected.loc[selected["period"].eq(period)
                                     & selected["activation_positive"].eq(state)]
                valid = cells["premium"].notna()
                available = cells.loc[valid, "available_at"].le(decision).all()
                _require(valid.sum() > 0 and available, f"{day} period {period} state {state}: bad pool")
                rows.append({"scope": "premium_pool", "path": "", "delivery_day": day,
                             "period": period, "activation_state": int(state),
                             "information_set": "gate_0730", "activation_representation": "",
                             "delivery_days": len(blocks["fit"]), "evaluated_days": "", "scenarios": "",
                             "valid_n": int(valid.sum()), "missing_n": int((~valid).sum()),
                             "availability_ok": bool(available)})
    return rows


def _interface_checks(loaded: dict, history: pd.DataFrame, days: list,
                      config: dict, forecast: dict) -> None:
    availability = config["premium_availability"]
    local = history["available_at"].dt.tz_convert(availability["civil_timezone"])
    local_dates = local.dt.strftime("%Y-%m-%d")
    _require(history.groupby("delivery_day")["available_at"].nunique().eq(1).all(),
             "One delivery day has multiple premium publication timestamps")
    _require(local.dt.strftime("%H:%M").eq(availability["publication_civil_time"]).all(),
             "Premium publication civil time changed")
    _require(local.dt.weekday.isin(availability["working_weekdays"]).all(),
             "Premium publication falls on a weekend")
    _require(not local_dates.isin(availability["holidays"]).any(),
             "Premium publication falls on a configured bank holiday")
    archive = next(item for item in loaded.values() if item["information_set"] == "gate_0730"
                   and item["targets"] == TARGETS_0730)
    day, hours = days[0], int(forecast["protocol"]["periods_per_day"])
    procurement = np.ones(hours)
    seed = int(forecast["protocol"]["sampling_seeds"][0])
    first = prepare_day_scenarios(archive, day, procurement, history, forecast, "gate_0730", seed)
    second = prepare_day_scenarios(archive, day, procurement, history, forecast, "gate_0730", seed)
    _require(np.array_equal(first["shortfall_price"], second["shortfall_price"], equal_nan=True),
             "Seeded premium sampling is not reproducible")
    source = archive["values"][archive["delivery_days"].index(day)]
    volume = source[:, archive["targets"].index("activation_volume")]
    quarters = int(forecast["protocol"]["quarters_per_hour"])
    _require(np.allclose(first["activation_volume"].reshape(len(volume), hours, quarters).sum(2), volume),
             "Hourly activation energy changed during quarter split")
    expected = np.repeat(first["activation_share"][:1], len(first["activation_share"]), axis=0)
    deterministic = required_activation_mwh(expected, np.ones(hours), "expected_value", forecast)
    _require(deterministic.ndim == 1, "Expected-value activation generated paths")
    validate_activation_prices(first["activation_price"],
                               required_activation_mwh(first["activation_share"], np.ones(hours),
                                                       "path", forecast))
    missing_price = first["activation_price"].copy()
    zero_required = np.zeros_like(first["activation_share"])
    missing_price[0, 0] = np.nan
    validate_activation_prices(missing_price, zero_required)
    base = np.arange(hours, dtype=float)[None, :]
    _require(hourly_to_quarters(base, hours - 1, "price", forecast).shape[1] ==
             (hours - 1) * quarters, "Spring DST truncation failed")
    expanded = hourly_to_quarters(base, hours + 1, "price", forecast)
    _require(np.array_equal(expanded[0, -quarters:], np.repeat(base[0, -1], quarters)),
             "Autumn DST last-hour repetition failed")


def write_audit(repository_root: Path, comparison_config_path: Path) -> pd.DataFrame:
    """Run the complete scenario-I/O self-check and write its deterministic audit table."""
    config = yaml.safe_load(comparison_config_path.read_text())
    forecast = yaml.safe_load((repository_root / config["base_config"]).read_text())
    storage = yaml.safe_load((repository_root / config["storage_config"]).read_text())
    days = test_delivery_days(forecast, repository_root)
    _require(len(days) == storage["acceptance"]["settled_test_days"], "S2/S3 test counts differ")
    loaded = load_common_scenarios(repository_root, config, forecast, days)
    history = load_premium_history(repository_root, config, forecast, storage)
    _interface_checks(loaded, history, days, config, forecast)
    rows = _file_audit(loaded, repository_root) + _pool_audit(history, days, forecast)
    audit = pd.DataFrame(rows).sort_values(
        ["scope", "path", "delivery_day", "period", "activation_state"]).reset_index(drop=True)
    output = repository_root / config["output"]["directory"] / config["output"]["scenario_io_audit_csv"]
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    return audit


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    audit = write_audit(root, root / arguments.config)
    print(f"scenario_io checks passed: rows={len(audit)}, valid_n={int(audit['valid_n'].sum())}")


if __name__ == "__main__":
    main()
