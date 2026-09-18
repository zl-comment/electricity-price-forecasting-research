"""Risk calibration, joint VaR/CVaR scores, and paired decision comparisons."""

import numpy as np
import pandas as pd
from scipy import stats

from s3_forecast_side.scoring import significance_metrics


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def empirical_tail_metrics(profits_k_eur: np.ndarray, alpha: float) -> dict:
    """Return the equal-weight lower-tail VaR and fractional-mass CVaR."""
    values = np.sort(np.asarray(profits_k_eur, dtype=float))
    _require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
             "Profits must be a finite non-empty vector")
    _require(0.0 < alpha < 1.0, "Lower-tail probability lies outside (0, 1)")
    mass = alpha * len(values)
    nearest = round(mass)
    if np.isclose(mass, nearest, rtol=0.0, atol=np.finfo(float).eps * len(values)):
        mass = float(nearest)
    full = int(np.floor(mass))
    fraction = mass - full
    index = int(np.ceil(mass)) - 1
    tail_sum = float(values[:full].sum())
    if fraction > 0.0:
        tail_sum += float(fraction * values[full])
    var = float(values[index])
    return {"var_k_eur": var, "cvar_k_eur": tail_sum / mass,
            "exceedance_rate": float(np.mean(values <= var))}


def joint_var_cvar_score(var_k_eur: float, cvar_k_eur: float,
                         realised_k_eur: float, alpha: float) -> float:
    """Evaluate the referenced lower-tail profit equation in kEUR."""
    _require(0.0 < alpha < 1.0, "Lower-tail probability lies outside (0, 1)")
    indicator = float(realised_k_eur <= var_k_eur)
    logistic = float(1.0 / (1.0 + np.exp(-cvar_k_eur)))
    first = (indicator - alpha) * (var_k_eur - realised_k_eur)
    second = logistic * (indicator * (var_k_eur - realised_k_eur) / alpha
                         + cvar_k_eur - var_k_eur)
    return float(first + second - np.logaddexp(0.0, cvar_k_eur))


def calibration_row(delivery_day: str, strategy: str, chi: float,
                    scenario_profits_eur: np.ndarray, realised_profit_eur: float,
                    cvar_level: float) -> dict:
    """Summarise one predicted profit distribution against realised settlement."""
    alpha = 1.0 - float(cvar_level)
    tail = empirical_tail_metrics(np.asarray(scenario_profits_eur) / 1000.0, alpha)
    realised = float(realised_profit_eur) / 1000.0
    return {"delivery_day": delivery_day, "strategy": strategy, "chi": float(chi),
            **tail, "realised_profit_k_eur": realised,
            "below_var": int(realised < tail["var_k_eur"]),
            "realised_minus_cvar_k_eur": realised - tail["cvar_k_eur"],
            "joint_score": joint_var_cvar_score(
                tail["var_k_eur"], tail["cvar_k_eur"], realised, alpha)}


def aggregate_calibration(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily risk calibration without changing the daily denominator."""
    records = []
    for (strategy, chi), frame in rows.groupby(["strategy", "chi"], sort=True):
        exceed = frame.loc[frame["below_var"].eq(1), "realised_minus_cvar_k_eur"]
        records.append({"strategy": strategy, "chi": chi, "days": len(frame),
                        "var_exceedance_rate": float(frame["below_var"].mean()),
                        "mean_exceedance_realised_minus_cvar_k_eur": (
                            float(exceed.mean()) if len(exceed) else np.nan),
                        "mean_joint_score": float(frame["joint_score"].mean())})
    return pd.DataFrame(records)


def paired_metrics(left: np.ndarray, right: np.ndarray, config: dict) -> dict:
    """Return the frozen DM test and moving-block interval for left minus right."""
    settings = config["evaluation"]
    return significance_metrics(
        -np.asarray(left, dtype=float), -np.asarray(right, dtype=float),
        int(settings["dm_newey_west_lags"]), settings["dm_sidedness"],
        int(settings["bootstrap_block_days"]), int(settings["bootstrap_repetitions"]),
        float(settings["bootstrap_confidence_level"]), int(settings["bootstrap_seed"]))


def paired_row(daily: pd.DataFrame, strategy: str, comparator: str,
               metric: str, config: dict) -> dict:
    """Align two strategies by delivery day and compare a benefit metric."""
    left = daily.loc[daily["strategy"].eq(strategy), ["delivery_day", metric]]
    right = daily.loc[daily["strategy"].eq(comparator), ["delivery_day", metric]]
    joined = left.merge(right, on="delivery_day", suffixes=("_left", "_right"),
                        validate="one_to_one")
    expected = min(len(left), len(right))
    _require(len(joined) == expected, f"{strategy}/{comparator}: paired days differ")
    result = paired_metrics(joined[f"{metric}_left"], joined[f"{metric}_right"], config)
    return {"strategy": strategy, "comparator": comparator, "metric": metric,
            "mean_difference_strategy_minus_comparator": -result["mean_loss_difference_a_minus_b"],
            "confidence_lower": -result["confidence_upper"],
            "confidence_upper": -result["confidence_lower"],
            "dm_statistic": -result["dm_statistic"], "p_value": result["p_value"],
            "n_days": result["n_valid_days"]}


def cross_score_dm(score_i: np.ndarray, score_self: np.ndarray, config: dict) -> dict:
    """Test the referenced scenario score: candidate score minus self score."""
    settings = config["evaluation"]
    first = np.asarray(score_i, dtype=float)
    second = np.asarray(score_self, dtype=float)
    _require(first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all(),
             "Cross-score vectors differ or contain missing values")
    difference = first - second
    n = len(difference)
    centered = difference - difference.mean()
    variance = float(centered @ centered / n)
    lags = int(settings["dm_newey_west_lags"])
    for lag in range(1, lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    _require(np.isfinite(variance) and variance > 0.0, "Cross-score DM variance is invalid")
    statistic = float(difference.mean() / np.sqrt(variance / n))
    return {"mean_score_difference_i_minus_self": float(difference.mean()),
            "dm_statistic": statistic, "p_value_two_sided": float(2.0 * stats.norm.sf(abs(statistic))),
            "n_days": n, "newey_west_lags": lags,
            "null": "candidate score minus self score >= 0"}
