"""R2-specific LightGBM fit/predict split for shared multi-step models."""

from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np


def _fit_one_hour(arguments: tuple):
    Xtrain, labels, model_params = arguments
    dataset = lgb.Dataset(Xtrain, label=labels, free_raw_data=False)
    return lgb.train(params=model_params, train_set=dataset)


def fit_hourly_models(Xtrain: np.ndarray, Ytrain: np.ndarray,
                      params: dict, seed: int, workers: int) -> list:
    """Fit one deterministic model per forecast horizon."""
    model_params = dict(params)
    model_params["seed"] = seed
    arguments = [(Xtrain, Ytrain[:, hour], model_params)
                 for hour in range(Ytrain.shape[1])]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_fit_one_hour, arguments))


def predict_hourly_models(boosters: list, Xtest: np.ndarray) -> np.ndarray:
    """Predict every forecast horizon from already fitted models."""
    values = []
    for hour, booster in enumerate(boosters):
        prediction = np.asarray(booster.predict(Xtest), dtype=float).reshape(-1)
        if prediction.size != 1 or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid LightGBM prediction for hour {hour}")
        values.append(float(prediction[0]))
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise ValueError("LightGBM returned a non-finite forecast")
    return result
