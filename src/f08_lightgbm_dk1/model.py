"""LightGBM models built on raw paper-era LEAR design matrices."""

import lightgbm as lgb
import numpy as np
import pandas as pd


def build_matrices(lear_class, frame: pd.DataFrame, forecast_day: pd.Timestamp,
                   calibration_window: int) -> tuple:
    """Call pinned LEAR's private builder and return its unscaled matrices."""
    df_train = frame.loc[:forecast_day - pd.Timedelta(hours=1)]
    df_train = df_train.iloc[-calibration_window * 24:]
    df_test = frame.loc[forecast_day - pd.Timedelta(weeks=2):]
    if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
        raise ValueError("LightGBM's LEAR train and test slices must start at hour zero")
    model = lear_class(calibration_window=calibration_window)
    return model._build_and_split_XYs(df_train=df_train, df_test=df_test, date_test=forecast_day)


def fit_predict_hourly(Xtrain: np.ndarray, Ytrain: np.ndarray, Xtest: np.ndarray,
                       params: dict, seed: int) -> np.ndarray:
    """Fit one deterministic unscaled LightGBM model per delivery hour."""
    model_params = dict(params)
    model_params["seed"] = seed
    values = []
    for hour in range(Ytrain.shape[1]):
        dataset = lgb.Dataset(Xtrain, label=Ytrain[:, hour], free_raw_data=False)
        booster = lgb.train(params=model_params, train_set=dataset)
        prediction = np.asarray(booster.predict(Xtest), dtype=float).reshape(-1)
        if prediction.size != 1 or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid LightGBM prediction for hour {hour}")
        values.append(float(prediction[0]))
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise ValueError("LightGBM returned a non-finite daily forecast")
    return result
