"""Construct the full Table A.1 feature vocabulary for Finnish aFRR."""

import numpy as np
import pandas as pd


MARKET = ["Up", "Down", "sp", "Down_Cap", "Up_Cap", "electricity_consumption",
          "electricity_consumption_Finnish_networks", "electricity_consumption_forecast"]
WEATHER = ["cloud_amount", "wind_speed", "precipitation_amount", "pressure",
           "air_temperature", "relative_humidity", "wind_direction"]


def signed_log1p(values: pd.Series) -> pd.Series:
    return np.sign(values) * np.log1p(values.abs())


def market_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[MARKET + WEATHER].copy()
    for column in ("Down", "Up", "sp"):
        for lag in (12, 24, 36, 48):
            result[f"{column}_lag_{lag}"] = frame[column].shift(lag)
    for lag in (24, 48):
        result[f"electricity_consumption_lag_{lag}"] = frame["electricity_consumption"].shift(lag)
    for column in ("Down", "Up"):
        for window in (12, 24):
            rolling = frame[column].rolling(window)
            result[f"{column}_rolling_mean_{window}"] = rolling.mean()
            result[f"{column}_rolling_std_{window}"] = rolling.std()
    for window in (24, 48):
        result[f"sp_mean_{window}"] = frame["sp"].rolling(window).mean()
    for window in (12, 24):
        result[f"sp_std_{window}"] = frame["sp"].rolling(window).std()
    result["electricity_consumption_rolling_mean_24"] = (
        frame["electricity_consumption"].rolling(24).mean())
    result["electricity_consumption_Finnish_networks_rolling_mean_24"] = (
        frame["electricity_consumption_Finnish_networks"].rolling(24).mean())
    result["log_Down"] = signed_log1p(frame["Down"])
    result["log_Up"] = signed_log1p(frame["Up"])
    return result


def perceived_temperature(frame: pd.DataFrame) -> pd.DataFrame:
    temperature = frame["air_temperature"]
    humidity = frame["relative_humidity"]
    wind_kmh = frame["wind_speed"] * 3.6
    wind_chill = (13.12 + 0.6215 * temperature - 11.37 * wind_kmh.pow(0.16)
                  + 0.3965 * temperature * wind_kmh.pow(0.16))
    heat_index = (-8.784695 + 1.61139411 * temperature + 2.338549 * humidity
                  - 0.14611605 * temperature * humidity)
    alpha = np.log(humidity / 100.0) + 17.625 * temperature / (243.04 + temperature)
    return pd.DataFrame({
        "wind_chill": wind_chill.where((temperature <= 10) & (wind_kmh > 4.8), temperature),
        "heat_index": heat_index.where((temperature >= 27) & (humidity > 40), temperature),
        "dew_point": 243.04 * alpha / (17.625 - alpha),
    }, index=frame.index)


def weather_features(frame: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    temperature = frame["air_temperature"]
    for lag in (3, 6, 9):
        result[f"lag_temp_{lag}h"] = temperature.shift(lag)
    result["lag_pressure_6h"] = frame["pressure"].shift(6)
    for window in (6, 12):
        result[f"rolling_temp_{window}h"] = temperature.rolling(window).mean()
    result["rolling_humidity_6h"] = frame["relative_humidity"].rolling(6).mean()
    radians = np.deg2rad(frame["wind_direction"])
    result["wind_direction_sin"], result["wind_direction_cos"] = np.sin(radians), np.cos(radians)
    result = result.join(perceived_temperature(frame))
    result["wind_power"] = frame["wind_speed"].pow(2)
    result["storm_index"] = frame["wind_speed"] * frame["precipitation_amount"]
    result["humidity_x_wind"] = frame["relative_humidity"] * frame["wind_speed"]
    hour, weekday, month = frame.index.hour, frame.index.dayofweek, frame.index.month
    result["wind_speed_x_hour"] = frame["wind_speed"] * hour
    result["relative_humidity_x_air_temp"] = frame["relative_humidity"] * temperature
    result["wind_speed_x_dayofweek"] = frame["wind_speed"] * weekday
    result["precipitation_x_hour"] = frame["precipitation_amount"] * hour
    result["temp_x_hour"] = temperature * hour
    result["temp_x_dayofweek"] = temperature * weekday
    result["temp_x_month"] = temperature * month
    result["temp_diff_1h"], result["temp_diff_24h"] = temperature.diff(1), temperature.diff(24)
    result["pressure_change_6h"] = frame["pressure"].diff(6)
    result["pressure_change_12h"] = frame["pressure"].diff(12)
    result["log_wind_speed"] = np.log1p(frame["wind_speed"])
    result["sqrt_precipitation"] = np.sqrt(frame["precipitation_amount"])
    result["is_high_wind"] = (frame["wind_speed"] >= thresholds["high_wind_threshold_mps"]).astype(float)
    result["is_high_wind"] = result["is_high_wind"].where(frame["wind_speed"].notna())
    result["is_heavy_rain"] = (frame["precipitation_amount"] >= thresholds["heavy_rain_threshold_mm"]).astype(float)
    result["is_heavy_rain"] = result["is_heavy_rain"].where(frame["precipitation_amount"].notna())
    result["is_heat_wave"] = (temperature >= thresholds["heat_wave_threshold_c"]).astype(float)
    result["is_heat_wave"] = result["is_heat_wave"].where(temperature.notna())
    return result


def time_features(frame: pd.DataFrame) -> pd.DataFrame:
    index = frame.index
    week = index.isocalendar().week.to_numpy(dtype=float)
    result = pd.DataFrame({
        "hour": index.hour, "dayofweek": index.dayofweek, "month": index.month,
        "year": index.year, "dayofyear": index.dayofyear, "dayofmonth": index.day,
        "weekofyear": week, "quarter": index.quarter,
        "hour_sin": np.sin(2 * np.pi * index.hour / 24),
        "hour_cos": np.cos(2 * np.pi * index.hour / 24),
        "dayofweek_sin": np.sin(2 * np.pi * index.dayofweek / 7),
        "dayofweek_cos": np.cos(2 * np.pi * index.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * index.month / 12),
        "month_cos": np.cos(2 * np.pi * index.month / 12),
        "week_sin": np.sin(2 * np.pi * week / 52),
        "week_cos": np.cos(2 * np.pi * week / 52),
        "is_weekend": (index.dayofweek >= 5).astype(int),
        "is_summer": np.isin(index.month, [6, 7, 8]).astype(int),
        "is_winter": np.isin(index.month, [12, 1, 2]).astype(int),
        "is_public_holiday": frame["is_public_holiday"],
    }, index=index)
    return result


def build_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = market_features(frame).join(weather_features(frame, config)).join(time_features(frame))
    if result.columns.duplicated().any() or result.shape[1] != config["expected_count"]:
        raise ValueError(f"Unexpected Table A.1 feature count: {result.shape[1]}")
    values = result.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Table A.1 features contain infinite values")
    return result
