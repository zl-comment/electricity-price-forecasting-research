"""Daily rolling recalibration with the paper-era epftoolbox LEAR."""

import hashlib
import importlib.util
import inspect
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from sklearn.linear_model import Lasso, LassoLarsIC

from .data import EPFData, build_available_data


def load_paper_lear(settings: dict):
    import epftoolbox

    environment_fingerprint = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
    }
    expected_versions = {
        "scikit_learn": settings["scikit_learn_version"],
        "numpy": settings["numpy_version"],
        "pandas": settings["pandas_version"],
        "scipy": settings["scipy_version"],
        "statsmodels": settings["statsmodels_version"],
        "matplotlib": settings["matplotlib_version"],
    }
    if ".".join(platform.python_version_tuple()[:2]) != settings["python_major_minor"]:
        raise RuntimeError(f"Unexpected Python version: {platform.python_version()}")
    if any(environment_fingerprint[key] != value for key, value in expected_versions.items()):
        raise RuntimeError(f"Unexpected environment: {environment_fingerprint}")
    source = Path(epftoolbox.__file__).parent / "models" / "_lear.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != settings["lear_source_sha256"]:
        raise RuntimeError(f"Unexpected epftoolbox LEAR source: {digest}")
    lars = LassoLarsIC(
        criterion=settings["lear_criterion"], max_iter=settings["lear_max_iter"]
    ).get_params()
    lasso = Lasso(max_iter=settings["lear_max_iter"]).get_params()
    if lars["normalize"] is not True or lasso["normalize"] is not False:
        raise RuntimeError("Paper-era LASSO normalization defaults are unavailable")
    if "sigma2 = np.var(y)" not in inspect.getsource(LassoLarsIC.fit):
        raise RuntimeError("Paper-era LassoLarsIC AIC implementation is unavailable")

    spec = importlib.util.spec_from_file_location("epftoolbox_paper_lear", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LEAR source from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = {
        "epftoolbox_commit": settings["epftoolbox_commit"],
        "lear_source_sha256": digest,
        "scikit_learn_version": sklearn.__version__,
        "lars_normalize": lars["normalize"],
        "lasso_normalize": lasso["normalize"],
        "environment_fingerprint": environment_fingerprint,
    }
    return module.LEAR, runtime


def run_window(
    lear_class, data: EPFData, calibration_days: int, protocol: dict
) -> pd.Series:
    hours = protocol["hours_per_day"]
    if protocol["recalibration_days"] != 1:
        raise ValueError("This acceptance test requires daily recalibration")
    dates = data.test.index[::hours]
    forecast = pd.Series(index=data.test.index, dtype=float, name=str(calibration_days))
    model = lear_class(calibration_window=calibration_days)
    for forecast_date in dates:
        available = build_available_data(data, forecast_date, hours)
        values = model.recalibrate_and_forecast_next_day(
            df=available,
            next_day_date=forecast_date,
            calibration_window=calibration_days,
        )
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != hours or not np.isfinite(values).all():
            raise ValueError(f"Invalid forecast for {forecast_date}")
        forecast.loc[forecast_date : forecast_date + pd.Timedelta(hours=hours - 1)] = values
    if forecast.isna().any():
        raise ValueError(f"Incomplete forecast for window {calibration_days}")
    return forecast
