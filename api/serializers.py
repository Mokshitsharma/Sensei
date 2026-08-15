"""Recursively converts numpy/pandas types into plain JSON-safe Python
values. The ML pipeline returns numpy scalars and DataFrames everywhere;
FastAPI's default encoder doesn't reliably handle those."""

import math

import numpy as np
import pandas as pd


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def df_to_records(df: pd.DataFrame) -> list:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = out["date"].astype(str)
    return to_jsonable(out.to_dict(orient="records"))
