from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def expiry_bucket(tau_days: pd.Series | np.ndarray, settings: dict[str, Any]) -> pd.Categorical:
    cfg = settings["buckets"]["expiry_days"]
    edges = [float(e) for e in cfg["edges"]]
    return pd.cut(np.asarray(tau_days, float), bins=edges, labels=cfg["labels"], right=False)


def delta_bucket(abs_delta: pd.Series | np.ndarray, settings: dict[str, Any]) -> pd.Categorical:
    cfg = settings["buckets"]["abs_delta"]
    return pd.cut(np.asarray(abs_delta, float), bins=[float(e) for e in cfg["edges"]],
                  labels=cfg["labels"], right=False)


def quantile_bucket(x: pd.Series, n: int, edges: np.ndarray | None = None
                    ) -> tuple[pd.Categorical, np.ndarray]:
    if edges is None:
        edges = np.unique(np.nanquantile(x.to_numpy(float), np.linspace(0, 1, n + 1)[1:-1]))
    full = np.concatenate([[-np.inf], edges, [np.inf]])
    labels = [f"Q{i + 1}" for i in range(len(full) - 1)]
    return pd.cut(x.to_numpy(float), bins=full, labels=labels, right=True), edges
