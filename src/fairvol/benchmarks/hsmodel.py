from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd


@nb.njit(cache=True)
def lad_2(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, max_iter: int, tol: float,
          eps: float) -> tuple[float, float, int]:
    n = y.shape[0]
    w = np.ones(n)
    c1 = 0.0
    c2 = 0.0
    it = 0
    for it in range(max_iter):
        a11 = 0.0
        a12 = 0.0
        a22 = 0.0
        b1 = 0.0
        b2 = 0.0
        for i in range(n):
            a11 += w[i] * x1[i] * x1[i]
            a12 += w[i] * x1[i] * x2[i]
            a22 += w[i] * x2[i] * x2[i]
            b1 += w[i] * x1[i] * y[i]
            b2 += w[i] * x2[i] * y[i]
        det = a11 * a22 - a12 * a12
        if det == 0.0:
            break
        n1 = (a22 * b1 - a12 * b2) / det
        n2 = (a11 * b2 - a12 * b1) / det
        done = it > 0 and abs(n1 - c1) <= tol * max(1.0, abs(n1)) and abs(n2 - c2) <= tol * max(1.0, abs(n2))
        c1 = n1
        c2 = n2
        if done:
            break
        for i in range(n):
            r = abs(y[i] - c1 * x1[i] - c2 * x2[i])
            w[i] = 1.0 / max(r, eps)
    return c1, c2, it + 1


def fit_cells(train: pd.DataFrame, cells: list[tuple[str, str]], min_obs: int, max_iter: int,
              tol: float, eps: float) -> pd.DataFrame:
    def fit(g: pd.DataFrame) -> tuple[float, float]:
        c1, c2, _ = lad_2(g["tick"].to_numpy(float), g["vega_btc_volpt"].to_numpy(float),
                          g["e_price"].to_numpy(float), max_iter, tol, eps)
        return c1, c2

    by_cell = {k: g for k, g in train.groupby(["expiry_bucket", "delta_bucket"], observed=True)}
    by_exp = {k: g for k, g in train.groupby("expiry_bucket", observed=True)}
    cache: dict = {}
    rows = []
    pooled = None
    for eb, db in cells:
        g = by_cell.get((eb, db))
        if g is not None and len(g) >= min_obs:
            key, level, n = ("cell", eb, db), "cell", len(g)
            if key not in cache:
                cache[key] = fit(g)
        else:
            ge = by_exp.get(eb)
            if ge is not None and len(ge) >= min_obs:
                key, level, n = ("expiry", eb), "expiry", len(ge)
                if key not in cache:
                    cache[key] = fit(ge)
            else:
                key, level, n = ("pooled",), "pooled", len(train)
                if pooled is None:
                    pooled = fit(train) if len(train) else (np.nan, np.nan)
                cache[key] = pooled
        c1, c2 = cache[key]
        rows.append({"expiry_bucket": eb, "delta_bucket": db, "c_tick": c1, "c_vega": c2,
                     "level": level, "n": n})
    return pd.DataFrame(rows)
