from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def estimate_halfspreads(df: pd.DataFrame, est_mask: np.ndarray, min_obs: int) -> pd.DataFrame:
    x = df.loc[est_mask].copy()
    x["sd"] = x["d"] * (x["iv"] - x["mark_iv"])
    x = x[np.isfinite(x["sd"])]
    pooled = float(x["sd"].median())
    cell = x.groupby(["expiry_bucket", "delta_bucket"], observed=True)["sd"].agg(["median", "size"])
    exp = x.groupby("expiry_bucket", observed=True)["sd"].agg(["median", "size"])
    rows = []
    for (eb, db) in df.groupby(["expiry_bucket", "delta_bucket"], observed=True).groups:
        if (eb, db) in cell.index and cell.loc[(eb, db), "size"] >= min_obs:
            rows.append((eb, db, cell.loc[(eb, db), "median"], "cell", int(cell.loc[(eb, db), "size"])))
        elif eb in exp.index and exp.loc[eb, "size"] >= min_obs:
            rows.append((eb, db, exp.loc[eb, "median"], "expiry", int(exp.loc[eb, "size"])))
        else:
            rows.append((eb, db, pooled, "pooled", int(len(x))))
    return pd.DataFrame(rows, columns=["expiry_bucket", "delta_bucket", "hs", "level", "n"])


def apply_halfspreads(df: pd.DataFrame, hs: pd.DataFrame) -> pd.Series:
    key = pd.MultiIndex.from_frame(df[["expiry_bucket", "delta_bucket"]].astype(str))
    tab = hs.assign(expiry_bucket=hs["expiry_bucket"].astype(str),
                    delta_bucket=hs["delta_bucket"].astype(str)).set_index(
        ["expiry_bucket", "delta_bucket"])["hs"]
    return pd.Series(tab.reindex(key).to_numpy(), index=df.index)


def side_adjust(iv: pd.Series, d: pd.Series, hs: pd.Series) -> pd.Series:
    return iv - d * hs


__all__: list[Any] = ["estimate_halfspreads", "apply_halfspreads", "side_adjust"]
