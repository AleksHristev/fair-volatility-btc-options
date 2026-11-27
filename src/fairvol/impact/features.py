from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PER_YEAR = 525_600.0


def time_since_last(df: pd.DataFrame, key: str) -> np.ndarray:
    q = pd.DataFrame({key: df[key].to_numpy(), "t": df["timestamp"].to_numpy(np.int64),
                      "_row": np.arange(len(df))}).sort_values("t", kind="mergesort")
    s = pd.DataFrame({key: df[key].to_numpy(), "t": df["info_ms"].to_numpy(np.int64)})
    s["t_src"] = s["t"]
    s = s.sort_values("t", kind="mergesort")
    m = pd.merge_asof(q, s, on="t", by=key, direction="backward",
                      allow_exact_matches=False).sort_values("_row")
    return (df["timestamp"].to_numpy(float) - m["t_src"].to_numpy(float)) / 60_000.0


def signed_flow(df: pd.DataFrame, window_ms: int, by: str | None) -> np.ndarray:
    out = np.zeros(len(df))
    groups = [(None, df.index)] if by is None else df.groupby(by, sort=False).groups.items()
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for _, idx in groups:
        ii = pos.loc[idx].to_numpy()
        src_t = df["info_ms"].to_numpy(np.int64)[ii]
        val = (df["d"].to_numpy(float) * df["vega_notional_usd"].to_numpy(float))[ii]
        o = np.argsort(src_t, kind="mergesort")
        st, sv = src_t[o], np.concatenate([[0.0], np.cumsum(val[o])])
        qt = df["timestamp"].to_numpy(np.int64)[ii]
        hi = np.searchsorted(st, qt, side="left")
        lo = np.searchsorted(st, qt - window_ms, side="left")
        out[ii] = sv[hi] - sv[lo]
    return out


def index_minute_grid(df: pd.DataFrame) -> pd.Series:
    t = df["timestamp"].to_numpy(np.int64)
    o = np.argsort(t, kind="mergesort")
    minute = t[o] // 60_000
    last = pd.Series(df["index_price"].to_numpy(float)[o], index=minute)
    last = last[~last.index.duplicated(keep="last")]
    full = pd.RangeIndex(int(minute.min()), int(minute.max()) + 1)
    return last.reindex(full).ffill()


def index_features(df: pd.DataFrame, grid: pd.Series, windows_min: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    m = df["timestamp"].to_numpy(np.int64) // 60_000
    g0 = int(grid.index[0])
    G = grid.to_numpy(float)
    lr = np.concatenate([[np.nan], np.diff(np.log(G))])
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(lr ** 2))])
    S = df["index_price"].to_numpy(float)
    for w in windows_min:
        j = m - w - g0
        valid = j >= 0
        then = np.where(valid, G[np.clip(j, 0, len(G) - 1)], np.nan)
        out[f"index_ret_{w}m"] = np.log(S / then)
        a = np.clip(m - w - g0, 0, len(G)) + 1
        b = np.clip(m - g0, 0, len(G))
        ss = csum[b] - csum[np.minimum(a, b)]
        rv = np.sqrt(ss * MIN_PER_YEAR / max(w - 1, 1)) * 100.0
        out[f"index_rv_{w}m"] = np.where(valid, rv, np.nan)
    return out
