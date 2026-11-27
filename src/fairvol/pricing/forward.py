from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from fairvol.pricing import black76 as b76

MS_PER_DAY = 86_400_000.0


@dataclass(frozen=True)
class Convention:
    underlying: str
    divisor: str
    day_count: float
    rate: float

    @property
    def label(self) -> str:
        return (f"U={self.underlying},D={self.divisor},dc={self.day_count:g},"
                f"r={self.rate:g}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def year_frac(tau_ms: np.ndarray | pd.Series, day_count: float) -> np.ndarray:
    return np.asarray(tau_ms, dtype=float) / (day_count * MS_PER_DAY)


def model_iv(price: np.ndarray, U: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray,
             is_call: np.ndarray, conv: Convention, bounds: tuple[float, float] = (0.01, 10.0)
             ) -> np.ndarray:
    div = U if conv.divisor == "underlying" else S
    iv = b76.implied_vol(price, U, K, T, is_call, rate=conv.rate, divisor=div, bounds=bounds)
    return 100.0 * iv


def self_implied_forward(price: np.ndarray, iv_volpts: np.ndarray, S: np.ndarray, K: np.ndarray,
                         T: np.ndarray, is_call: np.ndarray, conv: Convention,
                         bracket_ratio: tuple[float, float] = (0.8, 1.25)) -> np.ndarray:
    div = None if conv.divisor == "underlying" else S
    return b76.implied_forward(price, K, T, iv_volpts / 100.0, is_call, rate=conv.rate,
                               divisor=div, bracket=(bracket_ratio[0] * S, bracket_ratio[1] * S))


def window_median(src_t: np.ndarray, src_v: np.ndarray, q_t: np.ndarray, lookback_ms: int,
                  lookahead_ms: int = 0, strict: bool = True,
                  exclude_self_idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(src_v)
    if exclude_self_idx is not None:
        pos = np.cumsum(keep) - 1
        ex = np.where((exclude_self_idx >= 0) & keep[np.clip(exclude_self_idx, 0, None)],
                      pos[np.clip(exclude_self_idx, 0, None)], -1)
    t, v = src_t[keep], src_v[keep]
    lo = np.searchsorted(t, q_t - lookback_ms, side="left")
    if lookahead_ms == 0 and strict:
        hi = np.searchsorted(t, q_t, side="left")
    else:
        hi = np.searchsorted(t, q_t + lookahead_ms, side="right")
    med = np.full(len(q_t), np.nan)
    cnt = np.zeros(len(q_t), dtype=np.int64)
    for i in range(len(q_t)):
        a, b = lo[i], hi[i]
        if b <= a:
            continue
        w = v[a:b]
        if exclude_self_idx is not None and a <= ex[i] < b:
            w = np.delete(w, ex[i] - a)
        if len(w):
            med[i] = np.median(w)
            cnt[i] = len(w)
    return med, cnt


def grouped_window_median(df: pd.DataFrame, value_col: str, group_col: str, time_col: str,
                          lookback_ms: int, lookahead_ms: int = 0, strict: bool = True,
                          leave_one_out: bool = False) -> tuple[pd.Series, pd.Series]:
    med = pd.Series(np.nan, index=df.index)
    cnt = pd.Series(0, index=df.index, dtype="int64")
    for _, g in df.groupby(group_col, sort=False):
        g = g.sort_values(time_col, kind="mergesort")
        t = g[time_col].to_numpy(np.int64)
        v = g[value_col].to_numpy(float)
        ex = np.arange(len(g)) if leave_one_out else None
        m, c = window_median(t, v, t, lookback_ms, lookahead_ms, strict, ex)
        med.loc[g.index] = m
        cnt.loc[g.index] = c
    return med, cnt


def pcp_pairs(df: pd.DataFrame, conv: Convention, max_gap_ms: int) -> pd.DataFrame:
    cols = ["timestamp", "expiry_ms", "strike", "price", "index_price", "tau_ms"]
    calls = df.loc[df["is_call"], cols].sort_values("timestamp", kind="mergesort")
    puts = df.loc[~df["is_call"], cols].sort_values("timestamp", kind="mergesort")
    puts = puts.rename(columns={"timestamp": "t_put", "price": "p_put", "index_price": "S_put",
                                "tau_ms": "tau_put"})
    puts["t_put_key"] = puts["t_put"]
    m = pd.merge_asof(calls, puts.drop(columns=["t_put"]), left_on="timestamp",
                      right_on="t_put_key", by=["expiry_ms", "strike"], direction="nearest",
                      tolerance=max_gap_ms)
    m = m.dropna(subset=["p_put"]).copy()
    later_is_call = m["timestamp"] >= m["t_put_key"]
    m["pair_ms"] = np.where(later_is_call, m["timestamp"], m["t_put_key"]).astype("int64")
    m["S"] = np.where(later_is_call, m["index_price"], m["S_put"])
    T = year_frac(np.where(later_is_call, m["tau_ms"], m["tau_put"]), conv.day_count)
    dfac = np.exp(-conv.rate * T)
    x = (m["price"].to_numpy() - m["p_put"].to_numpy()) / dfac
    if conv.divisor == "underlying":
        with np.errstate(divide="ignore", invalid="ignore"):
            F = m["strike"].to_numpy() / (1.0 - x)
    else:
        F = m["strike"].to_numpy() + m["S"].to_numpy() * x
    m["F_pcp"] = F
    m["ratio"] = F / m["S"].to_numpy()
    ok = np.isfinite(m["ratio"]) & (m["ratio"] > 0.8) & (m["ratio"] < 1.25)
    return m.loc[ok, ["expiry_ms", "strike", "pair_ms", "S", "F_pcp", "ratio"]].reset_index(drop=True)


def pcp_ratio_estimate(df: pd.DataFrame, pairs: pd.DataFrame, lookback_ms: int
                       ) -> tuple[pd.Series, pd.Series]:
    med = pd.Series(np.nan, index=df.index)
    cnt = pd.Series(0, index=df.index, dtype="int64")
    pg = {k: g.sort_values("pair_ms", kind="mergesort") for k, g in pairs.groupby("expiry_ms")}
    for k, g in df.groupby("expiry_ms", sort=False):
        if k not in pg:
            continue
        p = pg[k]
        m, c = window_median(p["pair_ms"].to_numpy(np.int64), p["ratio"].to_numpy(float),
                             g["timestamp"].to_numpy(np.int64), lookback_ms)
        med.loc[g.index] = m
        cnt.loc[g.index] = c
    return med, cnt
