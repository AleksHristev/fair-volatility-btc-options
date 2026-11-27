from __future__ import annotations

from collections import Counter, deque

import numpy as np
import pandas as pd


def last_trade_predictor(df: pd.DataFrame, src_mask: np.ndarray, value_cols: list[str]
                         ) -> pd.DataFrame:
    q = df[["instrument_name", "timestamp"]].copy()
    q["_row"] = np.arange(len(df))
    q = q.sort_values("timestamp", kind="mergesort")
    src = df.loc[src_mask, ["instrument_name", "timestamp", *value_cols]].copy()
    src = src.rename(columns={c: f"last_{c}" for c in value_cols})
    src["last_ts"] = src["timestamp"]
    src = src.sort_values("timestamp", kind="mergesort")
    m = pd.merge_asof(q, src, on="timestamp", by="instrument_name", direction="backward",
                      allow_exact_matches=False)
    m = m.sort_values("_row")
    out = m[[f"last_{c}" for c in value_cols] + ["last_ts"]].reset_index(drop=True)
    out.index = df.index
    return out


def _sliding_strikes_and_range(src_t: np.ndarray, src_k: np.ndarray, src_strike: np.ndarray,
                               lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(lo)
    n_strikes = np.zeros(n, dtype=np.int64)
    kmin = np.full(n, np.nan)
    kmax = np.full(n, np.nan)
    cnt: Counter = Counter()
    dmin: deque = deque()
    dmax: deque = deque()
    a = b = 0
    for i in range(n):
        while b < hi[i]:
            cnt[src_strike[b]] += 1
            while dmin and src_k[dmin[-1]] >= src_k[b]:
                dmin.pop()
            dmin.append(b)
            while dmax and src_k[dmax[-1]] <= src_k[b]:
                dmax.pop()
            dmax.append(b)
            b += 1
        while a < lo[i]:
            c = cnt[src_strike[a]] - 1
            if c:
                cnt[src_strike[a]] = c
            else:
                del cnt[src_strike[a]]
            a += 1
        while dmin and dmin[0] < a:
            dmin.popleft()
        while dmax and dmax[0] < a:
            dmax.popleft()
        if b > a:
            n_strikes[i] = len(cnt)
            kmin[i] = src_k[dmin[0]]
            kmax[i] = src_k[dmax[0]]
    return n_strikes, kmin, kmax


def rolling_smile_predictor(df: pd.DataFrame, src_mask: np.ndarray, y_cols: list[str],
                            lookback_ms: int, min_trades: int, min_strikes: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in y_cols:
        out[f"smile_{c}"] = np.nan
    out["smile_n"] = 0
    out["smile_nstrikes"] = 0
    out["smile_kclamped"] = False
    for _, g in df.groupby("expiry_ms", sort=False):
        g = g.sort_values("timestamp", kind="mergesort")
        sm = src_mask[df.index.get_indexer(g.index)]
        src = g.loc[sm & np.isfinite(g["k"].to_numpy())]
        for c in y_cols:
            src = src[np.isfinite(src[c].to_numpy())]
        if len(src) == 0:
            continue
        st = src["timestamp"].to_numpy(np.int64)
        k = src["k"].to_numpy(float)
        qt = g["timestamp"].to_numpy(np.int64)
        lo = np.searchsorted(st, qt - lookback_ms, side="left")
        hi = np.searchsorted(st, qt, side="left")
        n = hi - lo
        nstr, kmin, kmax = _sliding_strikes_and_range(st, k, src["strike"].to_numpy(float), lo, hi)
        ok = (n >= min_trades) & (nstr >= min_strikes)
        powers = np.vstack([np.ones_like(k), k, k**2, k**3, k**4])
        P = np.concatenate([np.zeros((5, 1)), np.cumsum(powers, axis=1)], axis=1)
        M = P[:, hi] - P[:, lo]
        A = np.empty((len(qt), 3, 3))
        for i in range(3):
            for j in range(3):
                A[:, i, j] = M[i + j]
        kq = np.clip(g["k"].to_numpy(float), kmin, kmax)
        out.loc[g.index, "smile_kclamped"] = ok & (g["k"].to_numpy(float) != kq)
        out.loc[g.index, "smile_n"] = n
        out.loc[g.index, "smile_nstrikes"] = nstr
        for c in y_cols:
            y = src[c].to_numpy(float)
            yp = np.vstack([y, k * y, k * k * y])
            Q = np.concatenate([np.zeros((3, 1)), np.cumsum(yp, axis=1)], axis=1)
            bvec = (Q[:, hi] - Q[:, lo]).T
            pred = np.full(len(qt), np.nan)
            idx = np.where(ok)[0]
            if len(idx):
                A_i, bi = A[idx], bvec[idx]
                det = np.linalg.det(A_i)
                good = np.abs(det) > 1e-14 * np.maximum(1.0, np.abs(A_i).max(axis=(1, 2))) ** 3
                coef = np.full((len(idx), 3), np.nan)
                if good.any():
                    coef[good] = np.linalg.solve(A_i[good], bi[good][..., None])[..., 0]
                kk = kq[idx]
                pred[idx] = coef[:, 0] + coef[:, 1] * kk + coef[:, 2] * kk * kk
            out.loc[g.index, f"smile_{c}"] = pred
    return out
