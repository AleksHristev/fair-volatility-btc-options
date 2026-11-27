from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from fairvol.benchmarks.rolling import eval_quad, window_fits
WINDOW_MIN = 60
MIN_TRADES = 10
SVI_MIN_STRIKES = 5
MARK_MIN_STRIKES = 3
SVI_MAX_NFEV = 200
MIN_MS = 60_000
TRADE_COLS = ["trade_id_int", "timestamp", "info_ms", "expiry_ms", "instrument_name", "k", "k_causal", "T", "iv", "iv_F", "mark_iv",
              "is_screen", "strike", "d"]


def mark_star(df: pd.DataFrame, s: dict[str, Any]) -> np.ndarray:
    lo, hi = (100.0 * float(b) for b in s["pricing"]["iv_bounds"])
    iv, ivF, M = (df[c].to_numpy(float) for c in ("iv", "iv_F", "mark_iv"))
    m = iv - (ivF - M)
    ok = np.isfinite(m) & (m >= lo) & (m <= hi) & np.isfinite(df["k_causal"].to_numpy(float)) & np.isfinite(df["T"].to_numpy(float))
    return np.where(ok, m, np.nan)


def _grouped(sel: np.ndarray, df: pd.DataFrame, y: np.ndarray, exp_u: np.ndarray):
    idx = np.flatnonzero(sel)
    ec = pd.Index(exp_u).get_indexer(df["expiry_ms"].to_numpy(np.int64)[idx])
    order = np.lexsort((df["info_ms"].to_numpy(np.int64)[idx], ec))
    idx, ec = idx[order], ec[order]
    start = np.searchsorted(ec, np.arange(len(exp_u)), side="left").astype(np.int64)
    end = np.searchsorted(ec, np.arange(len(exp_u)), side="right").astype(np.int64)
    return (df["info_ms"].to_numpy(np.int64)[idx], df["k"].to_numpy(float)[idx], df["T"].to_numpy(float)[idx],
            y[idx], df["strike"].to_numpy(float)[idx], start, end)


def _quad_at(G, exp_u, t, e_ms, k, s, min_strikes) -> np.ndarray:
    f = s["filter"]
    out = np.full(len(t), np.nan)
    qe = pd.Index(exp_u).get_indexer(e_ms)
    ok = (qe >= 0) & np.isfinite(k) & (t > 0)
    if not ok.any():
        return out
    u = pd.DataFrame({"t": t[ok], "e": qe[ok]})
    uu = u.drop_duplicates().reset_index(drop=True)
    quad, kr, qok, *_ = window_fits(*G, uu["t"].to_numpy(np.int64), uu["e"].to_numpy(np.int64), WINDOW_MIN * MIN_MS,
                                    MIN_TRADES, int(min_strikes), float(f["atm_vol_floor"]), float(f["x_clamp"]), float(f["wing_knot"]))
    fidx = pd.MultiIndex.from_frame(uu).get_indexer(pd.MultiIndex.from_frame(u))
    out[ok] = eval_quad(quad, kr, qok, fidx.astype(np.int64), k[ok])
    return out


def svi_fit(k: np.ndarray, w: np.ndarray):
    wmax = float(w.max())
    if not np.isfinite(wmax) or wmax <= 0:
        return None
    kmin, kmax = float(k.min()), float(k.max())
    lo = np.array([-wmax, 0.0, -0.999, kmin - 1.0, 1e-4])
    hi = np.array([wmax, 10.0, 0.999, kmax + 1.0, 5.0])
    x0 = np.array([float(w.min()), 0.1, 0.0, float(k[np.argmin(w)]), 0.1])
    x0 = np.minimum(np.maximum(x0, lo + 1e-9 * (hi - lo)), hi - 1e-9 * (hi - lo))

    def res(p):
        a, b, rho, m, sig = p
        z = k - m
        return a + b * (rho * z + np.sqrt(z * z + sig * sig)) - w
    try:
        r = least_squares(res, x0, bounds=(lo, hi), method="trf", max_nfev=SVI_MAX_NFEV)
    except (ValueError, np.linalg.LinAlgError):
        return None
    a, b, rho, m, sig = r.x
    if not r.success or a + b * sig * math.sqrt(max(1.0 - rho * rho, 0.0)) < 0:
        return None
    return r.x


def svi_predictions(G, exp_u, t, e_ms, k_now, T_now, rows: np.ndarray, log=None) -> tuple[np.ndarray, dict]:
    in_t, in_k, in_T, in_y, in_strike, start, end = G
    out = np.full(len(t), np.nan)
    qe = pd.Index(exp_u).get_indexer(e_ms)
    sel = rows & (qe >= 0) & np.isfinite(k_now) & np.isfinite(T_now) & (T_now > 0)
    u = pd.DataFrame({"t": t[sel], "e": qe[sel], "row": np.flatnonzero(sel)})
    stats = {"pairs": 0, "fitted": 0, "too_few": 0, "failed": 0}
    for (tt, e), g in u.groupby(["t", "e"], sort=False):
        stats["pairs"] += 1
        a0, b0 = start[e], end[e]
        lo_ = a0 + np.searchsorted(in_t[a0:b0], tt - WINDOW_MIN * MIN_MS, side="left")
        hi_ = a0 + np.searchsorted(in_t[a0:b0], tt, side="left")
        if hi_ - lo_ < MIN_TRADES or len(np.unique(in_strike[lo_:hi_])) < SVI_MIN_STRIKES:
            stats["too_few"] += 1
            continue
        kk = in_k[lo_:hi_]
        w = (in_y[lo_:hi_] / 100.0) ** 2 * in_T[lo_:hi_]
        p = svi_fit(kk, w)
        if p is None:
            stats["failed"] += 1
            continue
        stats["fitted"] += 1
        r = g["row"].to_numpy()
        kq = np.clip(k_now[r], kk.min(), kk.max())
        a, b, rho, m, sig = p
        z = kq - m
        wq = a + b * (rho * z + np.sqrt(z * z + sig * sig))
        out[r] = np.where(wq > 0, 100.0 * np.sqrt(np.maximum(wq, 0.0) / T_now[r]), np.nan)
        if log and stats["pairs"] % 50_000 == 0:
            log.info("svi: %s", stats)
    return out, stats


def compute(df: pd.DataFrame, q: pd.DataFrame, s: dict[str, Any], svi_rows: np.ndarray, log=None) -> tuple[pd.DataFrame, dict]:
    elig = df["is_screen"].to_numpy(bool) & df["iv_adj_valid"].to_numpy(bool) & np.isfinite(df["k"].to_numpy(float))
    exp_u = np.unique(df["expiry_ms"].to_numpy(np.int64))
    y_adj = df["iv_adj"].to_numpy(float)
    Gt = _grouped(elig, df, y_adj, exp_u)
    mstar = mark_star(df, s)
    Gm = _grouped(np.isfinite(mstar) & np.isfinite(df["k"].to_numpy(float)), df, mstar, exp_u)
    qt = q["t_ms"].to_numpy(np.int64)
    qe = q["expiry_ms"].to_numpy(np.int64)
    k_now = q["k_causal"].to_numpy(float)
    T_now = q["T"].to_numpy(float)
    codes, _ = pd.factorize(pd.concat([df["instrument_name"].astype(str), q["instrument_name"].astype(str)], ignore_index=True))
    di, qi = codes[:len(df)], codes[len(df):]
    ei = np.flatnonzero(elig)
    order = np.lexsort((df["info_ms"].to_numpy(np.int64)[ei], di[ei]))
    ei = ei[order]
    key_e = di[ei].astype(np.int64) * (1 << 42) + df["info_ms"].to_numpy(np.int64)[ei]
    key_q = qi.astype(np.int64) * (1 << 42) + qt
    pos = np.searchsorted(key_e, key_q, side="left") - 1
    okl = (pos >= 0) & (di[ei[np.clip(pos, 0, None)]] == qi)
    lr = np.where(okl, ei[np.clip(pos, 0, None)], -1)
    has = lr >= 0
    lrc = np.clip(lr, 0, None)
    t_l = np.where(has, df["info_ms"].to_numpy(np.int64)[lrc], 0)
    k_l = np.where(has, df["k_causal"].to_numpy(float)[lrc], np.nan)
    y_l = np.where(has, y_adj[lrc], np.nan)
    m_l = np.where(has, mstar[lrc], np.nan)
    out = pd.DataFrame(index=q.index)
    b2 = s["benchmarks2"]
    q_now = _quad_at(Gt, exp_u, qt, qe, k_now, s, int(b2["min_strikes_quadratic"]))
    q_last = _quad_at(Gt, exp_u, t_l, qe, k_l, s, int(b2["min_strikes_quadratic"]))
    out["chk_quad_now"] = q_now
    move = q_now - q_last
    out["pred_last_surface"] = np.where(has & np.isfinite(move), y_l + move, y_l)
    qm_now = _quad_at(Gm, exp_u, qt, qe, k_now, s, MARK_MIN_STRIKES)
    qm_last = _quad_at(Gm, exp_u, t_l, qe, k_l, s, MARK_MIN_STRIKES)
    mmove = qm_now - qm_last
    base_m = np.where(np.isfinite(m_l), m_l, q["pred_mark_last"].to_numpy(float))
    out["pred_rebuilt_mark"] = np.where(np.isfinite(m_l) & np.isfinite(mmove), m_l + mmove, base_m)
    svi, st = svi_predictions(Gt, exp_u, qt, qe, k_now, T_now, svi_rows, log)
    out["svi_fitted"] = np.isfinite(svi)
    out["pred_svi"] = np.where(np.isfinite(svi), svi, q["pred_roll_quad"].to_numpy(float))
    stats = {**{f"svi_{k}": v for k, v in st.items()}, "n_queries": len(q), "last_found": int(has.sum()),
             "surface_move_used": int((has & np.isfinite(move)).sum()), "mark_move_used": int((np.isfinite(m_l) & np.isfinite(mmove)).sum())}
    return out, stats
