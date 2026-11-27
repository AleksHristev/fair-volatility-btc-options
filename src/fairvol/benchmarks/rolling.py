from __future__ import annotations

import math

import numba as nb
import numpy as np

from fairvol.filter.kalman import MAXM, _basis


@nb.njit(cache=True)
def _n_distinct(v: np.ndarray) -> int:
    if v.shape[0] == 0:
        return 0
    w = np.sort(v)
    c = 1
    for i in range(1, w.shape[0]):
        if w[i] != w[i - 1]:
            c += 1
    return c


@nb.njit(cache=True)
def window_fits(in_t, in_k, in_T, in_y, in_strike, exp_start, exp_end, q_t, q_exp, lookback_ms,
                min_trades, min_strikes, atm_floor, x_clamp, knot):
    nq = q_t.shape[0]
    quad = np.full((nq, 3), np.nan)
    krange = np.full((nq, 2), np.nan)
    quad_ok = np.zeros(nq, np.bool_)
    bcoef = np.full((3, nq, MAXM), np.nan)
    xrange = np.full((3, nq, 2), np.nan)
    b_ok = np.zeros((3, nq), np.bool_)
    atm_out = np.full(nq, np.nan)
    n_in = np.zeros(nq, np.int64)
    b = np.zeros(MAXM)
    for qi in range(nq):
        e = q_exp[qi]
        a0, b0 = exp_start[e], exp_end[e]
        if b0 <= a0:
            continue
        lo = a0 + np.searchsorted(in_t[a0:b0], q_t[qi] - lookback_ms, side="left")
        hi = a0 + np.searchsorted(in_t[a0:b0], q_t[qi], side="left")
        n = hi - lo
        n_in[qi] = n
        if n < min_trades:
            continue
        nd = _n_distinct(in_strike[lo:hi])
        if nd < min_strikes:
            continue
        k = in_k[lo:hi]
        y = in_y[lo:hi]
        X = np.empty((n, 3))
        for i in range(n):
            X[i, 0] = 1.0
            X[i, 1] = k[i]
            X[i, 2] = k[i] * k[i]
        XtX = X.T @ X
        if abs(np.linalg.det(XtX)) <= 1e-14 * max(1.0, np.abs(XtX).max()) ** 3:
            continue
        c = np.linalg.solve(XtX, X.T @ y)
        quad[qi, :] = c
        krange[qi, 0] = k.min()
        krange[qi, 1] = k.max()
        quad_ok[qi] = True
        atm = max(c[0], atm_floor) / 100.0
        atm_out[qi] = atm * 100.0
        for bid in range(3):
            m = 5 if bid == 2 else 3
            if nd < max(min_strikes, m):
                continue
            Z = np.empty((n, m))
            xmin = 1e300
            xmax = -1e300
            for i in range(n):
                x = k[i] / (atm * math.sqrt(in_T[lo + i]))
                x = min(max(x, -x_clamp), x_clamp)
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                _basis(x, bid, knot, b)
                for cc in range(m):
                    Z[i, cc] = b[cc]
            sol = np.linalg.lstsq(Z, y, rcond=1e-10)[0]
            for cc in range(m):
                bcoef[bid, qi, cc] = sol[cc]
            xrange[bid, qi, 0] = xmin
            xrange[bid, qi, 1] = xmax
            b_ok[bid, qi] = True
    return quad, krange, quad_ok, bcoef, xrange, b_ok, atm_out, n_in


@nb.njit(cache=True)
def eval_quad(coef: np.ndarray, krange: np.ndarray, ok: np.ndarray, fit_idx: np.ndarray, k: np.ndarray) -> np.ndarray:
    out = np.full(k.shape[0], np.nan)
    for i in range(k.shape[0]):
        f = fit_idx[i]
        if f < 0 or not ok[f] or not np.isfinite(k[i]):
            continue
        kk = min(max(k[i], krange[f, 0]), krange[f, 1])
        out[i] = coef[f, 0] + coef[f, 1] * kk + coef[f, 2] * kk * kk
    return out


@nb.njit(cache=True)
def eval_basis(bid: int, coef: np.ndarray, xrange: np.ndarray, ok: np.ndarray, atm: np.ndarray,
               fit_idx: np.ndarray, k: np.ndarray, T: np.ndarray, x_clamp: float, knot: float) -> np.ndarray:
    out = np.full(k.shape[0], np.nan)
    b = np.zeros(MAXM)
    m = 5 if bid == 2 else 3
    for i in range(k.shape[0]):
        f = fit_idx[i]
        if f < 0 or not ok[bid, f] or not np.isfinite(k[i]):
            continue
        x = k[i] / (atm[f] / 100.0 * math.sqrt(T[i]))
        x = min(max(x, xrange[bid, f, 0]), xrange[bid, f, 1])
        x = min(max(x, -x_clamp), x_clamp)
        _basis(x, bid, knot, b)
        v = 0.0
        for c in range(m):
            v += coef[bid, f, c] * b[c]
        out[i] = v
    return out
