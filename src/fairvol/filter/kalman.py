from __future__ import annotations

import math

import numba as nb
import numpy as np

MAXM = 5
DAY_MS = 86_400_000.0
P_QL, P_QS, P_QC, P_QW, P_ALPHA, P_S2E = 0, 1, 2, 3, 4, 5
P_S2B = 6
P_KAPPA, P_S2U, P_GAMMA, P_BETA, P_S2BLOCK, P_S2COMBO = 11, 12, 13, 14, 15, 16
N_PARAMS = 17
C_XCLAMP, C_ATMFLOOR, C_TAUREF, C_TAUFLOOR, C_SDLEVEL, C_SDOTHER, C_KNOT = 0, 1, 2, 3, 4, 5, 6
C_TICK_SMALL, C_TICK_LARGE, C_TICK_SWITCH, C_TNCAP, C_VFLOOR, C_FACTOR = 7, 8, 9, 10, 11, 12
N_CONSTS = 13
BASIS_IDS = {"Q": 0, "H": 1, "HW": 2}
CLS_SCREEN, CLS_BLOCK, CLS_COMBO = 0, 1, 2
OUT_FIELDS = ("has_pred", "m_prior", "v_prior", "r_obs", "nu", "m_post", "v_post", "v_target",
              "x", "seq_nu", "seq_s", "in_lik")


def n_state(basis_id: int) -> int:
    return 5 if basis_id == 2 else 3


@nb.njit(cache=True)
def _basis(x: float, basis_id: int, knot: float, out: np.ndarray) -> None:
    out[0] = 1.0
    out[1] = x
    if basis_id == 0:
        out[2] = x * x
    else:
        out[2] = math.sqrt(1.0 + x * x) - 1.0
    if basis_id == 2:
        out[3] = max(0.0, -x - knot)
        out[4] = max(0.0, x - knot)


@nb.njit(cache=True)
def _closed_int(a: float, b: float, alpha: float, tau_ref: float) -> float:
    if b <= a:
        return 0.0
    one = 1.0 - alpha
    if abs(one) < 1e-12:
        return tau_ref * math.log(b / a)
    return tau_ref ** alpha * a ** one * math.expm1(one * math.log(b / a)) / one


@nb.njit(cache=True)
def tau_integral(tau_hi: float, tau_lo: float, alpha: float, tau_ref: float, tau_floor: float) -> float:
    if tau_hi <= tau_lo:
        return 0.0
    rate_f = (tau_ref / tau_floor) ** alpha
    if tau_lo >= tau_floor:
        return _closed_int(tau_lo, tau_hi, alpha, tau_ref)
    if tau_hi <= tau_floor:
        return (tau_hi - tau_lo) * rate_f
    return _closed_int(tau_floor, tau_hi, alpha, tau_ref) + (tau_floor - tau_lo) * rate_f


@nb.njit(cache=True)
def _ncdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


@nb.njit(cache=True)
def tick_noise_at(iv_volpts: float, k: float, T: float, is_call: bool, consts: np.ndarray) -> float:
    sig = max(iv_volpts, consts[C_ATMFLOOR]) / 100.0
    st = sig * math.sqrt(T)
    d1 = (-k + 0.5 * st * st) / st
    d2 = d1 - st
    ek = math.exp(k)
    if is_call:
        price = _ncdf(d1) - ek * _ncdf(d2)
    else:
        price = ek * _ncdf(-d2) - _ncdf(-d1)
    vega = math.sqrt(T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) / 100.0
    tick = consts[C_TICK_LARGE] if price >= consts[C_TICK_SWITCH] else consts[C_TICK_SMALL]
    if vega <= 0.0:
        return consts[C_TNCAP]
    return min(tick / vega, consts[C_TNCAP])


@nb.njit(cache=True)
def _noise_var(cls: int, bucket: int, weighting: int, m_prior: float, k: float, T: float,
               is_call: bool, params: np.ndarray, consts: np.ndarray) -> float:
    if cls == CLS_BLOCK:
        return params[P_S2BLOCK]
    if cls == CLS_COMBO:
        return params[P_S2COMBO]
    if weighting == 0:
        return params[P_S2E]
    tn = tick_noise_at(m_prior, k, T, is_call, consts)
    return params[P_S2B + bucket] + tn * tn / 12.0


@nb.njit(cache=True)
def _chol_solve(A: np.ndarray, m: int, b: np.ndarray, Lw: np.ndarray, yv: np.ndarray, x: np.ndarray) -> bool:
    for i in range(m):
        for j in range(i + 1):
            acc = A[i, j]
            for k in range(j):
                acc -= Lw[i, k] * Lw[j, k]
            if i == j:
                if acc <= 0.0:
                    return False
                Lw[i, i] = math.sqrt(acc)
            else:
                Lw[i, j] = acc / Lw[j, j]
    for i in range(m):
        acc = b[i]
        for k in range(i):
            acc -= Lw[i, k] * yv[k]
        yv[i] = acc / Lw[i, i]
    for i in range(m - 1, -1, -1):
        acc = yv[i]
        for k in range(i + 1, m):
            acc -= Lw[k, i] * x[k]
        x[i] = acc / Lw[i, i]
    return True


@nb.njit(cache=True)
def run_filter(ev_t, ev_kind, ev_exp, ev_slot, ev_k, ev_T, ev_S, ev_y, ev_dhs, ev_upd, ev_cls,
               ev_bucket, ev_call, ev_dt_target, ev_bucket_target, exp_ms, slot_offset,
               params, basis_id, weighting, residual_on, spot_on, consts, burn_in, lik_start_ms,
               want_out):
    N = ev_t.shape[0]
    E = exp_ms.shape[0]
    m = 5 if basis_id == 2 else 3
    n_slots = slot_offset[E]
    theta = np.zeros((E, MAXM))
    P = np.zeros((E, MAXM, MAXM))
    init = np.zeros(E, np.bool_)
    t_last = np.zeros(E, np.int64)
    S_last = np.zeros(E)
    n_upd = np.zeros(E, np.int64)
    n_seen = np.zeros(E, np.int64)
    u = np.zeros(n_slots)
    v = np.zeros(n_slots)
    C = np.zeros((n_slots, MAXM))
    nout = N if want_out else 0
    out = np.full((12, nout), np.nan)
    q = np.zeros(MAXM)
    q[0] = params[P_QL]
    q[1] = params[P_QS]
    q[2] = params[P_QC]
    q[3] = params[P_QW]
    q[4] = params[P_QW]
    alpha = params[P_ALPHA]
    kappa = params[P_KAPPA]
    s2u = params[P_S2U]
    xcl = consts[C_XCLAMP]
    floor = consts[C_ATMFLOOR]
    knot = consts[C_KNOT]
    vfloor = consts[C_VFLOOR]
    loglik = 0.0
    n_lik = 0
    n_vclip = 0
    b = np.zeros(MAXM)
    g = np.zeros(MAXM)
    bt = np.zeros(MAXM)
    cs = np.zeros(MAXM)
    wv = np.zeros(MAXM)
    Lw = np.zeros((MAXM, MAXM))
    yv = np.zeros(MAXM)
    LOG2PI = math.log(2.0 * math.pi)

    i = 0
    while i < N:
        e = ev_exp[i]
        t = ev_t[i]
        kind = ev_kind[i]
        j_end = i + 1
        while j_end < N and ev_t[j_end] == t and ev_exp[j_end] == e and ev_kind[j_end] == kind:
            j_end += 1
        nb_ = j_end - i
        off = slot_offset[e]
        if residual_on:
            for jj in range(i, j_end):
                while off + n_seen[e] <= ev_slot[jj]:
                    s_new = off + n_seen[e]
                    u[s_new] = 0.0
                    v[s_new] = s2u
                    for c in range(MAXM):
                        C[s_new, c] = 0.0
                    n_seen[e] += 1
        own_init = -1
        if not init[e]:
            first_upd = -1
            if kind == 1:
                for jj in range(i, j_end):
                    if ev_upd[jj]:
                        first_upd = jj
                        break
            if first_upd < 0:
                i = j_end
                continue
            best = -1
            bestd = 1e300
            for e2 in range(E):
                if init[e2]:
                    dd = abs(float(exp_ms[e2] - exp_ms[e]))
                    if dd < bestd:
                        bestd = dd
                        best = e2
            for c in range(MAXM):
                theta[e, c] = 0.0
                for c2 in range(MAXM):
                    P[e, c, c2] = 0.0
            if best >= 0:
                theta[e, 0] = theta[best, 0]
            else:
                theta[e, 0] = ev_y[first_upd] - ev_dhs[first_upd]
                own_init = first_upd
            P[e, 0, 0] = consts[C_SDLEVEL] ** 2
            for c in range(1, m):
                P[e, c, c] = consts[C_SDOTHER] ** 2
            init[e] = True
            t_last[e] = t
            S_last[e] = ev_S[i]
        else:
            dt = (t - t_last[e]) / DAY_MS
            S_now = ev_S[i]
            if spot_on and S_now > 0.0 and S_last[e] > 0.0 and S_now != S_last[e]:
                r = math.log(S_now / S_last[e])
                atm = max(theta[e, 0], floor) / 100.0
                a = spot_on * params[P_GAMMA] * r / (atm * math.sqrt(ev_T[i]))
                theta[e, 0] += a * theta[e, 1] + params[P_BETA] * r
                for c in range(m):
                    P[e, 0, c] += a * P[e, 1, c]
                for c in range(m):
                    P[e, c, 0] += a * P[e, c, 1]
                if residual_on:
                    for s in range(off, off + n_seen[e]):
                        C[s, 0] += a * C[s, 1]
            if dt > 0.0:
                tau_hi = (exp_ms[e] - t_last[e]) / DAY_MS
                tau_lo = (exp_ms[e] - t) / DAY_MS
                integ = tau_integral(tau_hi, tau_lo, alpha, consts[C_TAUREF], consts[C_TAUFLOOR])
                for c in range(m):
                    P[e, c, c] += q[c] * integ
                if residual_on:
                    cc = math.exp(-kappa * dt)
                    add = s2u * (1.0 - cc * cc)
                    for s in range(off, off + n_seen[e]):
                        u[s] *= cc
                        v[s] = cc * cc * v[s] + add
                        for c in range(m):
                            C[s, c] *= cc
            t_last[e] = t
            S_last[e] = S_now

        atm = max(theta[e, 0], floor) / 100.0
        B = np.zeros((nb_, MAXM))
        for jj in range(i, j_end):
            row = jj - i
            x = ev_k[jj] / (atm * math.sqrt(ev_T[jj]))
            if x > xcl:
                x = xcl
            elif x < -xcl:
                x = -xcl
            _basis(x, basis_id, knot, b)
            for c in range(m):
                B[row, c] = b[c]
            if jj == own_init:
                continue
            s = ev_slot[jj]
            mp = 0.0
            for c in range(m):
                mp += b[c] * theta[e, c]
            vp = 0.0
            for c in range(m):
                pb = 0.0
                for c2 in range(m):
                    pb += P[e, c, c2] * b[c2]
                vp += b[c] * pb
            if residual_on:
                mp += u[s]
                cb = 0.0
                for c in range(m):
                    cb += C[s, c] * b[c]
                vp += 2.0 * cb + v[s]
            if want_out:
                R = _noise_var(ev_cls[jj], ev_bucket[jj], weighting, mp, ev_k[jj], ev_T[jj],
                               ev_call[jj], params, consts)
                out[0, jj] = 1.0
                out[1, jj] = mp
                out[2, jj] = vp
                out[3, jj] = R
                out[8, jj] = x
                if kind == 1:
                    out[4, jj] = ev_y[jj] - ev_dhs[jj] - mp
                else:
                    dtt = ev_dt_target[jj]
                    tau_now = (exp_ms[e] - t) / DAY_MS
                    integ = tau_integral(tau_now, max(tau_now - dtt, 1e-9), alpha, consts[C_TAUREF],
                                         consts[C_TAUFLOOR])
                    vt = 0.0
                    for c in range(m):
                        pb = 0.0
                        for c2 in range(m):
                            pb += P[e, c, c2] * b[c2]
                        vt += b[c] * pb + b[c] * b[c] * q[c] * integ
                    if residual_on:
                        cc = math.exp(-kappa * dtt)
                        cb = 0.0
                        for c in range(m):
                            cb += C[s, c] * b[c]
                        vt += 2.0 * cc * cb + cc * cc * v[s] + s2u * (1.0 - cc * cc)
                    Rt = _noise_var(CLS_SCREEN, ev_bucket_target[jj], weighting, mp, ev_k[jj], ev_T[jj],
                                    ev_call[jj], params, consts)
                    out[7, jj] = vt + Rt

        if kind == 0:
            i = j_end
            continue

        for jj in range(i, j_end):
            if not ev_upd[jj] or jj == own_init:
                continue
            row = jj - i
            s = ev_slot[jj]
            mp = 0.0
            for c in range(m):
                b[c] = B[row, c]
                mp += b[c] * theta[e, c]
            if residual_on:
                mp += u[s]
            for c in range(m):
                acc = 0.0
                for c2 in range(m):
                    acc += P[e, c, c2] * b[c2]
                if residual_on:
                    acc += C[s, c]
                g[c] = acc
            Sv = 0.0
            for c in range(m):
                Sv += b[c] * g[c]
            if residual_on:
                cb = 0.0
                for c in range(m):
                    cb += C[s, c] * b[c]
                Sv += cb + v[s]
            R = _noise_var(ev_cls[jj], ev_bucket[jj], weighting, mp, ev_k[jj], ev_T[jj], ev_call[jj],
                           params, consts)
            Sv = max(Sv, 0.0) + R
            nu = ev_y[jj] - ev_dhs[jj] - mp
            counted = (t >= lik_start_ms) and (n_upd[e] >= burn_in)
            if counted:
                loglik += -0.5 * (LOG2PI + math.log(Sv) + nu * nu / Sv)
                n_lik += 1
            if want_out:
                out[9, jj] = nu
                out[10, jj] = Sv
                out[11, jj] = 1.0 if counted else 0.0
            n_upd[e] += 1
            fac = nu / Sv
            if residual_on:
                for c in range(m):
                    bt[c] = b[c]
                if consts[C_FACTOR] > 0.5:
                    for c in range(m):
                        cs[c] = C[s, c]
                    if _chol_solve(P[e], m, cs, Lw, yv, wv):
                        for c in range(m):
                            bt[c] += wv[c]
                    else:
                        n_vclip += 1
                for s2 in range(off, off + n_seen[e]):
                    gk = 0.0
                    if s2 == s:
                        for c in range(m):
                            gk += C[s2, c] * b[c]
                        gk += v[s2]
                    else:
                        for c in range(m):
                            gk += C[s2, c] * bt[c]
                    u[s2] += gk * fac
                    for c in range(m):
                        C[s2, c] -= g[c] * gk / Sv
                    v[s2] -= gk * gk / Sv
                    if v[s2] < vfloor:
                        v[s2] = vfloor
                        n_vclip += 1
            for c in range(m):
                theta[e, c] += g[c] * fac
            for c in range(m):
                for c2 in range(m):
                    P[e, c, c2] -= g[c] * g[c2] / Sv
            for c in range(m):
                for c2 in range(c + 1, m):
                    avg = 0.5 * (P[e, c, c2] + P[e, c2, c])
                    P[e, c, c2] = avg
                    P[e, c2, c] = avg
            if residual_on and consts[C_FACTOR] > 0.5:
                for c in range(m):
                    cs[c] = C[s, c]
                if _chol_solve(P[e], m, cs, Lw, yv, wv):
                    dsch = v[s]
                    for c in range(m):
                        dsch -= cs[c] * wv[c]
                    if dsch < -1e-9 * max(1.0, v[s]):
                        n_vclip += 1
                else:
                    n_vclip += 1

        if want_out:
            for jj in range(i, j_end):
                if jj == own_init:
                    continue
                row = jj - i
                s = ev_slot[jj]
                mp = 0.0
                vp = 0.0
                for c in range(m):
                    mp += B[row, c] * theta[e, c]
                    pb = 0.0
                    for c2 in range(m):
                        pb += P[e, c, c2] * B[row, c2]
                    vp += B[row, c] * pb
                if residual_on:
                    mp += u[s]
                    cb = 0.0
                    for c in range(m):
                        cb += C[s, c] * B[row, c]
                    vp += 2.0 * cb + v[s]
                out[5, jj] = mp
                out[6, jj] = vp
        i = j_end
    return loglik, n_lik, n_vclip, out
