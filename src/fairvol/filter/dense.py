from __future__ import annotations

import math

import numba as nb
import numpy as np

from fairvol.filter.kalman import (C_ATMFLOOR, C_KNOT, C_SDLEVEL, C_SDOTHER, C_TAUFLOOR, C_TAUREF,
                                   C_XCLAMP, CLS_SCREEN, MAXM, P_ALPHA, P_BETA, P_GAMMA, P_KAPPA, P_QC, P_QL,
                                   P_QS, P_QW, P_S2U, _basis, _noise_var, tau_integral)

DAY_MS = 86_400_000.0


@nb.njit(cache=True)
def run_dense(ev_t, ev_kind, ev_exp, ev_slot, ev_k, ev_T, ev_S, ev_y, ev_dhs, ev_upd, ev_cls,
              ev_bucket, ev_call, exp_ms, slot_offset, params, basis_id, weighting, residual_on,
              spot_on, consts, burn_in, lik_start_ms):
    N = ev_t.shape[0]
    E = exp_ms.shape[0]
    m = 5 if basis_id == 2 else 3
    nmax = 0
    for e in range(E):
        nmax = max(nmax, slot_offset[e + 1] - slot_offset[e])
    D = m + (nmax if residual_on else 0)
    z = np.zeros((E, D))
    P = np.zeros((E, D, D))
    init = np.zeros(E, np.bool_)
    t_last = np.zeros(E, np.int64)
    S_last = np.zeros(E)
    n_upd = np.zeros(E, np.int64)
    q = np.array([params[P_QL], params[P_QS], params[P_QC], params[P_QW], params[P_QW]])
    alpha = params[P_ALPHA]
    kappa = params[P_KAPPA]
    s2u = params[P_S2U]
    m_prior = np.full(N, np.nan)
    v_prior = np.full(N, np.nan)
    m_post = np.full(N, np.nan)
    v_post = np.full(N, np.nan)
    loglik = 0.0
    n_lik = 0
    b = np.zeros(MAXM)
    LOG2PI = math.log(2.0 * math.pi)
    i = 0
    while i < N:
        e = ev_exp[i]
        t = ev_t[i]
        kind = ev_kind[i]
        j_end = i + 1
        while j_end < N and ev_t[j_end] == t and ev_exp[j_end] == e and ev_kind[j_end] == kind:
            j_end += 1
        ne = slot_offset[e + 1] - slot_offset[e]
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
            z[e, :] = 0.0
            P[e, :, :] = 0.0
            if best >= 0:
                z[e, 0] = z[best, 0]
            else:
                z[e, 0] = ev_y[first_upd] - ev_dhs[first_upd]
                own_init = first_upd
            P[e, 0, 0] = consts[C_SDLEVEL] ** 2
            for c in range(1, m):
                P[e, c, c] = consts[C_SDOTHER] ** 2
            if residual_on:
                for s in range(ne):
                    P[e, m + s, m + s] = s2u
            init[e] = True
            t_last[e] = t
            S_last[e] = ev_S[i]
        else:
            dt = (t - t_last[e]) / DAY_MS
            S_now = ev_S[i]
            if spot_on and S_now > 0.0 and S_last[e] > 0.0 and S_now != S_last[e]:
                r = math.log(S_now / S_last[e])
                atm = max(z[e, 0], consts[C_ATMFLOOR]) / 100.0
                a = params[P_GAMMA] * r / (atm * math.sqrt(ev_T[i]))
                z[e, 0] += a * z[e, 1] + params[P_BETA] * r
                for c in range(D):
                    P[e, 0, c] += a * P[e, 1, c]
                for c in range(D):
                    P[e, c, 0] += a * P[e, c, 1]
            if dt > 0.0:
                tau_hi = (exp_ms[e] - t_last[e]) / DAY_MS
                tau_lo = (exp_ms[e] - t) / DAY_MS
                integ = tau_integral(tau_hi, tau_lo, alpha, consts[C_TAUREF], consts[C_TAUFLOOR])
                for c in range(m):
                    P[e, c, c] += q[c] * integ
                if residual_on:
                    cc = math.exp(-kappa * dt)
                    for s in range(ne):
                        z[e, m + s] *= cc
                    for r_ in range(D):
                        for s in range(ne):
                            P[e, r_, m + s] *= cc
                            P[e, m + s, r_] *= cc
                    for s in range(ne):
                        P[e, m + s, m + s] += s2u * (1.0 - cc * cc)
            t_last[e] = t
            S_last[e] = S_now
        atm = max(z[e, 0], consts[C_ATMFLOOR]) / 100.0
        nb_ = j_end - i
        H = np.zeros((nb_, D))
        for jj in range(i, j_end):
            row = jj - i
            x = ev_k[jj] / (atm * math.sqrt(ev_T[jj]))
            x = min(max(x, -consts[C_XCLAMP]), consts[C_XCLAMP])
            _basis(x, basis_id, consts[C_KNOT], b)
            for c in range(m):
                H[row, c] = b[c]
            if residual_on:
                H[row, m + ev_slot[jj] - slot_offset[e]] = 1.0
            if jj == own_init:
                continue
            h = H[row]
            m_prior[jj] = h @ z[e]
            v_prior[jj] = h @ (P[e] @ h)
        if kind == 0:
            i = j_end
            continue
        for jj in range(i, j_end):
            if not ev_upd[jj] or jj == own_init:
                continue
            h = H[jj - i]
            mp = h @ z[e]
            Ph = P[e] @ h
            R = _noise_var(ev_cls[jj], ev_bucket[jj], weighting, mp, ev_k[jj], ev_T[jj], ev_call[jj],
                           params, consts)
            Sv = h @ Ph + R
            nu = ev_y[jj] - ev_dhs[jj] - mp
            if t >= lik_start_ms and n_upd[e] >= burn_in:
                loglik += -0.5 * (LOG2PI + math.log(Sv) + nu * nu / Sv)
                n_lik += 1
            n_upd[e] += 1
            z[e] += Ph * (nu / Sv)
            P[e] -= np.outer(Ph, Ph) / Sv
            P[e] = 0.5 * (P[e] + P[e].T)
        for jj in range(i, j_end):
            if jj == own_init:
                continue
            h = H[jj - i]
            m_post[jj] = h @ z[e]
            v_post[jj] = h @ (P[e] @ h)
        i = j_end
    return loglik, n_lik, m_prior, v_prior, m_post, v_post


@nb.njit(cache=True)
def run_dense_core(ev_t, ev_kind, ev_exp, ev_slot, ev_k, ev_T, ev_S, ev_y, ev_dhs, ev_upd, ev_cls,
                   ev_bucket, ev_call, ev_dt_target, ev_bucket_target, exp_ms, slot_offset,
                   params, basis_id, weighting, residual_on, spot_on, consts, burn_in, lik_start_ms,
                   want_out, ev_mult, ev_clk, exp_clk):
    N = ev_t.shape[0]
    E = exp_ms.shape[0]
    m = 5 if basis_id == 2 else 3
    dim = np.empty(E, np.int64)
    zoff = np.zeros(E + 1, np.int64)
    poff = np.zeros(E + 1, np.int64)
    dmax = m
    for e in range(E):
        d = m + ((slot_offset[e + 1] - slot_offset[e]) if residual_on else 0)
        dim[e] = d
        dmax = max(dmax, d)
        zoff[e + 1] = zoff[e] + d
        poff[e + 1] = poff[e] + d * d
    Z = np.zeros(zoff[E])
    PP = np.zeros(poff[E])
    init = np.zeros(E, np.bool_)
    c_last = np.zeros(E)
    S_last = np.zeros(E)
    n_upd = np.zeros(E, np.int64)
    n_seen = np.zeros(E, np.int64)
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
    loglik = 0.0
    n_lik = 0
    n_neg = 0
    b = np.zeros(MAXM)
    Ph = np.zeros(dmax)
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
        d = dim[e]
        z = Z[zoff[e]:zoff[e + 1]]
        P = PP[poff[e]:poff[e + 1]].reshape((d, d))
        if residual_on:
            for jj in range(i, j_end):
                while off + n_seen[e] <= ev_slot[jj]:
                    sl = m + n_seen[e]
                    z[sl] = 0.0
                    for r in range(d):
                        P[sl, r] = 0.0
                        P[r, sl] = 0.0
                    P[sl, sl] = s2u
                    n_seen[e] += 1
        A = m + (n_seen[e] if residual_on else 0)
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
            for c in range(m):
                z[c] = 0.0
            for r in range(d):
                for c in range(m):
                    P[r, c] = 0.0
                    P[c, r] = 0.0
            if best >= 0:
                z[0] = Z[zoff[best]]
            else:
                z[0] = ev_y[first_upd] - ev_dhs[first_upd]
                own_init = first_upd
            P[0, 0] = consts[C_SDLEVEL] ** 2
            for c in range(1, m):
                P[c, c] = consts[C_SDOTHER] ** 2
            init[e] = True
            c_last[e] = ev_clk[i]
            S_last[e] = ev_S[i]
        else:
            dt = (ev_clk[i] - c_last[e]) / DAY_MS
            S_now = ev_S[i]
            if spot_on and S_now > 0.0 and S_last[e] > 0.0 and S_now != S_last[e]:
                r = math.log(S_now / S_last[e])
                atm = max(z[0], floor) / 100.0
                a = params[P_GAMMA] * r / (atm * math.sqrt(ev_T[i]))
                z[0] += a * z[1] + params[P_BETA] * r
                for c in range(A):
                    P[0, c] += a * P[1, c]
                for c in range(A):
                    P[c, 0] += a * P[c, 1]
            if dt > 0.0:
                tau_hi = (exp_clk[e] - c_last[e]) / DAY_MS
                tau_lo = (exp_clk[e] - ev_clk[i]) / DAY_MS
                integ = tau_integral(tau_hi, tau_lo, alpha, consts[C_TAUREF], consts[C_TAUFLOOR])
                for c in range(m):
                    P[c, c] += q[c] * integ
                if residual_on and A > m:
                    cc = math.exp(-kappa * dt)
                    cc2 = cc * cc
                    add = s2u * (1.0 - cc2)
                    for sl in range(m, A):
                        z[sl] *= cc
                    for r in range(m):
                        for c in range(m, A):
                            P[r, c] *= cc
                            P[c, r] *= cc
                    for r in range(m, A):
                        for c in range(m, A):
                            P[r, c] *= cc2
                        P[r, r] += add
            c_last[e] = ev_clk[i]
            S_last[e] = S_now

        atm = max(z[0], floor) / 100.0
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
            sl = m + ev_slot[jj] - off
            mp = 0.0
            vb = 0.0
            for c in range(m):
                mp += b[c] * z[c]
                pb = 0.0
                for c2 in range(m):
                    pb += P[c, c2] * b[c2]
                vb += b[c] * pb
            vp = vb
            cb = 0.0
            if residual_on:
                mp += z[sl]
                for c in range(m):
                    cb += P[c, sl] * b[c]
                vp += 2.0 * cb + P[sl, sl]
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
                    tau_now = (exp_clk[e] - ev_clk[i]) / DAY_MS
                    integ = tau_integral(tau_now, max(tau_now - dtt, 1e-9), alpha, consts[C_TAUREF],
                                         consts[C_TAUFLOOR])
                    vt = vb
                    for c in range(m):
                        vt += b[c] * b[c] * q[c] * integ
                    if residual_on:
                        cc = math.exp(-kappa * dtt)
                        vt += 2.0 * cc * cb + cc * cc * P[sl, sl] + s2u * (1.0 - cc * cc)
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
            sl = m + ev_slot[jj] - off
            mp = 0.0
            for c in range(m):
                b[c] = B[row, c]
                mp += b[c] * z[c]
            if residual_on:
                mp += z[sl]
            for r in range(A):
                acc = 0.0
                for c in range(m):
                    acc += P[r, c] * b[c]
                if residual_on:
                    acc += P[r, sl]
                Ph[r] = acc
            Sv = 0.0
            for c in range(m):
                Sv += b[c] * Ph[c]
            if residual_on:
                Sv += Ph[sl]
            R = _noise_var(ev_cls[jj], ev_bucket[jj], weighting, mp, ev_k[jj], ev_T[jj], ev_call[jj],
                           params, consts) * ev_mult[jj]
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
            inv = 1.0 / Sv
            for r in range(A):
                z[r] += Ph[r] * fac
            neg = False
            for r in range(A):
                pr = Ph[r]
                for c in range(A):
                    P[r, c] -= (pr * Ph[c]) * inv
                if P[r, r] < -1e-6:
                    neg = True
            if neg:
                n_neg += 1

        if want_out:
            for jj in range(i, j_end):
                if jj == own_init:
                    continue
                row = jj - i
                sl = m + ev_slot[jj] - off
                mp = 0.0
                vp = 0.0
                for c in range(m):
                    mp += B[row, c] * z[c]
                    pb = 0.0
                    for c2 in range(m):
                        pb += P[c, c2] * B[row, c2]
                    vp += B[row, c] * pb
                if residual_on:
                    mp += z[sl]
                    cb = 0.0
                    for c in range(m):
                        cb += P[c, sl] * B[row, c]
                    vp += 2.0 * cb + P[sl, sl]
                out[5, jj] = mp
                out[6, jj] = vp
        i = j_end
    return loglik, n_lik, n_neg, out


@nb.njit(cache=True)
def run_dense_out(ev_t, ev_kind, ev_exp, ev_slot, ev_k, ev_T, ev_S, ev_y, ev_dhs, ev_upd, ev_cls,
                  ev_bucket, ev_call, ev_dt_target, ev_bucket_target, exp_ms, slot_offset,
                  params, basis_id, weighting, residual_on, spot_on, consts, burn_in, lik_start_ms,
                  want_out):
    N = ev_t.shape[0]
    ones = np.ones(N)
    clk = np.empty(N)
    for i in range(N):
        clk[i] = float(ev_t[i])
    eclk = np.empty(exp_ms.shape[0])
    for e in range(exp_ms.shape[0]):
        eclk[e] = float(exp_ms[e])
    return run_dense_core(ev_t, ev_kind, ev_exp, ev_slot, ev_k, ev_T, ev_S, ev_y, ev_dhs, ev_upd, ev_cls,
                          ev_bucket, ev_call, ev_dt_target, ev_bucket_target, exp_ms, slot_offset,
                          params, basis_id, weighting, residual_on, spot_on, consts, burn_in, lik_start_ms,
                          want_out, ones, clk, eclk)


def run_engine(engine: str, *args):
    if engine == "dense":
        return run_dense_out(*args)
    if engine == "factor":
        from fairvol.filter.kalman import run_filter
        return run_filter(*args)
    raise ValueError(engine)
