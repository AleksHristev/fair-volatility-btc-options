from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fairvol.filter.kalman import (C_ATMFLOOR, C_KNOT, C_TAUFLOOR, C_TAUREF, C_XCLAMP, MAXM, P_ALPHA, P_BETA, P_GAMMA,
                                   P_KAPPA, P_QC, P_QL, P_QS, P_QW, P_S2B, P_S2E, P_S2U, _basis,
                                   n_state, tau_integral, tick_noise_at)
from fairvol.filter.params import Spec
from fairvol.filter.events import bucket_index

DAY_MS = 86_400_000


def simulate(spec: Spec, nat: np.ndarray, consts: np.ndarray, cfg: dict[str, Any],
             settings: dict[str, Any], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = n_state(spec.basis_id)
    q = np.array([nat[P_QL], nat[P_QS], nat[P_QC], nat[P_QW], nat[P_QW]])[:m]
    t0 = int(cfg["t0_ms"])
    t_end = t0 + int(cfg["days"] * DAY_MS)
    exps = [t0 + int(tau * DAY_MS) for tau in cfg["expiry_tau0_days"]]
    nk = int(cfg["n_strikes"])
    xs = np.linspace(-cfg["x_span"], cfg["x_span"], nk)
    S0 = float(cfg["spot0"])
    atm0 = float(cfg["theta0"][0]) / 100.0
    inst = []
    for e, ems in enumerate(exps):
        T0 = (ems - t0) / DAY_MS / 365.0
        for j, x in enumerate(xs):
            K = S0 * math.exp(x * atm0 * math.sqrt(T0))
            for call in (True, False):
                inst.append((e, ems, K, call, f"SIM-{e}-{j}-{'C' if call else 'P'}", x))
    inst_df = pd.DataFrame(inst, columns=["e", "expiry_ms", "K", "call", "name", "x0"])
    rows = []
    for e, ems in enumerate(exps):
        end = min(t_end, ems - 60_000)
        n = rng.poisson(cfg["rate_per_day"] * (end - t0) / DAY_MS)
        times = np.sort(rng.integers(t0, end, n))
        cand = inst_df.index[inst_df["e"] == e].to_numpy()
        w = np.exp(-0.25 * inst_df.loc[cand, "x0"].to_numpy() ** 2)
        w /= w.sum()
        for tt in times:
            k_extra = rng.integers(1, 3) if rng.random() < cfg["batch_prob"] else 0
            for _ in range(1 + k_extra):
                rows.append((int(tt), int(rng.choice(cand, p=w))))
    ev = pd.DataFrame(rows, columns=["t", "inst"]).sort_values(["t", "inst"], kind="mergesort").reset_index(drop=True)
    ut = np.unique(ev["t"].to_numpy())
    dt_y = np.diff(np.concatenate([[t0], ut])) / DAY_MS / 365.0
    sv = float(cfg["spot_vol"])
    logS = math.log(S0) + np.cumsum(-0.5 * sv * sv * dt_y + sv * np.sqrt(dt_y) * rng.standard_normal(len(ut)))
    S_at = dict(zip(ut.tolist(), np.exp(logS).tolist()))

    E = len(exps)
    theta = np.tile(np.asarray(cfg["theta0"], float)[:m], (E, 1))
    n_inst = len(inst_df)
    s2u = nat[P_S2U]
    u = rng.standard_normal(n_inst) * math.sqrt(s2u) if spec.residual else np.zeros(n_inst)
    exp_idx = [np.flatnonzero(inst_df["e"].to_numpy() == e) for e in range(E)]
    t_last = np.full(E, t0, dtype=np.int64)
    S_last = np.full(E, S0)
    b = np.zeros(MAXM)
    out = []
    for tt, ii in zip(ev["t"].to_numpy(), ev["inst"].to_numpy()):
        e, ems, K, call, name, _ = inst[ii]
        S = S_at[int(tt)]
        dt = (tt - t_last[e]) / DAY_MS
        T = (ems - tt) / DAY_MS / 365.0
        if spec.spot and S != S_last[e]:
            r = math.log(S / S_last[e])
            atm = max(theta[e, 0], consts[C_ATMFLOOR]) / 100.0
            theta[e, 0] += nat[P_GAMMA] * theta[e, 1] * r / (atm * math.sqrt(T)) + nat[P_BETA] * r
        if dt > 0:
            integ = tau_integral((ems - t_last[e]) / DAY_MS, (ems - tt) / DAY_MS, nat[P_ALPHA],
                                 float(consts[C_TAUREF]), float(consts[C_TAUFLOOR]))
            theta[e] += np.sqrt(q * integ) * rng.standard_normal(m)
            if spec.residual:
                cc = math.exp(-nat[P_KAPPA] * dt)
                sel = exp_idx[e]
                u[sel] = cc * u[sel] + math.sqrt(s2u * (1 - cc * cc)) * rng.standard_normal(len(sel))
        t_last[e] = tt
        S_last[e] = S
        k = math.log(K / S)
        atm = max(theta[e, 0], consts[C_ATMFLOOR]) / 100.0
        x = min(max(k / (atm * math.sqrt(T)), -consts[C_XCLAMP]), consts[C_XCLAMP])
        _basis(x, spec.basis_id, consts[C_KNOT], b)
        fair = float(b[:m] @ theta[e]) + (u[ii] if spec.residual else 0.0)
        tn = tick_noise_at(fair, k, T, call, consts)
        tau_days = (ems - tt) / DAY_MS
        bk = int(bucket_index(np.array([tau_days]), settings)[0])
        R = nat[P_S2E] if spec.weighting == "F0" else nat[P_S2B + bk] + tn * tn / 12.0
        d = 1 if rng.random() < 0.5 else -1
        hs = cfg["hs_c_tick"] * tn + cfg["hs_c_vega"]
        y = fair + d * hs + math.sqrt(R) * rng.standard_normal()
        out.append((int(tt), ems, name, k, T, S, y, d, hs, call, tau_days, fair, bool(call)))
    df = pd.DataFrame(out, columns=["info_ms", "expiry_ms", "instrument_name", "k_causal", "T", "index_price",
                                    "iv", "d", "hs", "is_call", "tau_days", "true_fair", "_c"]).drop(columns="_c")
    df["is_screen"] = True
    df["is_block"] = False
    df["is_combo"] = False
    df["is_liquidation"] = False
    return df
