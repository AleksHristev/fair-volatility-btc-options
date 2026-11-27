from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fairvol.filter.dense import run_dense_core
from fairvol.filter.events import Events, filter_consts, iv_valid
from fairvol.filter.kalman import OUT_FIELDS

F = {n: i for i, n in enumerate(OUT_FIELDS)}
EV_COLS = ["trade_id_int", "timestamp", "info_ms", "expiry_ms", "instrument_name", "k_causal", "T",
           "index_price", "iv", "d", "is_screen", "is_block", "is_combo", "is_liquidation", "is_call", "tau_days"]
Q_COLS = ["t_ms", "expiry_ms", "instrument_name", "k_causal", "T", "S", "is_call", "tau_days", "dt_target_days",
          "target_tau_days"]
TRADE_OUT = ("has_pred", "m_prior", "v_prior", "r_obs", "nu", "m_post", "v_post", "seq_nu", "seq_s", "in_lik", "x")


@dataclass
class Inputs:
    tr: pd.DataFrame
    q: pd.DataFrame
    ev: Events
    is_trade: np.ndarray
    tune0: int
    lik_start: int


def run(inp: Inputs, s: dict[str, Any], name: str, params: tuple, mult: np.ndarray | None = None,
        clk: np.ndarray | None = None, eclk: np.ndarray | None = None, arrays: tuple | None = None):
    spec, nat = params
    ev = inp.ev
    n = len(ev.t)
    mult = np.ones(n) if mult is None else np.ascontiguousarray(mult, float)
    clk = ev.t.astype(float) if clk is None else np.ascontiguousarray(clk, float)
    eclk = ev.exp_ms.astype(float) if eclk is None else np.ascontiguousarray(eclk, float)
    a = ev.arrays() if arrays is None else arrays
    ll, nl, nneg, out = run_dense_core(*a, nat, spec.basis_id, spec.weighting_id, spec.residual, spec.spot,
                                       filter_consts(s), int(s["filter"]["burn_in_updates"]), inp.lik_start, True,
                                       mult, clk, eclk)
    if nneg:
        raise RuntimeError(f"{name}: {nneg} updates left a negative state variance (stop rule)")
    return ll, nl, nneg, out


def trade_outputs(inp: Inputs, s: dict[str, Any], out: np.ndarray) -> pd.DataFrame:
    ev = inp.ev
    ist = inp.is_trade
    ti = ev.src[ist]
    n = len(inp.tr)
    cols = {c: np.full(n, np.nan) for c in TRADE_OUT}
    for c in cols:
        cols[c][ti] = out[F[c]][ist]
    tro = pd.DataFrame(cols)
    tro.insert(0, "trade_id_int", inp.tr["trade_id_int"].to_numpy())
    tro["upd"] = False
    tro.loc[ti, "upd"] = ev.upd[ist]
    tro["valid_iv"] = iv_valid(inp.tr, s)
    return tro


def grid_outputs(inp: Inputs, out: np.ndarray) -> pd.DataFrame:
    ev = inp.ev
    isq = ~inp.is_trade
    qi = ev.src[isq]
    pred = pd.DataFrame({"m": np.full(len(inp.q), np.nan), "v_state": np.nan, "v_target": np.nan, "r_obs": np.nan})
    pred.loc[qi, "m"] = out[F["m_prior"]][isq]
    pred.loc[qi, "v_state"] = out[F["v_prior"]][isq]
    pred.loc[qi, "v_target"] = out[F["v_target"]][isq]
    pred.loc[qi, "r_obs"] = out[F["r_obs"]][isq]
    return pred.astype("float64")


def trade_multipliers(inp: Inputs, per_trade: np.ndarray) -> np.ndarray:
    ev = inp.ev
    m = np.ones(len(ev.t))
    ist = inp.is_trade
    vals = np.asarray(per_trade, float)[ev.src[ist]]
    vals = np.where(np.isfinite(vals) & ev.upd[ist], vals, 1.0)
    m[ist] = vals
    return m
