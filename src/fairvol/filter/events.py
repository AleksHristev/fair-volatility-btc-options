from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fairvol.filter.kalman import CLS_BLOCK, CLS_COMBO, CLS_SCREEN, C_ATMFLOOR, C_KNOT, C_SDLEVEL, \
    C_SDOTHER, C_TAUFLOOR, C_TAUREF, C_TICK_LARGE, C_TICK_SMALL, C_TICK_SWITCH, C_TNCAP, C_VFLOOR, \
    C_XCLAMP, C_FACTOR, N_CONSTS

KIND_QUERY, KIND_TRADE = 0, 1


@dataclass
class Events:
    t: np.ndarray
    kind: np.ndarray
    exp: np.ndarray
    slot: np.ndarray
    k: np.ndarray
    T: np.ndarray
    S: np.ndarray
    y: np.ndarray
    dhs: np.ndarray
    upd: np.ndarray
    cls: np.ndarray
    bucket: np.ndarray
    call: np.ndarray
    dt_target: np.ndarray
    bucket_target: np.ndarray
    exp_ms: np.ndarray
    slot_offset: np.ndarray
    src: np.ndarray
    meta: dict = field(default_factory=dict)

    def arrays(self) -> tuple:
        return (self.t, self.kind, self.exp, self.slot, self.k, self.T, self.S, self.y, self.dhs,
                self.upd, self.cls, self.bucket, self.call, self.dt_target, self.bucket_target,
                self.exp_ms, self.slot_offset)


def bucket_index(tau_days: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    edges = np.array([float(e) for e in settings["buckets"]["expiry_days"]["edges"]])
    idx = np.searchsorted(edges, np.asarray(tau_days, float), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2).astype(np.int64)


def filter_consts(settings: dict[str, Any]) -> np.ndarray:
    f = settings["filter"]
    ins = settings["instruments"]
    steps = sorted(ins["tick_size_steps"], key=lambda x: x["above_price"])
    if len(steps) != 1:
        raise ValueError("filter tick-noise code supports exactly one tick-size step")
    c = np.zeros(N_CONSTS)
    c[C_XCLAMP] = float(f["x_clamp"])
    c[C_ATMFLOOR] = float(f["atm_vol_floor"])
    c[C_TAUREF] = float(f["tau_ref_days"])
    c[C_TAUFLOOR] = float(f["tau_floor_days"])
    c[C_SDLEVEL] = float(f["init_sd_level"])
    c[C_SDOTHER] = float(f["init_sd_other"])
    c[C_KNOT] = float(f["wing_knot"])
    c[C_TICK_SMALL] = float(ins["tick_size"])
    c[C_TICK_LARGE] = float(steps[0]["tick_size"])
    c[C_TICK_SWITCH] = float(steps[0]["above_price"])
    c[C_TNCAP] = float(f["tick_noise_cap"])
    c[C_VFLOOR] = float(f["v_floor"])
    mode = str(f["cov_approximation"])
    if mode not in ("factor", "registered"):
        raise ValueError(f"unknown filter.cov_approximation {mode!r}")
    c[C_FACTOR] = 1.0 if mode == "factor" else 0.0
    return c


def iv_valid(tr: pd.DataFrame, settings: dict[str, Any]) -> np.ndarray:
    y = tr["iv"].to_numpy(float)
    dhs = (tr["d"].to_numpy(float) * tr["hs"].to_numpy(float))
    lo, hi = (100.0 * float(b) for b in settings["pricing"]["iv_bounds"])
    y_adj = y - dhs
    return (np.isfinite(y) & (y > 0) & np.isfinite(dhs) & np.isfinite(tr["k_causal"].to_numpy(float))
            & np.isfinite(tr["T"].to_numpy(float)) & (y_adj >= lo) & (y_adj <= hi))


def trade_frame_events(tr: pd.DataFrame, settings: dict[str, Any], use_legs: bool) -> pd.DataFrame:
    y = tr["iv"].to_numpy(float)
    dhs = (tr["d"].to_numpy(float) * tr["hs"].to_numpy(float))
    k = tr["k_causal"].to_numpy(float)
    valid = iv_valid(tr, settings)
    screen = tr["is_screen"].to_numpy(bool)
    block = tr["is_block"].to_numpy(bool)
    combo = tr["is_combo"].to_numpy(bool)
    liq = tr["is_liquidation"].to_numpy(bool)
    cls = np.where(block, CLS_BLOCK, np.where(combo, CLS_COMBO, CLS_SCREEN))
    upd = valid & screen
    if use_legs:
        upd = upd | (valid & (block | combo) & ~liq)
    return pd.DataFrame({
        "t": tr["info_ms"].to_numpy(np.int64), "kind": KIND_TRADE, "expiry_ms": tr["expiry_ms"].to_numpy(np.int64),
        "instrument_name": tr["instrument_name"].astype(str).to_numpy(), "k": np.where(np.isfinite(k), k, 0.0),
        "T": tr["T"].to_numpy(float), "S": tr["index_price"].to_numpy(float), "y": y,
        "dhs": np.where(np.isfinite(dhs), dhs, 0.0), "upd": upd, "cls": cls.astype(np.int64),
        "bucket": bucket_index(tr["tau_days"].to_numpy(float), settings),
        "call": tr["is_call"].to_numpy(bool), "dt_target": np.nan, "bucket_target": 0,
        "src": np.arange(len(tr), dtype=np.int64)})


def query_frame_events(qs: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({
        "t": qs["t_ms"].to_numpy(np.int64), "kind": KIND_QUERY, "expiry_ms": qs["expiry_ms"].to_numpy(np.int64),
        "instrument_name": qs["instrument_name"].astype(str).to_numpy(), "k": qs["k_causal"].to_numpy(float),
        "T": qs["T"].to_numpy(float), "S": qs["S"].to_numpy(float), "y": np.nan, "dhs": 0.0, "upd": False,
        "cls": CLS_SCREEN, "bucket": bucket_index(qs["tau_days"].to_numpy(float), settings),
        "call": qs["is_call"].to_numpy(bool), "dt_target": qs["dt_target_days"].to_numpy(float),
        "bucket_target": bucket_index(qs["target_tau_days"].to_numpy(float), settings),
        "src": np.arange(len(qs), dtype=np.int64)})


def assemble(frames: list[pd.DataFrame]) -> Events:
    ev = pd.concat(frames, ignore_index=True)
    order = np.lexsort((ev["src"].to_numpy(), ev["expiry_ms"].to_numpy(), ev["kind"].to_numpy(),
                        ev["t"].to_numpy()))
    ev = ev.iloc[order].reset_index(drop=True)
    exp_codes, exp_u = pd.factorize(ev["expiry_ms"], sort=True)
    ev["exp"] = exp_codes
    key = ev["exp"].astype(str) + "|" + ev["instrument_name"]
    first = ~key.duplicated()
    rank = first.groupby(ev["exp"]).cumsum() - 1
    first_rank = pd.Series(rank[first].to_numpy(), index=key[first].to_numpy())
    local = first_rank.reindex(key.to_numpy()).to_numpy(np.int64)
    n_per = np.bincount(exp_codes, weights=first.to_numpy(), minlength=len(exp_u)).astype(np.int64)
    slot_offset = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64)
    slot = slot_offset[exp_codes] + local
    return Events(
        t=ev["t"].to_numpy(np.int64), kind=ev["kind"].to_numpy(np.int64), exp=exp_codes.astype(np.int64),
        slot=slot.astype(np.int64), k=ev["k"].to_numpy(float), T=ev["T"].to_numpy(float),
        S=ev["S"].to_numpy(float), y=ev["y"].to_numpy(float), dhs=ev["dhs"].to_numpy(float),
        upd=ev["upd"].to_numpy(bool), cls=ev["cls"].to_numpy(np.int64), bucket=ev["bucket"].to_numpy(np.int64),
        call=ev["call"].to_numpy(bool), dt_target=ev["dt_target"].to_numpy(float),
        bucket_target=ev["bucket_target"].to_numpy(np.int64), exp_ms=np.asarray(exp_u, np.int64),
        slot_offset=slot_offset, src=ev["src"].to_numpy(np.int64),
        meta={"n": len(ev), "n_trades": int((ev["kind"] == KIND_TRADE).sum()),
              "n_queries": int((ev["kind"] == KIND_QUERY).sum()), "n_slots": int(slot_offset[-1])})
