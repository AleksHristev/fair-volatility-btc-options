from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fairvol.filter.events import KIND_QUERY, KIND_TRADE, Events, bucket_index, query_frame_events, trade_frame_events
from fairvol.filter.kalman import CLS_SCREEN, OUT_FIELDS
from fairvol.filter.run import run
from fairvol.benchmarks.svi import mark_star

F = {n: i for i, n in enumerate(OUT_FIELDS)}


@dataclass
class FusedInputs:
    tr: pd.DataFrame
    q: pd.DataFrame
    ev: Events
    is_trade: np.ndarray
    is_mark: np.ndarray
    lik_start: int
    tune0: int


def mark_frame_events(tr: pd.DataFrame, s: dict[str, Any], extra: pd.DataFrame | None = None) -> pd.DataFrame:
    m = mark_star(tr, s)
    ok = np.isfinite(m)
    sub = tr.loc[ok]
    out = pd.DataFrame({
        "t": sub["info_ms"].to_numpy(np.int64), "kind": KIND_TRADE, "expiry_ms": sub["expiry_ms"].to_numpy(np.int64),
        "instrument_name": sub["instrument_name"].astype(str).to_numpy(), "k": sub["k_causal"].to_numpy(float),
        "T": sub["T"].to_numpy(float), "S": sub["index_price"].to_numpy(float), "y": m[ok], "dhs": 0.0, "upd": True,
        "cls": CLS_SCREEN, "bucket": bucket_index(sub["tau_days"].to_numpy(float), s), "call": sub["is_call"].to_numpy(bool),
        "dt_target": np.nan, "bucket_target": 0, "src": np.flatnonzero(ok).astype(np.int64), "sub": 0, "mark": 1})
    if extra is not None and len(extra):
        ex = pd.DataFrame({
            "t": extra["t"].to_numpy(np.int64), "kind": KIND_TRADE, "expiry_ms": extra["expiry_ms"].to_numpy(np.int64),
            "instrument_name": extra["instrument_name"].astype(str).to_numpy(), "k": extra["k_causal"].to_numpy(float),
            "T": extra["T"].to_numpy(float), "S": extra["S"].to_numpy(float), "y": extra["y"].to_numpy(float), "dhs": 0.0,
            "upd": True, "cls": CLS_SCREEN, "bucket": bucket_index(extra["tau_days"].to_numpy(float), s),
            "call": extra["is_call"].to_numpy(bool), "dt_target": np.nan, "bucket_target": 0,
            "src": -1 - np.arange(len(extra), dtype=np.int64), "sub": 0, "mark": 2})
        out = pd.concat([out, ex], ignore_index=True)
    return out


def assemble_fused(frames: list[pd.DataFrame]) -> tuple[Events, np.ndarray]:
    ev = pd.concat(frames, ignore_index=True)
    order = np.lexsort((ev["src"].to_numpy(), ev["sub"].to_numpy(), ev["expiry_ms"].to_numpy(), ev["kind"].to_numpy(),
                        ev["t"].to_numpy()))
    ev = ev.iloc[order].reset_index(drop=True)
    exp_codes, exp_u = pd.factorize(ev["expiry_ms"], sort=True)
    key = pd.Series(exp_codes).astype(str) + "|" + ev["instrument_name"]
    first = ~key.duplicated()
    rank = first.groupby(exp_codes).cumsum() - 1
    first_rank = pd.Series(rank[first].to_numpy(), index=key[first].to_numpy())
    local = first_rank.reindex(key.to_numpy()).to_numpy(np.int64)
    n_per = np.bincount(exp_codes, weights=first.to_numpy(), minlength=len(exp_u)).astype(np.int64)
    slot_offset = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64)
    slot = slot_offset[exp_codes] + local
    E = Events(
        t=ev["t"].to_numpy(np.int64), kind=ev["kind"].to_numpy(np.int64), exp=exp_codes.astype(np.int64),
        slot=slot.astype(np.int64), k=ev["k"].to_numpy(float), T=ev["T"].to_numpy(float), S=ev["S"].to_numpy(float),
        y=ev["y"].to_numpy(float), dhs=ev["dhs"].to_numpy(float), upd=ev["upd"].to_numpy(bool),
        cls=ev["cls"].to_numpy(np.int64), bucket=ev["bucket"].to_numpy(np.int64), call=ev["call"].to_numpy(bool),
        dt_target=ev["dt_target"].to_numpy(float), bucket_target=ev["bucket_target"].to_numpy(np.int64),
        exp_ms=np.asarray(exp_u, np.int64), slot_offset=slot_offset, src=ev["src"].to_numpy(np.int64),
        meta={"n": len(ev), "n_trades": int(((ev["kind"] == KIND_TRADE) & (ev["mark"] == 0)).sum()),
              "n_marks": int((ev["mark"] == 1).sum()), "n_live_marks": int((ev["mark"] == 2).sum()),
              "n_queries": int((ev["kind"] == KIND_QUERY).sum()), "n_slots": int(slot_offset[-1])})
    return E, ev["mark"].to_numpy(np.int8)


def build(tr: pd.DataFrame, q: pd.DataFrame, s: dict[str, Any], with_marks: bool = True, extra: pd.DataFrame | None = None,
          lik_start: int = 0, tune0: int = 0) -> FusedInputs:
    te = trade_frame_events(tr, s, use_legs=False).assign(sub=1, mark=0)
    qe = query_frame_events(q, s).assign(sub=0, mark=0)
    frames = [te, qe] + ([mark_frame_events(tr, s, extra)] if with_marks else [])
    ev, mk = assemble_fused(frames)
    return FusedInputs(tr, q, ev, (ev.kind == KIND_TRADE) & (mk == 0), mk > 0, lik_start, tune0)


def run_fused(fi: FusedInputs, s: dict[str, Any], name: str, params, m_mark: float, c: float,
              marks_update: bool = True, trades_update: bool = True, n_events: int | None = None):
    ev = fi.ev
    n = len(ev.t) if n_events is None else int(n_events)
    upd = ev.upd.copy()
    if not trades_update:
        upd[fi.is_trade] = False
    upd[fi.is_mark] = bool(marks_update)
    mult = np.ones(len(ev.t))
    mult[fi.is_trade & upd] = float(c)
    mult[fi.is_mark] = float(m_mark)
    a = list(ev.arrays())
    a[9] = upd
    a = tuple(x[:n] if i < 15 else x for i, x in enumerate(a))
    _, _, _, out = run(fi, s, name, mult=mult[:n], clk=ev.t[:n].astype(float), eclk=ev.exp_ms.astype(float), arrays=a,
                       params=params)
    ist = fi.is_trade[:n]
    isq = ev.kind[:n] == KIND_QUERY
    tro = pd.DataFrame({c_: np.full(len(fi.tr), np.nan) for c_ in ("m_prior", "v_prior", "r_obs", "nu", "m_post", "v_post")})
    ti = ev.src[:n][ist]
    for c_ in tro.columns:
        tro.loc[ti, c_] = out[F[c_]][ist]
    tro.insert(0, "trade_id_int", fi.tr["trade_id_int"].to_numpy())
    tro["upd"] = False
    tro.loc[ti, "upd"] = upd[:n][ist]
    pred = pd.DataFrame({"m": np.full(len(fi.q), np.nan), "v_state": np.nan, "v_target": np.nan, "r_obs": np.nan})
    qi = ev.src[:n][isq]
    pred.loc[qi, "m"] = out[F["m_prior"]][isq]
    pred.loc[qi, "v_state"] = out[F["v_prior"]][isq]
    pred.loc[qi, "v_target"] = out[F["v_target"]][isq]
    pred.loc[qi, "r_obs"] = out[F["r_obs"]][isq]
    return tro, pred.astype("float64"), out
