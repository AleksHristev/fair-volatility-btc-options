from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fairvol.impact.features import index_features, index_minute_grid, signed_flow, time_since_last
from fairvol.impact.markout import first_observation

TYPES = ["screen", "liquidation", "combo", "block"]


def trade_type(df: pd.DataFrame) -> np.ndarray:
    t = np.zeros(len(df), np.int8)
    t[df["is_liquidation"].to_numpy(bool)] = 1
    t[df["is_combo"].to_numpy(bool)] = 2
    t[df["is_block"].to_numpy(bool)] = 3
    return t


def static_features(df: pd.DataFrame, s: dict[str, Any]) -> pd.DataFrame:
    c3 = s["learned"]
    q = df.copy()
    q["timestamp"] = q["info_ms"]
    out = pd.DataFrame(index=df.index)
    out["tick_noise"] = df["tick_noise_volpts"].to_numpy(float)
    out["vega_notional"] = df["vega_notional_usd"].to_numpy(float)
    out["d"] = df["d"].to_numpy(float)
    out["type"] = trade_type(df)
    out["abs_delta"] = df["abs_delta"].to_numpy(float)
    out["tau_days"] = df["tau_days"].to_numpy(float)
    out["tsl_opt_min"] = time_since_last(q, "instrument_name")
    out["tsl_exp_min"] = time_since_last(q, "expiry_ms")
    for w in c3["flow_windows_minutes"]:
        out[f"flow_exp_{w}"] = signed_flow(q, int(w) * 60_000, "expiry_ms")
        out[f"flow_all_{w}"] = signed_flow(q, int(w) * 60_000, None)
    grid = index_minute_grid(df)
    ix = index_features(q, grid, [int(w) for w in c3["index_windows_minutes"]])
    for w in c3["index_windows_minutes"]:
        out[f"idx_ret_{w}"] = ix[f"index_ret_{w}m"].to_numpy(float)
        out[f"idx_rv_{w}"] = ix[f"index_rv_{w}m"].to_numpy(float)
    ts = pd.to_datetime(df["info_ms"].to_numpy(np.int64), unit="ms", utc=True)
    out["hour_of_week"] = (ts.day_of_week * 24 + ts.hour).to_numpy(np.int16)
    out["hour_of_day"] = ts.hour.to_numpy(np.int16)
    out["weekend"] = (ts.day_of_week >= 5).astype(np.int8)
    out["near_expiry"] = (df["tau_days"].to_numpy(float) < float(c3["near_expiry_days"])).astype(np.int8)
    return out


def observations(df: pd.DataFrame, s: dict[str, Any], period_end_ms: np.ndarray) -> pd.DataFrame:
    h = int(s["learned"]["horizon_minutes"]) * 60_000
    mult = float(s["markout"]["max_lookforward_multiple"])
    inst, _ = pd.factorize(df["instrument_name"])
    out = pd.DataFrame(index=df.index)
    src_p = df["is_screen"].to_numpy(bool) & df["iv_adj_valid"].to_numpy(bool)
    src_m = (df["is_screen"].to_numpy(bool) & np.isfinite(df["mark_iv"].to_numpy(float))
             & df["iv_adj_F_valid"].to_numpy(bool))
    for name, src, tcol in (("obs_pers", src_p, "info_ms"), ("obs_mark", src_m, "timestamp")):
        t = df[tcol].to_numpy(np.int64)
        o = np.full(len(df), -1, np.int64)
        for end in np.unique(period_end_ms):
            rows = period_end_ms == end
            oo = first_observation(inst, t, src, h, mult, int(end))
            o[rows] = oo[rows]
        out[name] = o
    return out


def pretrade_greeks(df: pd.DataFrame, m_prior: np.ndarray, consts: np.ndarray) -> pd.DataFrame:
    from scipy.special import ndtr

    from fairvol.filter.kalman import tick_noise_at

    k = df["k_causal"].to_numpy(float)
    T = df["T"].to_numpy(float)
    call = df["is_call"].to_numpy(bool)
    m = np.asarray(m_prior, float)
    sig = np.maximum(m, 1.0) / 100.0
    st = sig * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (-k + 0.5 * st * st) / st
    delta = np.where(call, ndtr(d1), ndtr(d1) - 1.0)
    vega_btc = np.sqrt(T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi) / 100.0
    tn = np.full(len(df), np.nan)
    ok = np.isfinite(m) & np.isfinite(k) & np.isfinite(T) & (T > 0)
    for i in np.flatnonzero(ok):
        tn[i] = tick_noise_at(m[i], k[i], T[i], bool(call[i]), consts)
    S = df["index_price"].to_numpy(float)
    out = pd.DataFrame(index=df.index)
    out["abs_delta_pre"] = np.where(ok, np.abs(delta), np.nan)
    out["vega_usd_volpt_pre"] = np.where(ok, vega_btc * S, np.nan)
    out["vega_notional_pre"] = out["vega_usd_volpt_pre"].to_numpy() * df["amount"].to_numpy(float)
    out["tick_noise_pre"] = tn
    return out
