from __future__ import annotations

from typing import Any

import numba as nb
import numpy as np
import pandas as pd

from fairvol.buckets import delta_bucket, expiry_bucket
from fairvol.data.instruments import tick_size
from fairvol.pricing import black76 as b76
from fairvol.pricing.forward import Convention, model_iv, self_implied_forward, year_frac

SRC_TRADE_IMPLIED, SRC_RECENT, SRC_INDEX = 0, 1, 2
CAUSAL_SOURCE_LABELS = {SRC_TRADE_IMPLIED: "trade_implied", SRC_RECENT: "expiry_recent_ratio",
                        SRC_INDEX: "index"}


@nb.njit(cache=True)
def _window_median_sorted(src_t: np.ndarray, src_v: np.ndarray, q_t: np.ndarray,
                          lookback_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = q_t.shape[0]
    med = np.full(n, np.nan)
    cnt = np.zeros(n, np.int64)
    recent = np.full(n, np.nan)
    lo = np.searchsorted(src_t, q_t - lookback_ms, side="left")
    hi = np.searchsorted(src_t, q_t, side="left")
    for i in range(n):
        a, b = lo[i], hi[i]
        if b > 0:
            recent[i] = src_v[b - 1]
        if b > a:
            med[i] = np.median(src_v[a:b])
            cnt[i] = b - a
    return med, cnt, recent


def causal_forward_ratio(src_t: np.ndarray, src_ratio: np.ndarray, src_expiry: np.ndarray,
                         q_t: np.ndarray, q_expiry: np.ndarray, lookback_ms: int
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ti = np.full(len(q_t), np.nan)
    cnt = np.zeros(len(q_t), np.int64)
    recent = np.full(len(q_t), np.nan)
    ok = np.isfinite(src_ratio)
    s_t, s_v, s_e = src_t[ok], src_ratio[ok], src_expiry[ok]
    s_order = np.lexsort((s_t, s_e))
    s_t, s_v, s_e = s_t[s_order], s_v[s_order], s_e[s_order]
    exp_u, s_start = np.unique(s_e, return_index=True)
    s_end = np.append(s_start[1:], len(s_e))
    bounds = {int(e): (int(a), int(b)) for e, a, b in zip(exp_u, s_start, s_end)}
    q_order = np.argsort(q_expiry, kind="mergesort")
    qe_sorted = q_expiry[q_order]
    q_u, q_start = np.unique(qe_sorted, return_index=True)
    q_end = np.append(q_start[1:], len(qe_sorted))
    for e, a, b in zip(q_u, q_start, q_end):
        rows = q_order[a:b]
        if int(e) not in bounds:
            continue
        sa, sb = bounds[int(e)]
        m, c, r = _window_median_sorted(s_t[sa:sb].astype(np.int64), s_v[sa:sb].astype(np.float64),
                                        q_t[rows].astype(np.int64), int(lookback_ms))
        ti[rows], cnt[rows], recent[rows] = m, c, r
    causal = np.where(np.isfinite(ti), ti, recent)
    src = np.where(np.isfinite(ti), SRC_TRADE_IMPLIED, np.where(np.isfinite(recent), SRC_RECENT, SRC_INDEX))
    return ti, cnt, causal, src


def derive_fields(df: pd.DataFrame, conv: Convention, settings: dict[str, Any],
                  carry: pd.DataFrame | None = None) -> pd.DataFrame:
    fw = settings["forward"]
    band_lo, band_hi = fw["identified_abs_delta"]
    look_ms = int(fw["lookback_minutes"]) * 60_000
    S = df["index_price"].to_numpy(float)
    K = df["strike"].to_numpy(float)
    P = df["price"].to_numpy(float)
    IV = df["iv"].to_numpy(float)
    C = df["is_call"].to_numpy(bool)
    tau = df["tau_ms"].to_numpy(float)
    screen = df["is_screen"].to_numpy(bool)
    ts = df["timestamp"].to_numpy(np.int64)
    exp = df["expiry_ms"].to_numpy(np.int64)

    d_idx = np.abs(b76.delta_forward(S, K, year_frac(tau, float(settings["pricing"]["default_day_count"])),
                                     IV / 100.0, C))
    ident = screen & (d_idx >= band_lo) & (d_idx <= band_hi)
    T = year_frac(tau, conv.day_count)
    Fself = np.full(len(df), np.nan)
    if conv.underlying == "forward" and ident.any():
        Fself[ident] = self_implied_forward(P[ident], IV[ident], S[ident], K[ident], T[ident], C[ident], conv)
    ratio_self = Fself / S

    src_t, src_r, src_e = ts, ratio_self, exp
    if carry is not None and len(carry):
        src_t = np.concatenate([carry["timestamp"].to_numpy(np.int64), ts])
        src_r = np.concatenate([carry["ratio"].to_numpy(float), ratio_self])
        src_e = np.concatenate([carry["expiry_ms"].to_numpy(np.int64), exp])
    ti, n_ti, causal, csrc = causal_forward_ratio(src_t, src_r, src_e, ts, exp, look_ms)

    F = S * ti
    fsrc = np.where(np.isfinite(F), "trade_implied", "").astype(object)
    fb1 = ~np.isfinite(F) & np.isfinite(Fself)
    F[fb1] = Fself[fb1]
    fsrc[fb1] = "own_identified"
    fb2 = ~np.isfinite(F)
    F[fb2] = S[fb2]
    fsrc[fb2] = "index"
    F_causal = np.where(np.isfinite(causal), S * causal, S)

    out = df.copy()
    out["T"] = T
    out["F_self"] = Fself
    out["ratio_self"] = ratio_self
    out["n_forward_window"] = n_ti
    out["F"] = F
    out["forward_source"] = fsrc
    out["F_causal"] = F_causal
    out["causal_forward_source"] = pd.Categorical.from_codes(
        csrc, [CAUSAL_SOURCE_LABELS[i] for i in range(3)])
    out["k"] = np.log(K / F)
    out["k_causal"] = np.log(K / F_causal)
    out["delta"] = b76.delta_forward(F, K, T, IV / 100.0, C, conv.rate)
    out["abs_delta"] = np.abs(out["delta"])
    div = F if conv.divisor == "underlying" else S
    out["vega_btc_volpt"] = b76.vega_inverse_per_volpt(F, K, T, IV / 100.0, conv.rate, divisor=div)
    out["tick"] = tick_size(P, settings["instruments"]["tick_size"], settings["instruments"]["tick_size_steps"])
    out["tick_noise_volpts"] = out["tick"] / out["vega_btc_volpt"]
    out["is_tick_noisy"] = out["tick_noise_volpts"] > float(settings["flags"]["tick_noise_max_volpts"])
    bounds = tuple(float(b) for b in settings["pricing"]["iv_bounds"])
    MP = df["mark_price"].to_numpy(float)
    okm = np.isfinite(MP) & (MP > 0)
    mark_iv = np.full(len(df), np.nan)
    mark_iv[okm] = model_iv(MP[okm], F[okm], S[okm], K[okm], T[okm], C[okm], conv, bounds)
    out["mark_iv"] = mark_iv
    iv_F = model_iv(P, F, S, K, T, C, conv, bounds)
    out["iv_F"] = iv_F
    out["usd_notional"] = df["amount"] * S
    out["vega_notional_usd"] = out["vega_btc_volpt"] * df["amount"] * S
    out["vega_usd_volpt"] = out["vega_btc_volpt"] * S
    out["expiry_bucket"] = expiry_bucket(df["tau_days"], settings)
    out["delta_bucket"] = delta_bucket(out["abs_delta"], settings)
    out["delta_band_ident"] = ident
    return out
