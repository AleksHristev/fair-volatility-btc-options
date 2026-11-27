from __future__ import annotations

import numpy as np
import pandas as pd


def reference_labels(settings: dict) -> dict[str, str]:
    cap = float(settings["impact"]["last_trade_max_age_minutes"])
    return {"smile": "smile_iv_adj", "last": "last_iv_adj", f"last<={cap:g}m": "lastcap_iv_adj"}


def find_observations(df: pd.DataFrame, src_mask: np.ndarray, horizon_ms: int,
                      max_multiple: float, value_cols: list[str]) -> pd.DataFrame:
    q = pd.DataFrame({"instrument_name": df["instrument_name"].to_numpy(),
                      "key": df["timestamp"].to_numpy(np.int64) + int(horizon_ms),
                      "_row": np.arange(len(df))}).sort_values("key", kind="mergesort")
    src = df.loc[src_mask, ["instrument_name", "timestamp", *value_cols]].rename(
        columns={c: f"obs_{c}" for c in value_cols})
    src = src.rename(columns={"timestamp": "key"})
    src["obs_ts"] = src["key"]
    src = src.sort_values("key", kind="mergesort")
    m = pd.merge_asof(q, src, on="key", by="instrument_name", direction="forward",
                      allow_exact_matches=True).sort_values("_row")
    t = df["timestamp"].to_numpy(np.int64)
    obs_ts = m["obs_ts"].to_numpy(float)
    ok = np.isfinite(obs_ts) & (obs_ts <= t + max_multiple * horizon_ms)
    out = pd.DataFrame(index=df.index)
    out["obs_ts"] = np.where(ok, obs_ts, np.nan)
    out["obs_delay_min"] = (out["obs_ts"] - t) / 60_000.0
    for c in value_cols:
        out[f"obs_{c}"] = np.where(ok, m[f"obs_{c}"].to_numpy(float), np.nan)
    return out


def assign_cells(df: pd.DataFrame, min_obs: int) -> pd.Series:
    typ = np.where(df["is_call"].to_numpy(bool), "C", "P")
    eb = df["expiry_bucket"].astype(str).to_numpy()
    db = df["delta_bucket"].astype(str).to_numpy()
    full = pd.Series(eb + "|" + db + "|" + typ, index=df.index)
    mid = pd.Series(eb + "|*|" + typ, index=df.index)
    cnt_full = full.map(full.value_counts())
    cnt_mid = mid.map(mid.value_counts())
    return pd.Series(np.where(cnt_full >= min_obs, full,
                              np.where(cnt_mid >= min_obs, mid, "pooled")), index=df.index)
