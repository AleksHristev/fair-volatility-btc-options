from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CONT_H2 = ["prior_sd", "r_obs", "tick_noise", "absz", "log_vega_notional", "d", "abs_delta", "log_tau",
           "log_tsl_opt", "log_tsl_exp", "asinh_flow_exp_5", "asinh_flow_all_5", "asinh_flow_exp_30",
           "asinh_flow_all_30", "idx_ret_5_pct", "idx_ret_30_pct", "idx_rv_5", "idx_rv_30"]
BIN_COMMON = ["near_expiry", "weekend"] + [f"hod_{h}" for h in range(1, 24)]
TYPE_DUMMIES = ["type_liquidation", "type_combo", "type_block"]
LGBM_H2 = ["nu", "prior_sd", "r_obs", "tick_noise", "absz", "vega_notional", "d", "type", "abs_delta", "tau_days",
           "tsl_opt_min", "tsl_exp_min", "flow_exp_5", "flow_all_5", "flow_exp_30", "flow_all_30", "idx_ret_5",
           "idx_ret_30", "idx_rv_5", "idx_rv_30", "hour_of_week", "near_expiry"]
H4_SWAP = {"tick_noise": "tick_noise_pre", "vega_notional": "vega_notional_pre", "abs_delta": "abs_delta_pre"}
LGBM_H4 = [H4_SWAP.get(c, c) for c in LGBM_H2 if c not in ("nu", "absz")]


def run_columns(feat: pd.DataFrame, fo: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=feat.index)
    m = fo["m_prior"].to_numpy(float)
    v = fo["v_prior"].to_numpy(float)
    r = fo["r_obs"].to_numpy(float)
    ivadj = np.where(feat["iv_adj_valid"].to_numpy(bool), feat["iv_adj"].to_numpy(float), np.nan)
    out["m_prior"] = m
    out["v_prior"] = v
    out["r_obs"] = r
    out["nu"] = ivadj - m
    out["prior_sd"] = np.sqrt(np.maximum(v, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["absz"] = np.abs(out["nu"].to_numpy()) / np.sqrt(v + r)
    o = feat["obs_pers"].to_numpy(np.int64)
    out["y"] = np.where(o >= 0, ivadj[np.clip(o, 0, None)] - m, np.nan)
    return out


def _encoded(df: pd.DataFrame, s: dict[str, Any], h4: bool) -> pd.DataFrame:
    c3 = s["learned"]
    cap = float(c3["tsl_cap_minutes"])
    sc = float(c3["flow_scale_usd"])
    sw = (lambda c: H4_SWAP.get(c, c)) if h4 else (lambda c: c)
    e = pd.DataFrame(index=df.index)
    e["prior_sd"] = df["prior_sd"]
    e["r_obs"] = df["r_obs"]
    e["tick_noise"] = df[sw("tick_noise")]
    if not h4:
        e["absz"] = df["absz"]
    e["log_vega_notional"] = np.log1p(np.maximum(df[sw("vega_notional")].to_numpy(float), 0.0))
    e["d"] = df["d"]
    e["abs_delta"] = df[sw("abs_delta")]
    e["log_tau"] = np.log(np.maximum(df["tau_days"].to_numpy(float), 1e-4))
    for k in ("opt", "exp"):
        x = df[f"tsl_{k}_min"].to_numpy(float)
        e[f"log_tsl_{k}"] = np.log1p(np.where(np.isfinite(x), np.minimum(x, cap), cap))
    for w in (5, 30):
        e[f"asinh_flow_exp_{w}"] = np.arcsinh(df[f"flow_exp_{w}"].to_numpy(float) / sc)
        e[f"asinh_flow_all_{w}"] = np.arcsinh(df[f"flow_all_{w}"].to_numpy(float) / sc)
        e[f"idx_ret_{w}_pct"] = 100.0 * df[f"idx_ret_{w}"].to_numpy(float)
        e[f"idx_rv_{w}"] = df[f"idx_rv_{w}"].to_numpy(float)
    e["near_expiry"] = df["near_expiry"].to_numpy(float)
    e["weekend"] = df["weekend"].to_numpy(float)
    hod = df["hour_of_day"].to_numpy(int)
    for h in range(1, 24):
        e[f"hod_{h}"] = (hod == h).astype(float)
    typ = df["type"].to_numpy(int)
    for code, nm in ((1, "type_liquidation"), (2, "type_combo"), (3, "type_block")):
        e[nm] = (typ == code).astype(float)
    return e.replace([np.inf, -np.inf], np.nan)


def linear_design(df: pd.DataFrame, s: dict[str, Any], stats: dict | None = None, h4: bool = False,
                  with_type: bool = True) -> tuple[np.ndarray, list[str], dict]:
    e = _encoded(df, s, h4)
    cont = [c for c in e.columns if c not in BIN_COMMON and c not in TYPE_DUMMIES]
    if stats is None:
        stats = {c: (float(np.nanmean(e[c])), float(np.nanstd(e[c])) or 1.0) for c in cont}
    for c in cont:
        mu, sd = stats[c]
        e[c] = (e[c].to_numpy(float) - mu) / sd
    cols = cont + BIN_COMMON + (TYPE_DUMMIES if with_type else [])
    Z = np.nan_to_num(e[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    return Z, cols, stats


def lgbm_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy().replace([np.inf, -np.inf], np.nan)
    if "type" in out:
        out["type"] = pd.Categorical(out["type"].astype(int), categories=[0, 1, 2, 3])
    return out


def huber_irls(X: np.ndarray, y: np.ndarray, delta: float, max_iter: int, tol: float,
               w0: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    w = np.ones(len(y)) if w0 is None else w0.astype(float).copy()
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    for it in range(1, max_iter + 1):
        r = y - X @ beta
        a = np.abs(r)
        wi = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-300))
        sw = np.sqrt(wi)
        XtX = (X * wi[:, None]).T @ X
        Xty = (X * wi[:, None]).T @ y
        new = np.linalg.solve(XtX + 1e-10 * np.eye(X.shape[1]), Xty)
        if np.max(np.abs(new - beta)) < tol * (1.0 + np.max(np.abs(beta))):
            return new, it
        beta = new
        del sw
    return beta, max_iter


def huber_loss(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
