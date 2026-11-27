from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fairvol.data.instruments import parse_instrument_series

RAW_FIELDS_EXCLUDED_FROM_DUP = ("file_day",)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(pd.NA, index=df.index, dtype="object")


def clean_trades(raw: pd.DataFrame, settings: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    steps: list[dict[str, Any]] = [{"rule": "raw rows", "removed": 0, "remaining": len(df)}]

    def drop(mask: pd.Series, rule: str) -> None:
        nonlocal df
        n = int(mask.sum())
        df = df.loc[~mask].copy()
        steps.append({"rule": rule, "removed": n, "remaining": len(df)})

    parsed = parse_instrument_series(df["instrument_name"], settings["instruments"]["expiry_hour_utc"])
    df = df.join(parsed[["underlying", "expiry_ms", "strike", "is_call", "parse_ok"]])
    drop(~df["parse_ok"] | (df["underlying"] != "BTC"), "instrument name not a parseable BTC option")
    drop(df["price"].isna() | (df["price"] <= 0), "missing or non-positive price")
    drop(df["iv"].isna() | (df["iv"] <= 0), "missing or non-positive iv")
    drop(df["timestamp"] >= df["expiry_ms"].astype("int64"), "trade at or after expiry")
    dup_cols = [c for c in raw.columns if c not in RAW_FIELDS_EXCLUDED_FROM_DUP]
    drop(df.duplicated(subset=dup_cols, keep="first"), "exact duplicate (all fields)")

    df["expiry_ms"] = df["expiry_ms"].astype("int64")
    df["is_call"] = df["is_call"].astype(bool)
    df["tau_ms"] = df["expiry_ms"] - df["timestamp"]
    df["tau_days"] = df["tau_ms"] / 86_400_000.0
    df["d"] = np.where(df["direction"] == "buy", 1, -1).astype("int8")
    if not df["direction"].isin(["buy", "sell"]).all():
        raise ValueError("unexpected direction values: "
                         f"{sorted(df.loc[~df['direction'].isin(['buy', 'sell']), 'direction'].unique())}")

    df["is_block"] = _col(df, "block_trade_id").notna().to_numpy()
    df["is_combo"] = (_col(df, "combo_id").notna() | _col(df, "combo_trade_id").notna()).to_numpy()
    df["is_liquidation"] = _col(df, "liquidation").notna().to_numpy()
    bd = pd.to_numeric(_col(df, "block_delayed"), errors="coerce")
    df["is_block_delayed"] = bd.notna().to_numpy()
    df["info_ms"] = np.where(bd.notna(), np.maximum(bd.fillna(0).to_numpy(), df["timestamp"].to_numpy()),
                             df["timestamp"].to_numpy()).astype("int64")
    df["is_screen"] = ~(df["is_block"] | df["is_combo"] | df["is_liquidation"])

    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["day"] = ts.dt.strftime("%Y-%m-%d")
    df["dom"] = ts.dt.day.astype("int16")
    df["hour"] = ts.dt.hour.astype("int16")
    df["weekday"] = ts.dt.weekday.astype("int16")
    df["is_weekend"] = df["weekday"] >= 5
    df["iv_dec"] = df["iv"] / 100.0
    df["trade_id_int"] = df["trade_id"].astype("int64")
    df = df.drop(columns=["parse_ok", "underlying"]).sort_values(["timestamp", "trade_id_int"],
                                                                 kind="mergesort")
    df = df.reset_index(drop=True)
    return df, pd.DataFrame(steps)
