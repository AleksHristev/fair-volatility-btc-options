from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

MONTHS: dict[str, int] = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

NAME_RE = re.compile(
    r"^(?P<under>[A-Z]+(?:_[A-Z]+)?)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yy>\d{2})"
    r"-(?P<strike>\d+(?:d\d+)?)-(?P<type>[CP])$"
)


@dataclass(frozen=True)
class OptionSpec:
    name: str
    underlying: str
    expiry: datetime
    strike: float
    is_call: bool

    @property
    def expiry_ms(self) -> int:
        return int(self.expiry.timestamp() * 1000)


def parse_instrument_name(name: str, expiry_hour_utc: int = 8) -> OptionSpec:
    m = NAME_RE.match(name)
    if m is None:
        raise ValueError(f"not an option instrument name: {name!r}")
    mon = m.group("mon")
    if mon not in MONTHS:
        raise ValueError(f"unknown month {mon!r} in {name!r}")
    year = 2000 + int(m.group("yy"))
    expiry = datetime(year, MONTHS[mon], int(m.group("day")), expiry_hour_utc, tzinfo=timezone.utc)
    strike = float(m.group("strike").replace("d", "."))
    return OptionSpec(name=name, underlying=m.group("under"), expiry=expiry,
                      strike=strike, is_call=m.group("type") == "C")


def parse_instrument_series(names: pd.Series, expiry_hour_utc: int = 8) -> pd.DataFrame:
    uniq = pd.Series(names.unique())
    rows: list[dict[str, Any]] = []
    for n in uniq:
        try:
            spec = parse_instrument_name(str(n), expiry_hour_utc)
            rows.append({"instrument_name": n, "underlying": spec.underlying,
                         "expiry_ms": spec.expiry_ms, "strike": spec.strike,
                         "is_call": spec.is_call, "parse_ok": True})
        except (ValueError, TypeError):
            rows.append({"instrument_name": n, "underlying": None, "expiry_ms": None,
                         "strike": np.nan, "is_call": None, "parse_ok": False})
    table = pd.DataFrame(rows).set_index("instrument_name")
    out = table.reindex(names.to_numpy())
    out.index = names.index
    out["expiry_ms"] = out["expiry_ms"].astype("Int64")
    out["is_call"] = out["is_call"].astype("boolean")
    out["parse_ok"] = out["parse_ok"].fillna(False).astype(bool)
    return out


def tick_size(price: np.ndarray | pd.Series | float, base_tick: float,
              steps: list[dict[str, float]]) -> np.ndarray:
    p = np.asarray(price, dtype=np.float64)
    out = np.full(p.shape, base_tick, dtype=np.float64)
    for step in sorted(steps, key=lambda s: s["above_price"]):
        out = np.where(p >= step["above_price"], step["tick_size"], out)
    return out


def on_tick_grid(price: np.ndarray | pd.Series, base_tick: float,
                 steps: list[dict[str, float]], rel_tol: float = 1e-6) -> np.ndarray:
    p = np.asarray(price, dtype=np.float64)
    t = tick_size(p, base_tick, steps)
    q = p / t
    return np.abs(q - np.round(q)) < rel_tol * np.maximum(1.0, np.abs(q))
