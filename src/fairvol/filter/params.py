from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from fairvol.filter.kalman import (BASIS_IDS, N_PARAMS, P_ALPHA, P_BETA, P_GAMMA, P_KAPPA, P_QC, P_QL,
                                   P_QS, P_QW, P_S2B, P_S2BLOCK, P_S2COMBO, P_S2E, P_S2U)

N_BUCKETS = 5


@dataclass(frozen=True)
class Spec:
    basis: str
    weighting: str
    residual: bool
    spot: bool
    legs: bool = False

    @property
    def name(self) -> str:
        return (f"{self.weighting}-{self.basis}-res{int(self.residual)}-spot{int(self.spot)}"
                + ("-legs" if self.legs else ""))

    @property
    def basis_id(self) -> int:
        return BASIS_IDS[self.basis]

    @property
    def weighting_id(self) -> int:
        return 0 if self.weighting == "F0" else 1


def candidates(settings: dict[str, Any]) -> list[Spec]:
    f = settings["filter"]
    return [Spec(b, w, bool(r), bool(sp)) for w, b, r, sp in
            product(f["weightings"], f["bases"], f["residual_options"], f["spot_options"])]


def free_params(spec: Spec) -> list[tuple[str, int, str]]:
    out = [("q_level", P_QL, "log"), ("q_slope", P_QS, "log"), ("q_curv", P_QC, "log")]
    if spec.basis == "HW":
        out.append(("q_wing", P_QW, "log"))
    out.append(("alpha", P_ALPHA, "id"))
    if spec.weighting == "F0":
        out.append(("s2_e", P_S2E, "log"))
    else:
        out += [(f"s2_b{i}", P_S2B + i, "log") for i in range(N_BUCKETS)]
    if spec.residual:
        out += [("kappa", P_KAPPA, "log"), ("s2_u", P_S2U, "log")]
    if spec.spot:
        out += [("gamma", P_GAMMA, "id"), ("beta", P_BETA, "scaled")]
    if spec.legs:
        out += [("s2_block", P_S2BLOCK, "log"), ("s2_combo", P_S2COMBO, "log")]
    return out


def _is_bucket_var(name: str) -> bool:
    return len(name) == 5 and name.startswith("s2_b") and name[4].isdigit()


def _start_natural(name: str, st: dict[str, float]) -> float:
    if _is_bucket_var(name):
        return float(st["s2_b"])
    return float(st[name])


def _bounds_natural(name: str, bd: dict[str, list[float]]) -> tuple[float, float]:
    key = "q" if name.startswith("q_") else ("s2" if name.startswith("s2_") else name)
    lo, hi = bd[key]
    return float(lo), float(hi)


class ParamMap:
    def __init__(self, spec: Spec, settings: dict[str, Any]) -> None:
        f = settings["filter"]
        self.spec = spec
        self.free = free_params(spec)
        self.scale = float(f["beta_scale"])
        self.start_nat = np.array([_start_natural(n, f["start"]) for n, _, _ in self.free])
        self.bounds_nat = [_bounds_natural(n, f["bounds"]) for n, _, _ in self.free]
        self.base = np.zeros(N_PARAMS)
        self.base[P_S2E] = float(f["start"]["s2_e"])
        self.base[P_S2B:P_S2B + N_BUCKETS] = float(f["start"]["s2_b"])
        self.base[P_QW] = float(f["start"]["q_wing"])
        self.base[P_KAPPA] = float(f["start"]["kappa"])
        self.base[P_S2U] = float(f["start"]["s2_u"])
        self.base[P_S2BLOCK] = float(f["start"]["s2_block"])
        self.base[P_S2COMBO] = float(f["start"]["s2_combo"])

    @property
    def names(self) -> list[str]:
        return [n for n, _, _ in self.free]

    def to_opt(self, nat: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.free))
        for i, (_, _, tr) in enumerate(self.free):
            out[i] = np.log(nat[i]) if tr == "log" else (nat[i] / self.scale if tr == "scaled" else nat[i])
        return out

    def to_nat(self, z: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.free))
        for i, (_, _, tr) in enumerate(self.free):
            out[i] = np.exp(z[i]) if tr == "log" else (z[i] * self.scale if tr == "scaled" else z[i])
        return out

    def vector(self, nat_free: np.ndarray) -> np.ndarray:
        v = self.base.copy()
        for (_, idx, _), val in zip(self.free, nat_free):
            v[idx] = val
        return v

    def opt_bounds(self) -> list[tuple[float, float]]:
        lo = np.array([b[0] for b in self.bounds_nat])
        hi = np.array([b[1] for b in self.bounds_nat])
        return list(zip(self.to_opt(lo), self.to_opt(hi)))


def spec_from_name(name: str) -> Spec:
    parts = name.split("-")
    return Spec(basis=parts[1], weighting=parts[0], residual=parts[2] == "res1", spot=parts[3] == "spot1",
                legs=len(parts) > 4 and parts[4] == "legs")


def truth_vector(spec: Spec, truth: dict[str, Any], settings: dict[str, Any]) -> tuple[ParamMap, np.ndarray]:
    pm = ParamMap(spec, settings)
    vals = []
    for n in pm.names:
        vals.append(float(truth["s2_b"][int(n[-1])]) if _is_bucket_var(n) else float(truth[n]))
    return pm, np.array(vals)
