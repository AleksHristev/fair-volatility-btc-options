from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


def day_weights(n_days: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=n_boot).astype(float)


def _solve_cells(stats: np.ndarray) -> np.ndarray:
    n, Sd, Sr, Sdd, Sdr, Srr, SD, SdD, SrD = np.moveaxis(stats, -1, 0)
    XtX = np.stack([np.stack([n, Sd, Sr], -1), np.stack([Sd, Sdd, Sdr], -1),
                    np.stack([Sr, Sdr, Srr], -1)], -2)
    XtD = np.stack([SD, SdD, SrD], -1)[..., None]
    return (np.linalg.pinv(XtX) @ XtD)[..., 0]


@dataclass
class ImpactResult:
    labels: list[str]
    n: np.ndarray
    n_days: np.ndarray
    mean: np.ndarray
    draws: np.ndarray

    def table(self, ci_level: float) -> pd.DataFrame:
        a = (1.0 - ci_level) / 2.0
        lo = np.nanquantile(self.draws, a, axis=0)
        hi = np.nanquantile(self.draws, 1.0 - a, axis=0)
        return pd.DataFrame({"group": self.labels, "n": self.n, "n_days": self.n_days,
                             "mean": self.mean, "ci_lo": lo, "ci_hi": hi,
                             "boot_se": np.nanstd(self.draws, axis=0, ddof=1)})

    def wald(self, min_obs: int) -> dict[str, float]:
        keep = self.n >= min_obs
        m = self.mean[keep]
        G = len(m)
        if G < 2:
            return {"groups_tested": G, "wald": np.nan, "df": 0, "p_value": np.nan}
        X = self.draws[:, keep]
        cov = np.cov(X, rowvar=False, ddof=1)
        R = np.hstack([np.eye(G - 1), -np.ones((G - 1, 1))])
        Rm = R @ m
        V = R @ cov @ R.T
        w = float(Rm @ np.linalg.solve(V, Rm))
        return {"groups_tested": G, "wald": w, "df": G - 1, "p_value": float(chi2.sf(w, G - 1))}


class ImpactBootstrap:
    def __init__(self, day: np.ndarray, cell: np.ndarray, d: np.ndarray, D: np.ndarray,
                 r: np.ndarray, weights: np.ndarray, day_order: np.ndarray) -> None:
        self.day_idx = pd.Index(day_order).get_indexer(day)
        if (self.day_idx < 0).any():
            raise ValueError("day labels not covered by day_order")
        self.cell_codes, self.cell_labels = pd.factorize(cell, sort=True)
        self.d = d.astype(float)
        self.D = D.astype(float)
        self.r = r.astype(float)
        self.W = weights
        self.n_days = len(day_order)
        nc = len(self.cell_labels)
        comps = np.column_stack([np.ones_like(self.d), self.d, self.r, self.d * self.d,
                                 self.d * self.r, self.r * self.r, self.D, self.d * self.D,
                                 self.r * self.D])
        S = np.zeros((self.n_days, nc, 9))
        np.add.at(S, (self.day_idx, self.cell_codes), comps)
        self.S = S
        self.coef_point = _solve_cells(S.sum(axis=0))
        self.coef_boot = _solve_cells(np.einsum("bd,dck->bck", self.W, S))

    def per_obs(self, centred: bool = True) -> np.ndarray:
        a, _, b = self.coef_point[self.cell_codes].T
        base = self.D - b * self.r - (a if centred else 0.0)
        return self.d * base

    def groups(self, group: np.ndarray | None, centred: bool = True,
               order: list[str] | None = None) -> ImpactResult:
        if group is None:
            group = np.array(["overall"] * len(self.d), dtype=object)
        g = pd.Series(group).astype(str).to_numpy()
        labels = order if order is not None else sorted(pd.unique(g))
        gi = pd.Index(labels).get_indexer(g)
        keep = gi >= 0
        nc, ng = len(self.cell_labels), len(labels)
        comps = np.column_stack([np.ones_like(self.d), self.d, self.d * self.r, self.d * self.D])[keep]
        T = np.zeros((self.n_days, ng, nc, 4))
        np.add.at(T, (self.day_idx[keep], gi[keep], self.cell_codes[keep]), comps)

        def means(Tw: np.ndarray, coef: np.ndarray) -> np.ndarray:
            a = coef[..., None, :, 0]
            b = coef[..., None, :, 2]
            num = Tw[..., 3] - b * Tw[..., 2] - (a * Tw[..., 1] if centred else 0.0)
            den = Tw[..., 0].sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num.sum(axis=-1) / den

        point = means(T.sum(axis=0), self.coef_point)
        Tb = np.einsum("bd,dgck->bgck", self.W, T)
        draws = means(Tb, self.coef_boot)
        n = T[..., 0].sum(axis=(0, 2))
        nd = (T[..., 0].sum(axis=2) > 0).sum(axis=0)
        return ImpactResult(labels=list(labels), n=n.astype(int), n_days=nd.astype(int),
                            mean=point, draws=draws)


def shuffle_within(d: np.ndarray, keys: list[np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes = pd.MultiIndex.from_arrays([pd.Series(k).astype(str).to_numpy() for k in keys]).factorize()[0] \
        if len(keys) > 1 else pd.factorize(pd.Series(keys[0]).astype(str))[0]
    out = d.copy()
    order = np.argsort(codes, kind="mergesort")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for block in np.split(order, bounds):
        out[block] = d[rng.permutation(block)]
    return out
