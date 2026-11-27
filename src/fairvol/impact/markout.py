from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd


@nb.njit(cache=True)
def _first_obs(q_inst, q_t, s_inst_start, s_inst_end, s_t, lo_off, hi_off, t_max):
    n = q_t.shape[0]
    out = np.full(n, -1, np.int64)
    for i in range(n):
        a = s_inst_start[q_inst[i]]
        b = s_inst_end[q_inst[i]]
        if b <= a:
            continue
        lo = q_t[i] + lo_off
        hi = min(q_t[i] + hi_off, t_max)
        if hi < lo:
            continue
        L, R = a, b
        while L < R:
            mid = (L + R) // 2
            if s_t[mid] < lo:
                L = mid + 1
            else:
                R = mid
        if L < b and s_t[L] <= hi:
            out[i] = L
    return out


def first_observation(inst: np.ndarray, t: np.ndarray, src_mask: np.ndarray, horizon_ms: int,
                      mult: float, t_max: int) -> np.ndarray:
    inst = np.asarray(inst, np.int64)
    t = np.asarray(t, np.int64)
    src = np.flatnonzero(src_mask)
    order = src[np.lexsort((t[src], inst[src]))]
    s_inst, s_t = inst[order], t[order]
    n_inst = int(inst.max()) + 1 if len(inst) else 0
    start = np.searchsorted(s_inst, np.arange(n_inst), side="left")
    end = np.searchsorted(s_inst, np.arange(n_inst), side="right")
    pos = _first_obs(inst, t, start.astype(np.int64), end.astype(np.int64), s_t, int(horizon_ms),
                     int(round(mult * horizon_ms)), int(t_max))
    return np.where(pos >= 0, order[np.clip(pos, 0, None)], -1)


class MeanBoot:
    def __init__(self, day_idx: np.ndarray, weights: np.ndarray, n_days: int) -> None:
        self.day_idx = np.asarray(day_idx, np.int64)
        self.W = weights
        self.n_days = n_days

    def groups(self, values: np.ndarray, group: np.ndarray | None, labels: list[str] | None = None
               ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v = np.asarray(values, float)
        g = np.full(len(v), "overall", dtype=object) if group is None else np.asarray(group).astype(str)
        labs = list(labels) if labels is not None else sorted(pd.unique(g).tolist())
        codes = pd.Index(labs).get_indexer(g).astype(np.int64)
        keep = (codes >= 0) & np.isfinite(v)
        G = len(labs)
        cnt = np.zeros((self.n_days, G))
        tot = np.zeros((self.n_days, G))
        np.add.at(cnt, (self.day_idx[keep], codes[keep]), 1.0)
        np.add.at(tot, (self.day_idx[keep], codes[keep]), v[keep])
        with np.errstate(invalid="ignore", divide="ignore"):
            point = tot.sum(0) / cnt.sum(0)
            draws = (self.W @ tot) / (self.W @ cnt)
        return labs, cnt.sum(0).astype(int), (cnt > 0).sum(0).astype(int), point, draws
