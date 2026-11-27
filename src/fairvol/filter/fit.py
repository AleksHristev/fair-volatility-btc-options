from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.optimize import minimize

from fairvol.filter.events import Events
from fairvol.filter.kalman import run_filter
from fairvol.filter.params import ParamMap, Spec


@dataclass
class FitResult:
    spec: str
    names: list[str]
    params: np.ndarray
    loglik: float
    n_obs: int
    n_vclip: int
    iterations: int
    n_evals: int
    converged: bool
    message: str
    seconds: float
    at_bound: list[str]
    coord_rounds: int = 0
    coord_max_improvement: float = 0.0
    n_nonfinite: int = 0
    cpu_seconds: float = 0.0
    n_resumes: int = 0
    cpu_seconds_earlier: float = 0.0
    checkpoint_saves: int = 0

    def row(self) -> dict[str, Any]:
        out = {"spec": self.spec, "loglik": self.loglik, "n_obs": self.n_obs,
               "loglik_per_obs": self.loglik / max(self.n_obs, 1), "n_free_params": len(self.names),
               "iterations": self.iterations, "n_evals": self.n_evals, "converged": self.converged,
               "message": self.message, "seconds": round(self.seconds, 1), "n_vclip": self.n_vclip,
               "at_bound": ",".join(self.at_bound), "coord_rounds": self.coord_rounds,
               "coord_max_improvement_ll": self.coord_max_improvement, "n_nonfinite_evals": self.n_nonfinite,
               "cpu_seconds": round(self.cpu_seconds, 1), "n_resumes": self.n_resumes,
               "checkpoint_saves": self.checkpoint_saves}
        for n, v in zip(self.names, self.params):
            out[f"p_{n}"] = v
        return out


def total_loglik(nat: np.ndarray, spec: Spec, runs: list[tuple[Events, int]], consts: np.ndarray,
                 burn_in: int) -> tuple[float, int, int]:
    ll, n, clip = 0.0, 0, 0
    for ev, lik_start in runs:
        a, b, c, _ = run_filter(*ev.arrays(), nat, spec.basis_id, spec.weighting_id, spec.residual,
                                spec.spot, consts, burn_in, lik_start, False)
        ll += a
        n += b
        clip += c
    return ll, n, clip


CKPT_VERSION = 1


def _ns(x, fun, success, message, nit=0) -> SimpleNamespace:
    return SimpleNamespace(x=np.asarray(x, float).copy(), fun=float(fun), success=bool(success),
                           message=str(message), nit=int(nit))


def _write_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def fit_spec(spec: Spec, runs: list[tuple[Events, int]], settings: dict[str, Any], consts: np.ndarray,
             start_nat: np.ndarray | None = None, log=None, max_rounds: int | None = None,
             pm: ParamMap | None = None, ckpt: Path | None = None, ckpt_every: float | None = None) -> FitResult:
    ml = settings["filter"]["ml"]
    burn_in = int(settings["filter"]["burn_in_updates"])
    pm = pm if pm is not None else ParamMap(spec, settings)
    cap = int(ml["coord_check_max_rounds"]) if max_rounds is None else int(max_rounds)
    restarts = int(ml.get("restarts", 0))
    every = float(ml.get("checkpoint_cpu_seconds", 300.0)) if ckpt_every is None else float(ckpt_every)
    cpu0 = time.process_time()
    z0 = pm.to_opt(pm.start_nat if start_nat is None else start_nat)
    bounds = pm.opt_bounds()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    z0 = np.clip(z0, lo, hi)
    h = float(ml["fd_step"])
    counter = {"n": 0, "bad": 0}
    _, n_obs, _ = total_loglik(pm.vector(pm.to_nat(z0)), spec, runs, consts, burn_in)

    key = {"version": CKPT_VERSION, "spec": spec.name, "names": list(pm.names), "z0": [float(v) for v in z0],
           "lo": [float(v) for v in lo], "hi": [float(v) for v in hi], "cap": cap, "restarts": restarts,
           "n_obs": int(n_obs)}
    state = None
    if ckpt is not None and ckpt.exists():
        try:
            st = json.loads(ckpt.read_text())
        except (OSError, ValueError):
            st = None
        if st is not None and st.get("key") == key:
            state = st
        elif log is not None:
            log.warning("checkpoint %s does not match this fit; starting afresh", ckpt)
    prev = state or {}
    n_resumes = int(prev.get("n_resumes", -1)) + 1 if state else 0
    track = {"z": None, "f": np.inf, "last": time.process_time(), "saves": int(prev.get("n_saves", 0))}
    if state:
        track["z"], track["f"] = np.clip(np.array(state["z_best"], float), lo, hi), float(state["f_best"])
    stage: dict[str, Any] = dict(prev.get("stage", {"name": "lbfgs"}))
    nitb = [int(prev.get("nit", 0))]
    t0 = time.time()

    def save(force: bool = False) -> None:
        if ckpt is None or track["z"] is None:
            return
        now = time.process_time()
        if not force and now - track["last"] < every:
            return
        track["last"] = now
        track["saves"] += 1
        _write_atomic(ckpt, {"key": key, "stage": stage, "z_best": [float(v) for v in track["z"]],
                             "f_best": float(track["f"]), "n_evals": int(prev.get("n_evals", 0)) + counter["n"],
                             "n_bad": int(prev.get("n_bad", 0)) + counter["bad"], "nit": nitb[0],
                             "cpu_seconds": float(prev.get("cpu_seconds", 0.0)) + now - cpu0,
                             "wall_seconds": float(prev.get("wall_seconds", 0.0)) + time.time() - t0,
                             "n_resumes": n_resumes, "n_saves": track["saves"],
                             "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def f(z: np.ndarray) -> float:
        counter["n"] += 1
        ll, n, _ = total_loglik(pm.vector(pm.to_nat(z)), spec, runs, consts, burn_in)
        val = -ll / n_obs
        if not np.isfinite(val):
            counter["bad"] += 1
            return 1e30
        if val < track["f"]:
            track["z"], track["f"] = np.array(z, float).copy(), val
        save()
        return val

    def fg(z: np.ndarray) -> tuple[float, np.ndarray]:
        f0 = f(z)
        g = np.empty_like(z)
        for i in range(len(z)):
            zi = z.copy()
            step = h if z[i] + h <= hi[i] else -h
            zi[i] += step
            g[i] = (f(zi) - f0) / step
        return f0, g

    def resume_point(x, fun) -> np.ndarray:
        return track["z"].copy() if (track["z"] is not None and track["f"] < fun) else np.asarray(x, float).copy()

    opts = {"maxiter": int(ml["maxiter"]), "ftol": float(ml["ftol"]), "gtol": float(ml["gtol"])}
    step_c = float(ml["coord_check_step"])
    if stage["name"] in ("lbfgs", "restart"):
        if stage["name"] == "lbfgs":
            res = minimize(fg, z0 if state is None else track["z"], jac=True, method="L-BFGS-B", bounds=bounds,
                           options=opts)
            nitb[0] += int(res.nit)
            d0, first = 0, None
        else:
            res = _ns(stage["x"], stage["fun"], stage["success"], stage["message"])
            d0, first = int(stage["done"]), resume_point(stage["x"], stage["fun"])
        stage = {"name": "restart", "done": d0, "x": [float(v) for v in res.x], "fun": float(res.fun),
                 "success": bool(res.success), "message": str(res.message)}
        save(force=True)
        for r in range(d0, restarts):
            x_start = first if (first is not None and r == d0) else res.x
            res2 = minimize(fg, x_start, jac=True, method="L-BFGS-B", bounds=bounds, options=opts)
            nitb[0] += int(res2.nit)
            if res2.fun <= res.fun:
                res = res2
            stage = {"name": "restart", "done": r + 1, "x": [float(v) for v in res.x], "fun": float(res.fun),
                     "success": bool(res.success), "message": str(res.message)}
            save(force=True)
        best_z, best_f = res.x.copy(), float(res.fun)
        rounds, max_impr = 0, 0.0
    else:
        res = _ns(stage["x"], stage["fun"], stage["success"], stage["message"])
        best_z = resume_point(stage["best_x"], stage["best_f"])
        best_f = min(float(stage["best_f"]), float(track["f"]))
        rounds, max_impr = int(stage["rounds"]), float(stage["max_impr"])

    def coord_stage() -> dict[str, Any]:
        return {"name": "coord", "rounds": rounds, "max_impr": max_impr, "best_x": [float(v) for v in best_z],
                "best_f": best_f, "x": [float(v) for v in res.x], "fun": float(res.fun),
                "success": bool(res.success), "message": str(res.message)}
    stage = coord_stage()
    save(force=True)
    while rounds < cap:
        cand_f, cand_z = best_f, None
        for i in range(len(best_z)):
            for sgn in (1.0, -1.0):
                z = best_z.copy()
                z[i] = min(max(z[i] + sgn * step_c, lo[i]), hi[i])
                if z[i] == best_z[i]:
                    continue
                fz = f(z)
                if fz < cand_f:
                    cand_f, cand_z = fz, z
        if cand_z is None or cand_f >= best_f - 1e-12:
            break
        max_impr = max(max_impr, (best_f - cand_f) * n_obs)
        r3 = minimize(fg, cand_z, jac=True, method="L-BFGS-B", bounds=bounds, options=opts)
        nitb[0] += int(r3.nit)
        best_z, best_f = (r3.x.copy(), float(r3.fun)) if r3.fun < cand_f else (cand_z, cand_f)
        res = r3 if r3.fun < cand_f else res
        rounds += 1
        stage = coord_stage()
        save(force=True)
    res.x = best_z
    res.nit = nitb[0]
    nat = pm.to_nat(res.x)
    ll, n, clip = total_loglik(pm.vector(nat), spec, runs, consts, burn_in)
    tol = 1e-6
    at_bound = [nm for nm, z, a, b in zip(pm.names, res.x, lo, hi) if z <= a + tol or z >= b - tol]
    cpu_prev = float(prev.get("cpu_seconds", 0.0))
    out = FitResult(spec.name, pm.names, nat, float(ll), int(n), int(clip), int(res.nit),
                    int(prev.get("n_evals", 0)) + counter["n"], bool(res.success), str(res.message),
                    float(prev.get("wall_seconds", 0.0)) + time.time() - t0, at_bound, rounds, float(max_impr),
                    int(prev.get("n_bad", 0)) + int(counter["bad"]), cpu_prev + time.process_time() - cpu0,
                    n_resumes, cpu_prev, track["saves"])
    if log is not None:
        log.info("%s: ll/n=%.6f it=%d evals=%d conv=%s (%.0fs)%s %s", spec.name, ll / n, res.nit, out.n_evals,
                 res.success, out.seconds, f" [resumed {n_resumes}x from checkpoint]" if n_resumes else "",
                 dict(zip(pm.names, np.round(nat, 5))))
    return out


HESSIAN_METHOD = "central differences for every term; NaN where -H is not positive definite"


def hessian_se(spec: Spec, nat_free: np.ndarray, runs: list[tuple[Events, int]], settings: dict[str, Any],
               consts: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    burn_in = int(settings["filter"]["burn_in_updates"])
    pm = ParamMap(spec, settings)
    z = pm.to_opt(nat_free)
    h = float(settings["filter"]["ml"]["fd_step"]) * 10.0
    p = len(z)
    E = np.eye(p) * h

    def ll(zz: np.ndarray) -> float:
        return total_loglik(pm.vector(pm.to_nat(zz)), spec, runs, consts, burn_in)[0]

    f0 = ll(z)
    H = np.zeros((p, p))
    for i in range(p):
        H[i, i] = (ll(z + E[i]) - 2 * f0 + ll(z - E[i])) / h ** 2
        for j in range(i + 1, p):
            H[i, j] = H[j, i] = (ll(z + E[i] + E[j]) - ll(z + E[i] - E[j]) - ll(z - E[i] + E[j])
                                 + ll(z - E[i] - E[j])) / (4 * h ** 2)
    eig = np.linalg.eigvalsh(-H)
    info = {"hessian_method": HESSIAN_METHOD, "hessian_step_opt": h, "hessian_neg_definite": bool(eig.min() > 0),
            "hessian_min_eig": float(eig.min()), "hessian_max_eig": float(eig.max()),
            "hessian_n_nonpositive_eig": int((eig <= 0).sum()), "hessian_matrix": H.tolist(),
            "hessian_names": list(pm.names)}
    try:
        cov_z = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan), info
    d = np.diag(cov_z)
    se_z = np.where(d > 0, np.sqrt(np.where(d > 0, d, 1.0)), np.nan)
    jac = np.array([np.exp(zi) if tr == "log" else (pm.scale if tr == "scaled" else 1.0)
                    for zi, (_, _, tr) in zip(z, pm.free)])
    return se_z * jac, info
