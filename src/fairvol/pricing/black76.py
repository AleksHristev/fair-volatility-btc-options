from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)
FloatArray = NDArray[np.float64]


def _arr(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def norm_pdf(x: ArrayLike) -> FloatArray:
    x = _arr(x)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def d1_d2(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike) -> tuple[FloatArray, FloatArray]:
    F, K, T, sigma = map(_arr, (F, K, T, sigma))
    valid = (F > 0) & (K > 0) & (T > 0) & (sigma > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_sqrt_t = sigma * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
    d1 = np.where(valid, d1, np.nan)
    d2 = np.where(valid, d2, np.nan)
    return d1, d2


def price_usd(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike,
              is_call: ArrayLike, rate: ArrayLike = 0.0) -> FloatArray:
    F, K, T = _arr(F), _arr(K), _arr(T)
    is_call = np.asarray(is_call, dtype=bool)
    d1, d2 = d1_d2(F, K, T, sigma)
    df = np.exp(-_arr(rate) * T)
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return df * np.where(is_call, call, put)


def price_inverse(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike,
                  is_call: ArrayLike, rate: ArrayLike = 0.0,
                  divisor: ArrayLike | None = None) -> FloatArray:
    div = _arr(F) if divisor is None else _arr(divisor)
    return price_usd(F, K, T, sigma, is_call, rate) / div


def delta_forward(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike,
                  is_call: ArrayLike, rate: ArrayLike = 0.0) -> FloatArray:
    T = _arr(T)
    d1, _ = d1_d2(F, K, T, sigma)
    df = np.exp(-_arr(rate) * T)
    return df * np.where(np.asarray(is_call, dtype=bool), ndtr(d1), ndtr(d1) - 1.0)


def gamma_forward(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike,
                  rate: ArrayLike = 0.0) -> FloatArray:
    F, T, sigma = _arr(F), _arr(T), _arr(sigma)
    d1, _ = d1_d2(F, K, T, sigma)
    df = np.exp(-_arr(rate) * T)
    return df * norm_pdf(d1) / (F * sigma * np.sqrt(T))


def vega_usd(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike,
             rate: ArrayLike = 0.0) -> FloatArray:
    F, T = _arr(F), _arr(T)
    d1, _ = d1_d2(F, K, T, sigma)
    df = np.exp(-_arr(rate) * T)
    return df * F * np.sqrt(T) * norm_pdf(d1)


def vega_inverse_per_volpt(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike,
                           rate: ArrayLike = 0.0,
                           divisor: ArrayLike | None = None) -> FloatArray:
    div = _arr(F) if divisor is None else _arr(divisor)
    return vega_usd(F, K, T, sigma, rate) / div / 100.0


def theta_usd_per_day(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike,
                      is_call: ArrayLike, rate: ArrayLike = 0.0,
                      day_count: float = 365.0) -> FloatArray:
    F, K, T, sigma, r = map(_arr, (F, K, T, sigma, rate))
    d1, d2 = d1_d2(F, K, T, sigma)
    df = np.exp(-r * T)
    decay = -df * F * norm_pdf(d1) * sigma / (2.0 * np.sqrt(T))
    carry = r * price_usd(F, K, T, sigma, is_call, r)
    return (decay + carry) / day_count


def implied_vol(price: ArrayLike, F: ArrayLike, K: ArrayLike, T: ArrayLike,
                is_call: ArrayLike, rate: ArrayLike = 0.0,
                divisor: ArrayLike | None = None,
                bounds: tuple[float, float] = (0.01, 10.0),
                tol: float = 1e-10, max_iter: int = 200,
                min_time_value: float = 1e-12) -> FloatArray:
    price, F, K, T = map(_arr, (price, F, K, T))
    shape = np.broadcast(price, F, K, T, np.asarray(is_call)).shape
    price, F, K, T = (np.broadcast_to(a, shape) for a in (price, F, K, T))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), shape)
    div = F if divisor is None else np.broadcast_to(_arr(divisor), shape)
    rate = np.broadcast_to(_arr(rate), shape)

    lo = np.full(shape, bounds[0])
    hi = np.full(shape, bounds[1])
    p_lo = price_inverse(F, K, T, lo, is_call, rate, div)
    p_hi = price_inverse(F, K, T, hi, is_call, rate, div)
    ok = np.isfinite(price) & (price > 0) & np.isfinite(p_lo) & (price >= p_lo) & (price <= p_hi)
    with np.errstate(invalid="ignore"):
        intrinsic = np.exp(-rate * T) * np.where(is_call, np.maximum(F - K, 0.0),
                                                 np.maximum(K - F, 0.0)) / div
    ok &= (price - intrinsic) > min_time_value

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = price_inverse(F, K, T, mid, is_call, rate, div)
        above = p_mid > price
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
        if np.nanmax(np.where(ok, hi - lo, 0.0), initial=0.0) < tol:
            break
    out = 0.5 * (lo + hi)
    return np.where(ok, out, np.nan)


def implied_forward(price: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike,
                    is_call: ArrayLike, rate: ArrayLike = 0.0,
                    divisor: ArrayLike | None = None,
                    bracket: tuple[ArrayLike, ArrayLike] | None = None,
                    tol_rel: float = 1e-12, max_iter: int = 200) -> FloatArray:
    price, K, T, sigma = map(_arr, (price, K, T, sigma))
    shape = np.broadcast(price, K, T, sigma, np.asarray(is_call)).shape
    price, K, T, sigma = (np.broadcast_to(a, shape) for a in (price, K, T, sigma))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), shape)
    rate = np.broadcast_to(_arr(rate), shape)
    if bracket is None:
        lo, hi = 0.2 * K, 5.0 * K
    else:
        lo, hi = (np.broadcast_to(_arr(b), shape).copy() for b in bracket)
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)

    def f(Fv: FloatArray) -> FloatArray:
        div = Fv if divisor is None else np.broadcast_to(_arr(divisor), shape)
        return price_inverse(Fv, K, T, sigma, is_call, rate, div)

    f_lo, f_hi = f(lo), f(hi)
    inc = is_call
    ok = np.isfinite(price) & (price > 0) & np.isfinite(f_lo) & np.isfinite(f_hi)
    ok &= np.where(inc, (price >= f_lo) & (price <= f_hi), (price <= f_lo) & (price >= f_hi))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        go_down = np.where(inc, fm > price, fm < price)
        hi = np.where(go_down, mid, hi)
        lo = np.where(go_down, lo, mid)
        if np.nanmax(np.where(ok, (hi - lo) / mid, 0.0), initial=0.0) < tol_rel:
            break
    return np.where(ok, 0.5 * (lo + hi), np.nan)
