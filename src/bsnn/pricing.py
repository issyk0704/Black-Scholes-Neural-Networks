"""Black-Scholes-Merton pricing for European options with a continuous dividend yield.

Every function takes scalars or NumPy arrays (broadcast together) and returns a
float for scalar input. ``is_call`` is True for calls and False for puts, and
may also be an array.

The two standard variants are the same formula with a different carry term
(see :func:`carry_yield` and :func:`model_greeks`):

- Black-76, for options on futures: spot is the futures price and the carry
  equals the interest rate, because holding a futures contract costs nothing.
- Garman-Kohlhagen, for FX options: the carry is the foreign interest rate.

Greeks use trading-desk units:

- delta: change in price per $1 move in spot
- gamma: change in delta per $1 move in spot
- vega:  change in price per 1 volatility point (0.01)
- theta: change in price per calendar day
- rho:   change in price per 1 percentage point of interest rate (0.01)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _out(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def validate(S, K, T, sigma) -> None:
    for name, value in (("Spot", S), ("Strike", K), ("Time to expiry", T), ("Volatility", sigma)):
        value = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError(f"{name} must be a positive number")


def _d1_d2(S, K, T, r, sigma, q):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, r, sigma, q=0.0, is_call=True):
    """Option price."""
    validate(S, K, T, sigma)
    S, K, T, r, sigma, q = (np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q))
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    spot_df, strike_df = S * np.exp(-q * T), K * np.exp(-r * T)
    call = spot_df * norm.cdf(d1) - strike_df * norm.cdf(d2)
    put = strike_df * norm.cdf(-d2) - spot_df * norm.cdf(-d1)
    return _out(np.where(is_call, call, put))


def greeks(S, K, T, r, sigma, q=0.0, is_call=True) -> dict:
    """Price and Greeks, keyed ``price``, ``delta``, ``gamma``, ``vega``, ``theta``, ``rho``."""
    validate(S, K, T, sigma)
    S, K, T, r, sigma, q = (np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q))
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    div_df, rate_df = np.exp(-q * T), np.exp(-r * T)
    pdf_d1, sqrt_t = norm.pdf(d1), np.sqrt(T)

    decay = -S * div_df * pdf_d1 * sigma / (2 * sqrt_t)
    call = {
        "price": S * div_df * norm.cdf(d1) - K * rate_df * norm.cdf(d2),
        "delta": div_df * norm.cdf(d1),
        "theta": decay - r * K * rate_df * norm.cdf(d2) + q * S * div_df * norm.cdf(d1),
        "rho": K * T * rate_df * norm.cdf(d2),
    }
    put = {
        "price": K * rate_df * norm.cdf(-d2) - S * div_df * norm.cdf(-d1),
        "delta": -div_df * norm.cdf(-d1),
        "theta": decay + r * K * rate_df * norm.cdf(-d2) - q * S * div_df * norm.cdf(-d1),
        "rho": -K * T * rate_df * norm.cdf(-d2),
    }
    out = {key: np.where(is_call, call[key], put[key]) for key in call}
    out["gamma"] = div_df * pdf_d1 / (S * sigma * sqrt_t)
    out["vega"] = S * div_df * pdf_d1 * sqrt_t / 100
    out["theta"] = out["theta"] / 365
    out["rho"] = out["rho"] / 100
    order = ("price", "delta", "gamma", "vega", "theta", "rho")
    return {key: _out(np.broadcast_to(out[key], np.broadcast(S, K, T, r, sigma, q, is_call).shape))
            for key in order}


MODELS = {
    "bsm": "Black-Scholes-Merton (stocks, ETFs)",
    "black76": "Black-76 (futures)",
    "garman_kohlhagen": "Garman-Kohlhagen (FX)",
}


def carry_yield(model: str, r, carry=0.0):
    """The ``q`` to pass to the Black-Scholes-Merton functions for ``model``.

    ``carry`` is the dividend yield for ``"bsm"`` and the foreign interest rate
    for ``"garman_kohlhagen"``; ``"black76"`` ignores it.
    """
    if model == "black76":
        return r
    if model in ("bsm", "garman_kohlhagen"):
        return carry
    raise ValueError(f"Unknown model {model!r}; choose from {', '.join(MODELS)}")


def model_greeks(model: str, S, K, T, r, sigma, carry=0.0, is_call=True) -> dict:
    """:func:`greeks` under ``model``. For Black-76, ``S`` is the futures price."""
    out = greeks(S, K, T, r, sigma, carry_yield(model, r, carry), is_call)
    if model == "black76":
        # The rate drives both discounting and carry, which cancel except for
        # discounting the whole price: d(price)/dr = -T * price.
        out["rho"] = _out(-np.asarray(T, dtype=float) * np.asarray(out["price"]) / 100)
    return out


def no_arbitrage_bounds(S, K, T, r, q=0.0, is_call=True):
    """(lower, upper) bounds a European option price must lie strictly between."""
    spot_df = np.asarray(S, dtype=float) * np.exp(-np.asarray(q) * T)
    strike_df = np.asarray(K, dtype=float) * np.exp(-np.asarray(r) * T)
    lower = np.where(is_call, np.maximum(spot_df - strike_df, 0.0), np.maximum(strike_df - spot_df, 0.0))
    upper = np.where(is_call, spot_df, strike_df)
    return _out(lower), _out(upper)


def implied_volatility(market_price, S, K, T, r, q=0.0, is_call=True, low=1e-4, high=5.0, iterations=80):
    """Volatility that makes the Black-Scholes price equal ``market_price``.

    Solved by bisection on whole arrays at once (price rises monotonically with
    volatility). Returns NaN where no volatility in ``[low, high]`` fits, e.g. a
    price outside the no-arbitrage bounds.
    """
    target, S, K, T, r, q, is_call = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (market_price, S, K, T, r, q)), np.asarray(is_call, dtype=bool))
    finite = np.isfinite(target) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(q)
    valid = finite & (S > 0) & (K > 0) & (T > 0)
    # Placeholder inputs keep the vectorised maths finite on rows that are already invalid.
    S, K, T = (np.where(valid, a, 1.0) for a in (S, K, T))
    r, q, target = (np.where(valid, a, 0.0) for a in (r, q, target))
    lower, upper = no_arbitrage_bounds(S, K, T, r, q, is_call)
    valid &= (target > lower) & (target < upper)

    def bs(sigma):
        return price(S, K, T, r, sigma, q, is_call)

    lo, hi = np.full(target.shape, float(low)), np.full(target.shape, float(high))
    valid &= (bs(lo) <= target) & (bs(hi) >= target)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        too_high = bs(mid) > target
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return _out(np.where(valid, (lo + hi) / 2, np.nan))


def payoff(spot_at_expiry, K, is_call=True):
    """Option value at expiry."""
    spot_at_expiry = np.asarray(spot_at_expiry, dtype=float)
    return _out(np.where(is_call, np.maximum(spot_at_expiry - K, 0.0), np.maximum(K - spot_at_expiry, 0.0)))
