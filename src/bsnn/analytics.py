"""Implied moves and dealer gamma levels from one option-chain snapshot.

Both take a dataset from :func:`bsnn.features.build_dataset` for a single
underlying: one row per quoted contract, with implied vol solved from the mid.

**Implied move.** What the options market is pricing for the size of the move
(not its direction) by each expiry:

- the at-the-money straddle (call + put), roughly the expected absolute move;
- the one-standard-deviation move, S * IV * sqrt(T), the range price stays
  inside about 68% of the time if returns were normal. The straddle is about
  0.8 of it.

**Gamma exposure (GEX).** How much stock dealers must buy or sell per 1% move to
stay hedged, summed over open interest. It uses the common simplifying
assumption that dealers are long the calls and short the puts customers trade.
That isn't always true, so the levels are a guide, not a fact:

- positive net gamma: dealers sell rallies and buy dips, which damps moves;
- negative net gamma: dealers chase the move, which amplifies it;
- call wall / put wall: strikes with the most call / put gamma, often acting as
  resistance / support;
- gamma flip: the price at which net gamma changes sign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bsnn import pricing

TRADING_DAYS = 252
CONTRACT_SIZE = 100  # shares (or index units) per contract


def implied_moves(df: pd.DataFrame, max_atm_distance: float = 0.02) -> pd.DataFrame:
    """One row per expiry: ATM strike and IV, straddle, one-sigma move and its range.

    An expiry is skipped when no strike within ``max_atm_distance`` of spot has
    both a call and a put quoted; a far-away "ATM" strike would give a meaningless move.
    """
    rows = []
    for expiry, group in df.groupby("expiry"):
        calls = group[group["is_call"] == 1].set_index("K")
        puts = group[group["is_call"] == 0].set_index("K")
        both = calls.index.intersection(puts.index)
        S, T = group["S"].iloc[0], group["time_to_expiry"].iloc[0]
        if both.empty or np.abs(both - S).min() > max_atm_distance * S:
            continue
        strike = both[np.abs(both - S).argmin()]
        atm_iv = (calls.at[strike, "market_iv"] + puts.at[strike, "market_iv"]) / 2
        straddle = calls.at[strike, "mid"] + puts.at[strike, "mid"]
        move = S * atm_iv * np.sqrt(T)
        rows.append({"expiry": expiry, "days": T * 365, "spot": S, "atm_strike": strike, "atm_iv": atm_iv,
                     "straddle": straddle, "straddle_pct": straddle / S, "move": move, "move_pct": move / S,
                     "low": S - move, "high": S + move})
    return pd.DataFrame(rows).sort_values("days", ignore_index=True) if rows else pd.DataFrame()


def one_day_move(moves: pd.DataFrame, min_days: float = 1.0) -> dict | None:
    """The move priced for one trading day, from the nearest expiry at least ``min_days`` out.

    The same-day expiry is skipped: its implied vol is dominated by the hours left.
    """
    eligible = moves[moves["days"] >= min_days]
    if eligible.empty:
        return None
    row = eligible.iloc[0]
    move = row["spot"] * row["atm_iv"] / np.sqrt(TRADING_DAYS)
    return {"expiry": row["expiry"], "atm_iv": row["atm_iv"], "move": move, "move_pct": move / row["spot"],
            "low": row["spot"] - move, "high": row["spot"] + move}


def _gex(df: pd.DataFrame, spot) -> np.ndarray:
    """Signed dollar gamma per 1% move for each contract, if the underlying were at ``spot``."""
    gamma = np.asarray(pricing.greeks(spot, df["K"].to_numpy(), df["time_to_expiry"].to_numpy(),
                                      df["risk_free_rate"].to_numpy(), df["market_iv"].to_numpy(),
                                      df["dividend_yield"].to_numpy(), df["is_call"].to_numpy() > 0)["gamma"])
    sign = np.where(df["is_call"].to_numpy() > 0, 1.0, -1.0)
    return sign * gamma * df["open_interest"].to_numpy() * CONTRACT_SIZE * np.asarray(spot) ** 2 * 0.01


def within_days(df: pd.DataFrame, max_days: float | None) -> pd.DataFrame:
    return df if max_days is None else df[df["time_to_expiry"] * 365 <= max_days]


def gamma_by_strike(df: pd.DataFrame) -> pd.DataFrame:
    """Call, put and net gamma exposure ($ per 1% move) at each strike, at the current spot."""
    gex = df.assign(gex=_gex(df, df["S"].to_numpy()))
    calls = gex[gex["is_call"] == 1].groupby("K")["gex"].sum()
    puts = gex[gex["is_call"] == 0].groupby("K")["gex"].sum()
    out = pd.concat({"call": calls, "put": puts}, axis=1).fillna(0.0)
    return out.assign(net=out["call"] + out["put"]).sort_index()


def gamma_curve(df: pd.DataFrame, spots: np.ndarray) -> np.ndarray:
    """Total net gamma exposure if the underlying moved to each price in ``spots``."""
    return np.array([_gex(df, s).sum() for s in spots])


def gamma_levels(df: pd.DataFrame, width: float = 0.15, points: int = 301) -> dict:
    """Net gamma, call wall, put wall and gamma flip for the contracts in ``df``."""
    spot = float(df["S"].iloc[0])
    by_strike = gamma_by_strike(df)
    spots = spot * np.linspace(1 - width, 1 + width, points)
    curve = gamma_curve(df, spots)
    return {
        "spot": spot,
        "net": float(by_strike["net"].sum()),
        "call_wall": float(by_strike["call"].idxmax()) if by_strike["call"].max() > 0 else np.nan,
        "put_wall": float(by_strike["put"].idxmin()) if by_strike["put"].min() < 0 else np.nan,
        "flip": _zero_crossing_nearest(spots, curve, spot),
        "curve": (spots, curve),
        "by_strike": by_strike,
    }


def _zero_crossing_nearest(x: np.ndarray, y: np.ndarray, target: float) -> float:
    """Where ``y`` crosses zero, linearly interpolated; the crossing closest to ``target``."""
    crossings = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    if crossings.size == 0:
        return np.nan
    roots = x[crossings] - y[crossings] * (x[crossings + 1] - x[crossings]) / (y[crossings + 1] - y[crossings])
    return float(roots[np.abs(roots - target).argmin()])
