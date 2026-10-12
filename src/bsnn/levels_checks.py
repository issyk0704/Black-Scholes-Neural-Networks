"""Checks a morning card must pass before it's posted to Discord.

Each rule holds whatever the market does, so a failure means a bug or bad data,
not an unusual day. A market that fails is left out of the post, and the run
fails, so the problem shows as a red run rather than a wrong card:

- the levels are for the day being posted, from a recent snapshot;
- every chain the market should have went into them (e.g. QQQ and NDX for NQ);
- they were measured at the current futures price;
- call walls sit above that price and put walls below it ("none" is allowed);
- each column has a 1σ move of a sane size, widening from day to week to month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bsnn import instruments
from bsnn.levels import HorizonLevels, MarketLevels, option_sources

MAX_SNAPSHOT_AGE_DAYS = 4  # Friday's close still serves the Tuesday after a Monday holiday
MOVE_BOUNDS = (0.0005, 0.25)  # a 1σ move outside 0.05%-25% of price is a bug, not a market


def problems(levels: MarketLevels, session: pd.Timestamp, prefer: str = "etf") -> list[str]:
    """Why ``levels`` shouldn't be posted for ``session``; empty if they're fine."""
    found = []
    session = pd.Timestamp(session).normalize()
    if levels.session is None or levels.session != session:
        found.append(f"levels are for {_day(levels.session)}, not {_day(session)}")
    age = (session - levels.snapshot_time.tz_localize(None).normalize()).days
    if age > MAX_SNAPSHOT_AGE_DAYS:
        found.append(f"the options were read {age} days before {_day(session)}")
    inst = instruments.resolve(levels.symbol)
    missing = set(option_sources(inst, prefer)) - set(levels.sources or (levels.source,))
    if missing:
        found.append(f"{', '.join(sorted(s.lstrip('^') for s in missing))} options are missing")
    if not np.isfinite(levels.measured_at):
        found.append("no current futures price to measure the levels at")
    if not levels.horizons:
        found.append("no day, week or month levels")
    price = levels.measured_at if np.isfinite(levels.measured_at) else levels.spot
    found += _wall_problems(levels.horizons, price, "")
    found += _wall_problems(levels.own_horizons, price, f"{levels.source.lstrip('^')} ")
    found += _move_problems(levels.horizons)
    return found


def _wall_problems(horizons: list[HorizonLevels], price: float, label: str) -> list[str]:
    found = []
    for h in horizons:
        name = label + "/".join(h.covers)
        if h.call_wall <= price:  # False for "none" (NaN)
            found.append(f"{name} call wall {h.call_wall:,.2f} isn't above price {price:,.2f}")
        if h.put_wall >= price:
            found.append(f"{name} put wall {h.put_wall:,.2f} isn't below price {price:,.2f}")
    return found


def _move_problems(horizons: list[HorizonLevels]) -> list[str]:
    found = []
    low, high = MOVE_BOUNDS
    for h in horizons:
        if not low <= h.move_pct <= high:  # also catches a missing (NaN) move
            found.append(f"{'/'.join(h.covers)} 1σ of {h.move_pct:.2%} is outside {low:.2%}-{high:.0%}")
    moves = [h.move_pct for h in horizons]
    if any(later < earlier for earlier, later in zip(moves, moves[1:])):
        found.append("1σ doesn't widen from day to week to month")
    return found


def _day(day) -> str:
    return "an unknown day" if day is None else f"{pd.Timestamp(day):%a %d %b}"
