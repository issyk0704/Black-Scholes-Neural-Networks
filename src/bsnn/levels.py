"""Daily gamma levels and implied moves for the futures we trade, from saved option snapshots.

Shared by the Moves & gamma tab and ``bsnn-levels``, which posts them to Discord.
Levels come from a whole session's snapshot: open interest only updates once a
day, so the previous session's chain is the most current one before the open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bsnn import analytics, instruments, market_data
from bsnn.features import Filters, build_dataset

# Every strike with a usable quote; gamma far from spot is tiny, so a wide range costs nothing.
# min_days of 0.01 (about 15 minutes) keeps 0DTE contracts in until shortly before the close.
FILTERS = Filters(min_days=0.01, min_moneyness=0.5, max_moneyness=2.0, max_relative_spread=1.0, min_price=0.01)
DEFAULT_MARKETS = ("NQ", "ES", "YM")
SOURCES = ("etf", "index")  # which options to read first: QQQ/SPY/DIA, or NDX/SPX


def futures_ratio(inst: instruments.Instrument, raw: pd.DataFrame) -> float:
    """Futures price / option underlying price when the snapshot was taken (e.g. NQ / QQQ).

    Uses the futures price recorded with the snapshot. Older snapshots don't have
    one, so they fall back to the futures' daily close, which is taken at a
    different time from the underlying's price and so is only approximate.
    """
    spot = float(raw["underlyingPrice"].iloc[0])
    if "futuresPrice" in raw and np.isfinite(raw["futuresPrice"].iloc[0]):
        return float(raw["futuresPrice"].iloc[0]) / spot
    history = market_data.get_history(inst.price_ticker, refresh=True)
    return market_data.unadjusted_close(history, raw["snapshotTime"].iloc[0]) / spot


def option_sources(inst: instruments.Instrument, prefer: str = "etf") -> list[str]:
    """Where a market's levels can come from, in the order to try them.

    ``"etf"`` reads QQQ / SPY / DIA first; ``"index"`` reads NDX / SPX first, where
    most index-options gamma sits. Either falls back to the other.
    """
    if prefer not in SOURCES:
        raise ValueError(f"Unknown source {prefer!r}; choose from {', '.join(SOURCES)}")
    order = (inst.options_proxy, inst.index_options) if prefer == "etf" else (inst.index_options, inst.options_proxy)
    return [t for t in order if t]


def snapshot_date(path: Path) -> pd.Timestamp:
    return pd.Timestamp(market_data.snapshot_label(path).split()[-1])


def latest_snapshot(ticker: str, before=None, directory: Path | None = None) -> Path | None:
    """The newest saved snapshot for ``ticker``, dated before ``before`` if given."""
    files = [p for p in market_data.list_option_snapshots(directory)
             if p.name.split("_options_")[0] == ticker.upper()]
    if before is not None:
        files = [p for p in files if snapshot_date(p) < pd.Timestamp(before).normalize()]
    return files[-1] if files else None


@dataclass
class MarketLevels:
    symbol: str  # the futures, e.g. NQ
    source: str  # the option underlying the levels came from, e.g. ^NDX
    snapshot_time: pd.Timestamp  # New York time
    spot: float  # option underlying at the snapshot
    ratio: float  # futures / underlying on the snapshot day; NaN if unknown
    net_gamma: float  # $ per 1% move
    flip: float
    call_wall: float
    put_wall: float
    one_day_move_pct: float

    def in_futures(self, level: float) -> float:
        return level * self.ratio


def compute_levels(symbol: str, raw: pd.DataFrame) -> MarketLevels | None:
    """Levels from one saved chain, or None if it has no usable quotes or open interest."""
    dataset = build_dataset(raw, FILTERS)
    if dataset.empty or dataset["open_interest"].sum() == 0:
        return None
    day = analytics.one_day_move(analytics.implied_moves(dataset))
    gamma = analytics.gamma_levels(dataset)
    inst = instruments.resolve(symbol)
    try:
        ratio = futures_ratio(inst, raw) if inst else np.nan
    except Exception:
        ratio = np.nan
    return MarketLevels(
        symbol=symbol, source=raw["ticker"].iloc[0],
        snapshot_time=pd.Timestamp(raw["snapshotTime"].iloc[0]).tz_convert(market_data.MARKET_TZ),
        spot=float(dataset["S"].iloc[0]), ratio=ratio, net_gamma=gamma["net"], flip=gamma["flip"], call_wall=gamma["call_wall"],
        put_wall=gamma["put_wall"], one_day_move_pct=day["move_pct"] if day else np.nan)


@dataclass
class ZeroDteLevels(MarketLevels):
    """Levels from a single expiry, normally today's (0DTE), read from a live chain."""

    expiry: str = ""
    is_today: bool = True  # False when the market has no expiry today and the nearest one is used
    straddle: float = np.nan  # at-the-money straddle: the expected move to the expiry, in underlying units
    one_sigma: float = np.nan  # S * IV * sqrt(time left), in underlying units
    top_calls: list[tuple[float, float]] | None = None  # (strike, contracts traded today), busiest first
    top_puts: list[tuple[float, float]] | None = None


def compute_zero_dte(symbol: str, raw: pd.DataFrame, top: int = 3) -> ZeroDteLevels | None:
    """0DTE gamma levels, rest-of-day move and busiest strikes from a live chain."""
    dataset = build_dataset(raw, FILTERS)
    front, is_today = analytics.front_expiry(dataset)
    if front.empty or front["open_interest"].sum() == 0:
        return None
    gamma = analytics.gamma_levels(front)
    moves = analytics.implied_moves(front)
    volume = analytics.volume_by_strike(front)
    inst = instruments.resolve(symbol)
    try:
        ratio = futures_ratio(inst, raw) if inst else np.nan
    except Exception:
        ratio = np.nan
    move = moves.iloc[0] if not moves.empty else None
    return ZeroDteLevels(
        symbol=symbol, source=raw["ticker"].iloc[0],
        snapshot_time=pd.Timestamp(raw["snapshotTime"].iloc[0]).tz_convert(market_data.MARKET_TZ),
        spot=float(front["S"].iloc[0]), ratio=ratio, net_gamma=gamma["net"], flip=gamma["flip"],
        call_wall=gamma["call_wall"], put_wall=gamma["put_wall"],
        one_day_move_pct=move["move_pct"] if move is not None else np.nan,
        expiry=str(front["expiry"].iloc[0]), is_today=is_today,
        straddle=move["straddle"] if move is not None else np.nan,
        one_sigma=move["move"] if move is not None else np.nan,
        top_calls=[(k, v) for k, v in volume["call"].nlargest(top).items() if v > 0],
        top_puts=[(k, v) for k, v in volume["put"].nlargest(top).items() if v > 0])


def live_zero_dte(symbol: str, prefer: str = "etf") -> ZeroDteLevels | None:
    """Fetch a live chain now and return its 0DTE levels; None if no source has usable quotes.

    The chain isn't saved: intraday reads would only overwrite the day's snapshot.
    """
    inst = instruments.resolve(symbol)
    if inst is None:
        raise ValueError(f"{symbol} isn't on the watchlist")
    for source in option_sources(inst, prefer):
        raw = market_data.fetch_option_chain(source, save=False)
        if market_data.snapshot_problem(raw) is None:
            levels = compute_zero_dte(symbol, raw)
            if levels is not None:
                return levels
    return None


def levels_for(symbol: str, before=None, directory: Path | None = None, prefer: str = "etf") -> MarketLevels | None:
    """Levels for a watchlist market from its newest usable snapshot (see :func:`option_sources`)."""
    inst = instruments.resolve(symbol)
    if inst is None:
        raise ValueError(f"{symbol} isn't on the watchlist")
    for source in option_sources(inst, prefer):
        path = latest_snapshot(source, before, directory)
        if path is not None:
            levels = compute_levels(symbol, market_data.load_option_snapshots([path]))
            if levels is not None:
                return levels
    return None
