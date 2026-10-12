"""Gamma levels and implied moves for the futures we trade, from saved option snapshots.

Each market gets levels for the day, the week and the month ahead, each from the
options expiring within that horizon, with the move priced over it.

A market's gamma levels add up all its option chains: QQQ and NDX for NQ, SPY and
SPX for ES (SPX holds most S&P options positioning), DIA alone for YM. The other
chains are restated in the first chain's prices (see analytics.in_units_of), so the
levels convert to futures at that chain's ratio. The first chain's own levels are
kept too, in its own prices.

Shared by the Moves & gamma tab and ``bsnn-levels``, which posts them to Discord.
Levels come from the previous session's snapshot. Open interest is published
overnight, so the daily post swaps in the morning's (see refresh_open_interest).
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass, field
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


def horizon_ends(session) -> dict[str, pd.Timestamp]:
    """The last day of each horizon that starts with ``session``: the session itself,
    that week's Friday and the month's last weekday (holidays aren't known)."""
    day = pd.Timestamp(session).normalize()
    return {"day": day, "week": day + pd.Timedelta(days=4 - day.weekday()), "month": pd.offsets.BMonthEnd().rollforward(day)}


def next_session(after) -> pd.Timestamp:
    """The first weekday after ``after``."""
    return pd.Timestamp(np.busday_offset(pd.Timestamp(after).date(), 1, roll="forward"))


MARKET_CLOSE = dt.time(16, 0)  # New York


def session_for(snapshot_time: pd.Timestamp) -> pd.Timestamp:
    """The trading day a chain's levels are for: the snapshot's own day if it was read
    before that day's close (a live chain), otherwise the next weekday."""
    day = pd.Timestamp(snapshot_time.date())
    if day.weekday() >= 5:
        return pd.Timestamp(np.busday_offset(day.date(), 0, roll="forward"))
    return day if snapshot_time.time() < MARKET_CLOSE else next_session(day)


@dataclass
class HorizonLevels:
    """Gamma levels from the options expiring within one horizon, and the move priced over it."""

    covers: tuple[str, ...]  # "day", "week" and/or "month": several when they end together (e.g. a Friday's day and week)
    end: pd.Timestamp  # last day of the horizon
    net_gamma: float
    flip: float
    call_wall: float
    put_wall: float
    move_pct: float  # one-standard-deviation move over the horizon's trading days
    nearest_expiry: str = ""  # set when nothing expires within the horizon and the nearest expiry was used


def compute_horizons(dataset: pd.DataFrame, session, moves: pd.DataFrame | None = None) -> list[HorizonLevels]:
    """Levels for the session, its week and its month, each from the options expiring by then.

    Gamma from contracts that expire within a horizon is what dealers must hedge
    until that candle closes; longer-dated positions belong to the later horizons.
    ``moves`` (from :func:`analytics.implied_moves`) default to the dataset's own.
    """
    session = pd.Timestamp(session).normalize()
    moves = analytics.implied_moves(dataset) if moves is None else moves
    out = []
    for horizon, end in horizon_ends(session).items():
        if out and out[-1].end == end:
            out[-1].covers += (horizon,)
            continue
        contracts, inside = analytics.expiring_between(dataset, session, end)
        if contracts.empty or contracts["open_interest"].sum() == 0:
            continue
        gamma = analytics.gamma_levels(contracts)
        move = analytics.horizon_move(moves, session, end) if not moves.empty else None
        out.append(HorizonLevels(covers=(horizon,), end=end, net_gamma=gamma["net"], flip=gamma["flip"],
                                 call_wall=gamma["call_wall"], put_wall=gamma["put_wall"],
                                 move_pct=move["move_pct"] if move else np.nan,
                                 nearest_expiry="" if inside else str(contracts["expiry"].iloc[0])))
    return out


@dataclass
class MarketLevels:
    symbol: str  # the futures, e.g. NQ
    source: str  # the option underlying whose prices the levels are in, e.g. QQQ
    snapshot_time: pd.Timestamp  # New York time
    spot: float  # option underlying at the snapshot
    ratio: float  # futures / underlying on the snapshot day; NaN if unknown
    net_gamma: float  # $ per 1% move
    flip: float
    call_wall: float
    put_wall: float
    one_day_move_pct: float
    open_interest_time: pd.Timestamp | None = None  # New York; set when newer open interest replaced the snapshot's
    session: pd.Timestamp | None = None  # the trading day the levels are for
    horizons: list[HorizonLevels] = field(default_factory=list)  # today, this week, this month
    sources: tuple[str, ...] = ()  # every chain added into the levels, e.g. ("QQQ", "^NDX"); source first
    measured_at: float = np.nan  # underlying price the gamma levels were measured at, when not the snapshot's
    own_horizons: list[HorizonLevels] = field(default_factory=list)  # source's chain alone, in its prices

    def in_futures(self, level: float) -> float:
        return level * self.ratio


MIN_OI_CHANGED = 0.10  # below this share of changed contracts, the newer chain hasn't had its overnight update


def refresh_open_interest(raw: pd.DataFrame, newer: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Replace a saved chain's open interest with a newer chain's.

    Open interest is published overnight, so the morning after a snapshot the chain
    carries that session's full positioning, while its quotes are still blank before
    the open. Contracts that have expired since the snapshot drop out. If the newer
    chain hasn't been updated yet, the snapshot comes back unchanged with None.
    """
    if newer.empty:
        return raw, None
    latest = newer.drop_duplicates("contractSymbol").set_index("contractSymbol")["openInterest"]
    kept = raw[raw["contractSymbol"].isin(latest.index)].copy()
    if kept.empty:
        return raw, None
    fresh = kept["contractSymbol"].map(latest).fillna(0.0)
    if (fresh != kept["openInterest"].fillna(0.0)).mean() < MIN_OI_CHANGED:
        return raw, None
    kept["openInterest"] = fresh
    return kept, pd.Timestamp(newer["snapshotTime"].iloc[0]).tz_convert(market_data.MARKET_TZ)


def usable_dataset(raw: pd.DataFrame) -> pd.DataFrame | None:
    """The chain's usable contracts, or None if none have quotes and open interest."""
    dataset = build_dataset(raw, FILTERS)
    return None if dataset.empty or dataset["open_interest"].sum() == 0 else dataset


def combine(dataset: pd.DataFrame, others: list[pd.DataFrame]) -> pd.DataFrame:
    """``dataset`` plus the ``others`` (usable datasets of the same market), restated in its prices."""
    spot = float(dataset["S"].iloc[0])
    return pd.concat([dataset] + [analytics.in_units_of(o, spot) for o in others], ignore_index=True)


def compute_levels(symbol: str, raw: pd.DataFrame, open_interest_time: pd.Timestamp | None = None,
                   session=None, others: list[pd.DataFrame] = (), price_now: float | None = None) -> MarketLevels | None:
    """Levels from a saved chain plus ``others`` (raw chains for the same market and day),
    or None if the first chain has no usable quotes or open interest.

    Gamma levels add up every chain; the implied moves, the futures ratio and the
    prices come from the first. ``session`` is the trading day the levels are for (by
    default from :func:`session_for`); the day, week and month horizons start from it.

    ``price_now`` is the current futures price. When given, the gamma levels are measured
    there (converted at the snapshot's ratio) rather than at the snapshot's price: walls
    are the most gamma above and below where price is now, not where it closed. The
    contracts keep their vols and times to expiry.
    """
    own = usable_dataset(raw)
    if own is None:
        return None
    extra = [d for d in (usable_dataset(o) for o in others) if d is not None]
    dataset = combine(own, extra)
    moves = analytics.implied_moves(own)
    day = analytics.one_day_move(moves)
    inst = instruments.resolve(symbol)
    try:
        ratio = futures_ratio(inst, raw) if inst else np.nan
    except Exception:
        ratio = np.nan
    measured_at = price_now / ratio if price_now and np.isfinite(ratio) else np.nan
    if np.isfinite(measured_at):
        dataset, own = dataset.assign(S=measured_at), own.assign(S=measured_at)
    gamma = analytics.gamma_levels(dataset)
    snapshot_time = pd.Timestamp(raw["snapshotTime"].iloc[0]).tz_convert(market_data.MARKET_TZ)
    session = session_for(snapshot_time) if session is None else pd.Timestamp(session).normalize()
    horizons = compute_horizons(dataset, session, moves)
    return MarketLevels(
        symbol=symbol, source=raw["ticker"].iloc[0], snapshot_time=snapshot_time,
        spot=float(raw["underlyingPrice"].iloc[0]), ratio=ratio, net_gamma=gamma["net"], flip=gamma["flip"], call_wall=gamma["call_wall"],
        put_wall=gamma["put_wall"], one_day_move_pct=day["move_pct"] if day else np.nan,
        open_interest_time=open_interest_time, session=session, horizons=horizons,
        sources=tuple(d["ticker"].iloc[0] for d in [own] + extra), measured_at=measured_at,
        own_horizons=compute_horizons(own, session, moves) if extra else horizons)


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
        problem = market_data.snapshot_problem(raw)
        if problem is None:
            levels = compute_zero_dte(symbol, raw)
            if levels is not None:
                return levels
            problem = "no open interest in the front expiry"
        print(f"{symbol}: {source} chain not used: {problem}", file=sys.stderr)
    return None


def read_snapshot(symbol: str, source: str, path: Path, refresh_oi: bool) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """A saved chain, or with ``refresh_oi`` the freshest data there is.

    The live chain is fetched: with live quotes (09:45-20:00 New York) it replaces the
    snapshot outright, so prices, vols and open interest are all as of now. Before the
    open its quotes are blank, so only its open interest is taken (see :func:`refresh_open_interest`).
    """
    raw = market_data.load_option_snapshots([path])
    if not refresh_oi:
        return raw, None
    try:
        live = market_data.fetch_option_chain(source, save=False)
    except Exception as exc:
        print(f"{symbol}: kept the snapshot's open interest ({source} fetch failed: {exc})", file=sys.stderr)
        return raw, None
    if market_data.snapshot_problem(live) is None:
        print(f"{symbol}: using the live {source} chain", file=sys.stderr)
        return live, None
    raw, oi_time = refresh_open_interest(raw, live)
    if oi_time is None:
        print(f"{symbol}: {source} open interest not updated yet; using the snapshot's", file=sys.stderr)
    return raw, oi_time


def same_day_chains(symbol: str, sources: list[str], day: pd.Timestamp, before, directory: Path | None,
                    refresh_oi: bool) -> list[pd.DataFrame]:
    """The ``sources``' chains saved on ``day``. A chain from another day would mix two sessions' positioning."""
    chains = []
    for source in sources:
        path = latest_snapshot(source, before, directory)
        if path is None or snapshot_date(path) != day:
            print(f"{symbol}: no {source} snapshot from {day:%Y-%m-%d}; levels leave it out", file=sys.stderr)
            continue
        chains.append(read_snapshot(symbol, source, path, refresh_oi)[0])
    return chains


def levels_for(symbol: str, before=None, directory: Path | None = None, prefer: str = "etf",
               refresh_oi: bool = False, price_now: float | None = None) -> MarketLevels | None:
    """Levels for a watchlist market from its newest usable snapshots (see :func:`option_sources`).

    The first source with a usable snapshot sets the prices; the other sources' chains
    from the same day are added into the gamma levels. With ``refresh_oi``, the live
    chains are read (see :func:`read_snapshot`), and with ``price_now`` (the futures
    price) the levels are measured at the current price (see :func:`compute_levels`).
    The levels are for ``before`` (rolled forward to a weekday), or for the weekday
    after the snapshot when ``before`` isn't given.
    """
    session = pd.Timestamp(np.busday_offset(pd.Timestamp(before).date(), 0, roll="forward")) if before is not None else None
    inst = instruments.resolve(symbol)
    if inst is None:
        raise ValueError(f"{symbol} isn't on the watchlist")
    sources = option_sources(inst, prefer)
    for i, source in enumerate(sources):
        path = latest_snapshot(source, before, directory)
        if path is not None:
            raw, oi_time = read_snapshot(symbol, source, path, refresh_oi)
            others = same_day_chains(symbol, sources[i + 1:], snapshot_date(path), before, directory, refresh_oi)
            levels = compute_levels(symbol, raw, oi_time, session, others, price_now)
            if levels is not None:
                return levels
    return None
