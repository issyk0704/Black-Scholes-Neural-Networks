"""How the posted levels did: a scorecard for last week (on Monday's card) and last month
(on the month's first card), plus the running total.

Each morning card (the day's "scheduled" record in ``data/levels``) is scored against the
New York cash session (09:30-16:00) once its horizon has finished:

- ceiling held: price never traded above the call wall;
- floor held: price never traded below the put wall;
- closed inside the expected range (the straddle, 0.8 of one standard deviation).

Each result is set against chance: how often a level that far from the posting price would
hold anyway if price wandered at random with the move the options priced (2·Φ(d/σ) − 1), and
how often a close lands inside ±0.8σ (58%). A level far away holds most days whatever it is,
so "held 80%" means nothing until it beats that.

Scores are saved to ``data/levels_scores.csv`` as horizons finish, because Yahoo only keeps
about 60 days of 5-minute bars.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from bsnn import instruments, market_data
from bsnn.levels_record import STRADDLE_SHARE

SESSION_OPEN, SESSION_CLOSE = "09:30", "15:55"  # 5-minute bars starting in the New York cash session
CHANCE_INSIDE_EXPECTED = float(2 * norm.cdf(STRADDLE_SHARE) - 1)
KEY = ["session", "symbol", "covers"]
COLUMNS = KEY + ["end", "ceiling_held", "ceiling_chance", "floor_held", "floor_chance", "inside_expected"]


# --- Reading the record ----------------------------------------------------------

def morning_cards(directory: Path) -> pd.DataFrame:
    """One row per market and column of each day's first scheduled card (the levels known
    before the session), with the price they were posted at."""
    rows = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["kind"] != "scheduled":
                continue
            for m in record["markets"]:
                for h in m["horizons"]:
                    rows.append({"session": m["session"], "symbol": m["symbol"], "covers": "+".join(h["covers"]),
                                 "end": h["end"], "price": m["futures_price"], "move_pct": h["move_pct"],
                                 "ceiling": h["call_wall_futures"], "floor": h["put_wall_futures"]})
            break  # the first scheduled card of the day
    cards = pd.DataFrame(rows)
    return cards.dropna(subset=["price", "move_pct"]) if not cards.empty else cards


# --- Prices ------------------------------------------------------------------------

def fetch_bars(symbol: str) -> pd.DataFrame:
    """The last 60 days of 5-minute futures bars in the New York cash session."""
    import yfinance as yf

    ticker = instruments.resolve(symbol).price_ticker
    bars = yf.Ticker(ticker).history(period="60d", interval="5m", auto_adjust=False)
    bars.index = bars.index.tz_convert(market_data.MARKET_TZ)
    return bars.between_time(SESSION_OPEN, SESSION_CLOSE)[["High", "Low", "Close"]]


def _finished(bars: pd.DataFrame, end: pd.Timestamp) -> bool:
    """True once the bars cover the horizon's last session to its close (or a later day)."""
    if bars.empty:
        return False
    last = bars.index[-1]
    last_day = last.tz_localize(None).normalize()
    return last_day > end or (last_day == end and last.time() >= pd.Timestamp(SESSION_CLOSE).time())


# --- Scoring -----------------------------------------------------------------------

def _hold(level: float, distance: float, sigma: float, held: bool) -> tuple[float, float]:
    """(held 1/0, chance a level that far away holds); NaN for a missing or wrong-side level."""
    if not np.isfinite(level) or distance <= 0:
        return np.nan, np.nan
    return float(held), float(2 * norm.cdf(distance / sigma) - 1)


def score_card(card: pd.Series, bars: pd.DataFrame) -> dict | None:
    """One card column's result, or None if its horizon hasn't finished or has no bars."""
    session, end = pd.Timestamp(card["session"]), pd.Timestamp(card["end"])
    if not _finished(bars, end):
        return None
    days = bars.index.tz_localize(None).normalize()
    inside = bars[(days >= session) & (days <= end)]
    if inside.empty or days.min() > session:  # the bars don't reach back to the posting day
        return None
    high, low, close = inside["High"].max(), inside["Low"].min(), inside["Close"].iloc[-1]
    price, sigma = card["price"], card["price"] * card["move_pct"]
    expected = sigma * STRADDLE_SHARE
    ceiling = float(card["ceiling"]) if pd.notna(card["ceiling"]) else np.nan
    floor = float(card["floor"]) if pd.notna(card["floor"]) else np.nan
    ceiling_held, ceiling_chance = _hold(ceiling, ceiling - price, sigma, high < ceiling)
    floor_held, floor_chance = _hold(floor, price - floor, sigma, low > floor)
    return {"session": card["session"], "symbol": card["symbol"], "covers": card["covers"], "end": card["end"],
            "ceiling_held": ceiling_held, "ceiling_chance": ceiling_chance,
            "floor_held": floor_held, "floor_chance": floor_chance,
            "inside_expected": float(abs(close - price) <= expected)}


def update_scores(store: Path, records: Path, bars_for=None) -> pd.DataFrame:
    """Score every finished card column not yet in ``store``, save, and return all scores.
    ``bars_for(symbol)`` gives the bars (default :func:`fetch_bars`)."""
    bars_for = bars_for or fetch_bars
    scores = pd.read_csv(store, dtype={"session": str, "end": str}) if store.exists() else pd.DataFrame(columns=COLUMNS)
    cards = morning_cards(records)
    if not cards.empty:
        done = set(map(tuple, scores[KEY].astype(str).to_numpy()))
        todo = cards[[tuple(map(str, k)) not in done for k in cards[KEY].to_numpy()]]
        new = []
        for symbol, group in todo.groupby("symbol"):
            try:
                bars = bars_for(symbol)
            except Exception as exc:
                print(f"Scorecard: no {symbol} bars ({exc})", file=sys.stderr)
                continue
            new += [s for s in (score_card(card, bars) for _, card in group.iterrows()) if s]
        if new:
            scores = pd.concat([scores, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True)
            scores = scores.sort_values(KEY, ignore_index=True)
            store.parent.mkdir(parents=True, exist_ok=True)
            scores.to_csv(store, index=False)
    return scores


# --- When to show it ---------------------------------------------------------------

def scorecard_period(session) -> tuple[str, pd.Timestamp, pd.Timestamp] | None:
    """("month" | "week", first day, last day) of the period to review on ``session``'s card:
    last month on the month's first weekday, else last week on a Monday; None on other days."""
    day = pd.Timestamp(session).normalize()
    first_weekday = pd.Timestamp(np.busday_offset(day.replace(day=1).date(), 0, roll="forward"))
    if day == first_weekday:
        end = day.replace(day=1) - pd.Timedelta(days=1)
        return "month", end.replace(day=1), end
    if day.weekday() == 0:
        return "week", day - pd.Timedelta(days=7), day - pd.Timedelta(days=3)
    return None


# --- The card ------------------------------------------------------------------------

def _count(held: pd.Series, chance: pd.Series) -> tuple[int, int, float]:
    valid = held.notna()
    return int(held[valid].sum()), int(valid.sum()), float(chance[valid].sum())


def _line(name: str, held: pd.Series, chance: pd.Series) -> str:
    hits, n, expected = _count(held, chance)
    if n == 0:
        return f"{name}: no levels to score"
    return f"{name}: **{hits} of {n}** · chance alone: about {expected:.0f}"


def verdict(days: pd.DataFrame) -> str:
    """Whether ceilings and floors together held more often than chance.

    NQ, ES and YM move together, so three markets on one day count as about one test:
    the z-score is divided by the square root of the markets per day.
    """
    held = pd.concat([days["ceiling_held"], days["floor_held"]])
    chance = pd.concat([days["ceiling_chance"], days["floor_chance"]])
    valid = held.notna()
    held, chance = held[valid], chance[valid]
    if len(held) == 0:
        return "nothing scored yet"
    spread = float(np.sqrt((chance * (1 - chance)).sum()))
    markets = max(days["symbol"].nunique(), 1)
    z = (held.sum() - chance.sum()) / spread / np.sqrt(markets) if spread > 0 else 0.0
    gap = held.mean() - chance.mean()
    if z >= 2:
        return f"better than chance ({gap:+.0%})"
    if z <= -2:
        return f"worse than chance ({gap:+.0%})"
    leaning = "slightly better than" if gap > 0.02 else "slightly worse than" if gap < -0.02 else "about the same as"
    return f"{leaning} chance ({gap:+.0%}), not proven yet"


def _horizon_lines(scores: pd.DataFrame, kind: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """How each market's week or month column from the period's first card did."""
    sessions = pd.to_datetime(scores["session"])
    rows = scores[scores["covers"].str.contains(kind) & (sessions >= start) & (sessions <= end)]
    if rows.empty:
        return []
    first = rows[rows["session"] == rows["session"].min()]
    def word(value, good, bad):
        return "none" if pd.isna(value) else good if value else bad
    parts = [f"{r.symbol} ceiling {word(r.ceiling_held, 'held', 'broke')}, floor {word(r.floor_held, 'held', 'broke')}"
             for r in first.itertuples()]
    return [f"{kind.capitalize()} levels from the {pd.Timestamp(first['session'].iloc[0]):%a %d %b} card: " + " · ".join(parts)]


def scorecard_embed(scores: pd.DataFrame, kind: str, start: pd.Timestamp, end: pd.Timestamp) -> dict | None:
    """The review card: the period's day levels, its week or month levels, and the running total."""
    if scores.empty:
        return None
    sessions = pd.to_datetime(scores["session"])
    day_rows = scores[scores["covers"].str.contains("day")]
    period = day_rows[(sessions[day_rows.index] >= start) & (sessions[day_rows.index] <= end)]
    if period.empty:
        return None
    lines = [_line("Ceiling held", period["ceiling_held"], period["ceiling_chance"]),
             _line("Floor held", period["floor_held"], period["floor_chance"]),
             _line("Closed inside expected range", period["inside_expected"],
                   pd.Series(CHANCE_INSIDE_EXPECTED, index=period.index))]
    lines += _horizon_lines(scores, kind, start, end)
    since = pd.Timestamp(day_rows["session"].min())
    lines.append(f"Since {since:%d %b}, the day levels: {verdict(day_rows)}.")
    name = "week" if kind == "week" else "month"
    title = (f"Last {name}'s scorecard ({start:%a %d %b} – {end:%a %d %b})" if kind == "week"
             else f"Last month's scorecard ({start:%B %Y})")
    return {"title": title,
            "description": "\n".join(lines),
            "footer": {"text": "Each morning card's levels against the New York session (09:30-16:00), counted per "
                               "market and day. Chance alone: how often a level that far away holds anyway, given the "
                               "move the options priced."}}
