"""A record of every card posted to Discord, kept for scoring the levels later.

Each post appends one JSON line to ``data/levels/<posting day>.jsonl``: when and
how it was posted, and for each market every level in the option underlying's
prices and in futures points, the 1σ range, and the inputs behind them. The
scheduled workflow commits the folder to bsnn-data, so the history builds up
there. Missing values (a wall "none within ±15%") are stored as null.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from bsnn.levels import HorizonLevels, MarketLevels


def _number(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _time(value) -> str | None:
    return None if value is None else pd.Timestamp(value).isoformat()


def horizon_record(h: HorizonLevels, levels: MarketLevels, price: float | None) -> dict:
    """One column: levels in the underlying's prices, then in futures points, and the 1σ range around ``price``."""
    record = {"covers": list(h.covers), "end": f"{h.end:%Y-%m-%d}", "net_gamma": _number(h.net_gamma),
              "flip": _number(h.flip), "call_wall": _number(h.call_wall), "put_wall": _number(h.put_wall),
              "move_pct": _number(h.move_pct), "nearest_expiry": h.nearest_expiry or None}
    for name in ("flip", "call_wall", "put_wall"):
        record[f"{name}_futures"] = _number(levels.in_futures(getattr(h, name)))
    move = price * h.move_pct if price else math.nan
    record["range_low"], record["range_high"] = _number(price - move) if price else None, _number(price + move) if price else None
    return record


def market_record(levels: MarketLevels, price: float | None) -> dict:
    """Everything a card showed for one market, plus the inputs behind it."""
    return {
        "symbol": levels.symbol, "source": levels.source, "sources": list(levels.sources or (levels.source,)),
        "session": None if levels.session is None else f"{levels.session:%Y-%m-%d}",
        "snapshot_time": _time(levels.snapshot_time), "open_interest_time": _time(levels.open_interest_time),
        "spot": _number(levels.spot), "ratio": _number(levels.ratio), "measured_at": _number(levels.measured_at),
        "futures_price": None if price is None else _number(price), "net_gamma": _number(levels.net_gamma),
        "horizons": [horizon_record(h, levels, price) for h in levels.horizons],
        "own_horizons": [horizon_record(h, levels, None) for h in levels.own_horizons],
    }


def append(directory: Path, day, posted_at: pd.Timestamp, kind: str, all_levels: list[MarketLevels],
           prices: dict[str, float], message_id: str | None = None) -> Path:
    """Add one post to ``directory/<day>.jsonl`` and return the file."""
    path = Path(directory) / f"{pd.Timestamp(day):%Y-%m-%d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"posted_at": _time(posted_at), "kind": kind, "message_id": message_id,
            "markets": [market_record(lv, prices.get(lv.symbol)) for lv in all_levels]}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
    return path
