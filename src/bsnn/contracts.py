"""Which futures contract the levels use for the equity-index futures (NQ, ES, YM, RTY).

Our rule: stay on the current quarterly contract until the next one's open interest is
higher, then switch. Yahoo's continuous symbols (NQ=F) roll on their own dates instead, and
the two contracts differ by the cost of carry (NQ Mar vs Dec is about 1%, 300+ points), so
following Yahoo's roll would shift every level on the card without warning.

Open interest comes from Yahoo's quote for each contract (e.g. NQZ26.CME). If it can't be
read, the instrument's continuous symbol is used as before.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging

import pandas as pd

from bsnn import instruments

log = logging.getLogger(__name__)

QUARTER_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
MONTHS = {code: month for month, code in QUARTER_CODES.items()}
EXCHANGES = {"NQ": "CME", "ES": "CME", "RTY": "CME", "YM": "CBT"}  # Yahoo's suffix for each root


def expiry(year: int, month: int) -> dt.date:
    """Equity-index futures expire on the third Friday of the contract month."""
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)


def quarterly_contracts(root: str, day: dt.date, count: int = 2) -> list[str]:
    """The next ``count`` quarterly contracts not yet expired on ``day``, nearest first, e.g.
    ["NQZ26.CME", "NQH27.CME"]."""
    found, year, month = [], day.year, day.month
    while len(found) < count:
        if month in QUARTER_CODES and expiry(year, month) >= day:
            found.append(f"{root}{QUARTER_CODES[month]}{year % 100:02d}.{EXCHANGES[root]}")
        month, year = (1, year + 1) if month == 12 else (month + 1, year)
    return found


def contract_name(ticker: str) -> str:
    """"NQZ26.CME" -> "Dec 26"; a continuous symbol such as "NQ=F" -> "front" (Yahoo's own roll)."""
    code = ticker.split(".")[0]
    if "." not in ticker or len(code) < 3 or code[-3] not in MONTHS:
        return "front"
    return f"{dt.date(2000, MONTHS[code[-3]], 1):%b} {code[-2:]}"


def fetch_open_interest(ticker: str) -> float:
    import yfinance as yf

    value = yf.Ticker(ticker).info.get("openInterest")
    if value is None:
        raise ValueError(f"Yahoo has no open interest for {ticker}")
    return float(value)


@functools.lru_cache(maxsize=32)
def _active(root: str, day: dt.date) -> str | None:
    front, following = quarterly_contracts(root, day)
    try:
        return following if fetch_open_interest(following) > fetch_open_interest(front) else front
    except Exception:
        log.warning("Open interest unavailable for %s; using the continuous contract", root, exc_info=True)
        return None


def active_contract(symbol: str, day=None) -> str | None:
    """The contract we trade for ``symbol`` on ``day`` (default today): the next quarterly one
    once its open interest is higher, else the current one. None for markets without the rule
    or when open interest can't be read."""
    if symbol not in EXCHANGES:
        return None
    day = pd.Timestamp(day if day is not None else pd.Timestamp.now(tz="America/New_York")).date()
    return _active(symbol, day)


def futures_ticker(symbol: str, day=None) -> str:
    """The Yahoo symbol to price ``symbol`` with: the active contract, or the instrument's
    continuous symbol (NQ=F) when there's no rule or no open interest."""
    return active_contract(symbol, day) or instruments.resolve(symbol).price_ticker
