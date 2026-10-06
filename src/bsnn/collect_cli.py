"""Daily collection of option-chain snapshots for every watchlist market.

    bsnn-collect               # every options proxy in bsnn.instruments
    bsnn-collect SPY QQQ       # just these tickers
    bsnn-collect --force       # skip the US-market-hours check

Yahoo blanks bids and asks outside US trading hours, so a snapshot is only
useful when taken during the session (14:30-21:00 UK). The check below refuses
to run outside it, and any chain where fewer than half the contracts have a
two-sided quote (a holiday, or stale data) is skipped rather than saved.
``scripts/register-daily-collection.ps1`` schedules this on weekdays.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from bsnn import market_data, paths
from bsnn.instruments import option_proxies

SESSION_START, SESSION_END = dt.time(9, 45), dt.time(15, 55)  # New York time, avoiding the open and close
MIN_LIVE_SHARE = 0.5

log = logging.getLogger("bsnn.collect")


def us_session_open(now: pd.Timestamp | None = None) -> bool:
    """True on a weekday between 09:45 and 15:55 New York time."""
    now = (now if now is not None else pd.Timestamp.now(tz="UTC")).tz_convert(market_data.MARKET_TZ)
    return now.weekday() < 5 and SESSION_START <= now.time() <= SESSION_END


def collect(tickers: list[str], directory: Path | None = None,
            fetch: Callable[..., pd.DataFrame] = market_data.fetch_option_chain) -> dict[str, str]:
    """Fetch and save each ticker's chain; returns a one-line outcome per ticker."""
    outcomes = {}
    for ticker in tickers:
        try:
            chain = fetch(ticker, save=False)
            share = market_data.live_quote_share(chain)
            if share < MIN_LIVE_SHARE:
                outcomes[ticker] = f"skipped: only {share:.0%} of {len(chain)} contracts had a live quote"
                continue
            path = market_data.save_option_snapshot(chain, directory)
            outcomes[ticker] = f"saved {len(chain):,} contracts ({share:.0%} quoted) to {path.name}"
        except Exception as exc:
            outcomes[ticker] = f"failed: {exc}"
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save today's option chains for the watchlist proxies.")
    parser.add_argument("tickers", nargs="*", help="defaults to every options proxy in the watchlist")
    parser.add_argument("--force", action="store_true", help="run even outside US market hours")
    args = parser.parse_args(argv)

    handlers = _start_logging()
    try:
        if not args.force and not us_session_open():
            log.info("US options market is closed; nothing collected. Use --force to override.")
            return 0
        outcomes = collect([t.upper() for t in args.tickers] or option_proxies())
        for ticker, outcome in outcomes.items():
            log.info("%-5s %s", ticker, outcome)
        failures = sum(outcome.startswith("failed") for outcome in outcomes.values())
        return 1 if failures == len(outcomes) else 0
    finally:
        for handler in handlers:
            log.removeHandler(handler)
            handler.close()


def _start_logging() -> list[logging.Handler]:
    """Log to the console and to data/collect.log, independently of any other logging setup."""
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(paths.DATA_DIR / "collect.log", encoding="utf-8")]
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return handlers


if __name__ == "__main__":
    raise SystemExit(main())
