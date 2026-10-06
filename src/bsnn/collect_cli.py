"""Daily collection of option-chain snapshots for every watchlist market.

    bsnn-collect               # every options proxy and cash index (SPX, NDX, RUT) in bsnn.instruments
    bsnn-collect SPY QQQ       # just these tickers
    bsnn-collect --force       # run outside US hours anyway (tests that Yahoo is reachable)

Yahoo blanks bids and asks outside US trading hours, so a snapshot is only
useful when taken during the regular session (14:45-20:55 UK). Outside it the
collector does nothing unless forced, and even then nothing is saved: every
chain must pass :func:`bsnn.market_data.snapshot_problem` (taken in the session,
mostly quoted) to be kept. ``scripts/register-daily-collection.ps1`` schedules
this on weekdays.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from bsnn import market_data, paths
from bsnn.instruments import option_sources

log = logging.getLogger("bsnn.collect")


def collect(tickers: list[str], directory: Path | None = None,
            fetch: Callable[..., pd.DataFrame] = market_data.fetch_option_chain) -> dict[str, str]:
    """Fetch and save each ticker's chain; returns a one-line outcome per ticker."""
    outcomes = {}
    for ticker in tickers:
        try:
            chain = fetch(ticker, save=False)
            problem = market_data.snapshot_problem(chain)
            if problem:
                outcomes[ticker] = f"skipped: {problem}"
                continue
            path = market_data.save_option_snapshot(chain, directory)
            share = market_data.live_quote_share(chain)
            outcomes[ticker] = f"saved {len(chain):,} contracts ({share:.0%} quoted) to {path.name}"
        except Exception as exc:
            outcomes[ticker] = f"failed: {exc}"
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save today's option chains for the watchlist proxies.")
    parser.add_argument("tickers", nargs="*", help="defaults to every options proxy and cash index in the watchlist")
    parser.add_argument("--force", action="store_true", help="run even outside US market hours")
    args = parser.parse_args(argv)

    handlers = _start_logging()
    try:
        if not args.force and not market_data.us_session_open():
            log.info("US options market is closed; nothing collected. Use --force to override.")
            return 0
        outcomes = collect([t.upper() for t in args.tickers] or option_sources())
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
