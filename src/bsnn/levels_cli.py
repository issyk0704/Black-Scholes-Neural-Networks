"""Post gamma levels and implied moves for NQ, ES and YM to a Discord channel.

    bsnn-levels                         # print the daily message as JSON (nothing is sent)
    bsnn-levels --post                  # send it to the webhook in DISCORD_WEBHOOK_URL
    bsnn-levels --zero-dte              # today's 0DTE levels from live chains (US hours only)
    bsnn-levels --post --scheduled      # send only if one of the posting slots below is due
    bsnn-levels --markets NQ ES
    bsnn-levels --source index          # read NDX / SPX options instead of QQQ / SPY (DIA stays for YM)

Daily levels use the newest saved snapshot from before today for prices and vols,
with this morning's open interest, which is published overnight and so includes
the previous session's full trading. 0DTE levels read live chains, so they move with price and time left,
but they still rest on this morning's open interest.

The webhook URL is a secret: keep it in an environment variable (or a GitHub
Actions secret), never in a file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bsnn import __version__, instruments, market_data, paths
from bsnn.levels import DEFAULT_MARKETS, SOURCES, MarketLevels, ZeroDteLevels, levels_for, live_zero_dte

POSITIVE, NEGATIVE = 0x2FA36B, 0xE0663A  # embed colours
USER_AGENT = f"bsnn-levels/{__version__} (+https://github.com/issyk0704/Black-Scholes-Neural-Networks)"


@dataclass(frozen=True)
class Slot:
    """A posting window in New York time. GitHub can start scheduled jobs late, hence the width."""

    name: str
    start: dt.time
    end: dt.time


DAILY_SLOT = Slot("daily", dt.time(8, 30), dt.time(11, 0))  # scheduled for 09:15
ZERO_DTE_SLOTS = (
    Slot("0dte-10:00", dt.time(10, 0), dt.time(10, 40)),   # once quotes have settled after the open
    Slot("0dte-11:30", dt.time(11, 30), dt.time(12, 10)),  # late AM session, before lunch
    Slot("0dte-13:30", dt.time(13, 30), dt.time(14, 10)),  # start of the PM session
    Slot("0dte-15:00", dt.time(15, 0), dt.time(15, 40)),   # the last hour, when 0DTE gamma is strongest
)


# --- When to post ----------------------------------------------------------------

def posted_today(marker: Path, day: dt.date) -> set[str]:
    if not marker.exists():
        return set()
    lines = (line.split() for line in marker.read_text().splitlines())
    return {parts[1] for parts in lines if len(parts) == 2 and parts[0] == f"{day:%Y-%m-%d}"}


def record_post(marker: Path, day: dt.date, name: str) -> None:
    """Remember a post made today: a slot name, or a manual entry (older days are dropped)."""
    done = posted_today(marker, day) | {name}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("".join(f"{day:%Y-%m-%d} {name}\n" for name in sorted(done)))


def manual_entry(now: pd.Timestamp, zero_dte: bool) -> str:
    """The record for a manual post, e.g. "manual-0dte-15:20". It never matches a slot name,
    so manual posts don't block the scheduled ones."""
    ny = now.tz_convert(market_data.MARKET_TZ)
    return f"manual-{'0dte' if zero_dte else 'daily'}-{ny:%H:%M}"


def due_slot(now: pd.Timestamp, slots: tuple[Slot, ...], marker: Path) -> tuple[Slot | None, str]:
    """The slot to post for now, or None with the reason there isn't one."""
    ny = now.tz_convert(market_data.MARKET_TZ)
    if ny.weekday() >= 5:
        return None, "it's the weekend"
    current = [s for s in slots if s.start <= ny.time() < s.end]
    if not current:
        return None, f"it's {ny:%H:%M} in New York, outside every posting slot"
    if current[0].name in posted_today(marker, ny.date()):
        return None, f"the {current[0].name} update was already posted"
    return current[0], ""


# --- Message ---------------------------------------------------------------------

def _points(levels: MarketLevels, level: float) -> str:
    if not np.isfinite(level):
        return "none within ±15%"
    if np.isfinite(levels.ratio):
        return f"**{levels.in_futures(level):,.0f}**"
    return f"**{level:,.2f}** ({levels.source.lstrip('^')})"


def _size(levels: MarketLevels, amount: float) -> str:
    """A distance in futures points, or in the underlying's units if there's no ratio."""
    return f"{levels.in_futures(amount):,.0f} pts" if np.isfinite(levels.ratio) else f"{amount:,.2f}"


def _price_then(levels: MarketLevels) -> str:
    where = f"{levels.source.lstrip('^')} {levels.spot:,.2f}"
    if np.isfinite(levels.ratio):
        where += f" ≈ {levels.symbol} {levels.in_futures(levels.spot):,.0f}"
    return where


def _gamma_fields(levels: MarketLevels, prefix: str = "") -> list[dict]:
    regime = (f"Positive {prefix}gamma: dealers tend to sell rallies and buy dips, damping moves" if levels.net_gamma > 0
              else f"Negative {prefix}gamma: dealers tend to chase moves, amplifying them")
    return [
        {"name": "Regime", "value": regime, "inline": False},
        {"name": "Gamma flip", "value": _points(levels, levels.flip), "inline": True},
        {"name": "Call wall", "value": _points(levels, levels.call_wall), "inline": True},
        {"name": "Put wall", "value": _points(levels, levels.put_wall), "inline": True},
    ]


def market_embed(levels: MarketLevels, live_price: float | None) -> dict:
    """The daily card for a market: previous session's levels."""
    fields = _gamma_fields(levels)
    if np.isfinite(levels.one_day_move_pct):
        move = f"±{levels.one_day_move_pct:.2%}"
        if live_price:
            points = live_price * levels.one_day_move_pct
            move += (f" ≈ ±{points:,.0f} pts\nFrom {live_price:,.0f}: **{live_price - points:,.0f} – "
                     f"{live_price + points:,.0f}**")
        fields.append({"name": "1-day implied move (1σ)", "value": move, "inline": False})
    taken, source = levels.snapshot_time, levels.source.lstrip("^")
    description = (f"From {source} options at {taken:%H:%M} New York, {taken:%a %d %b} ({_price_then(levels)} "
                   f"then). Net gamma {levels.net_gamma / 1e9:+.2f}bn $ per 1% move.")
    if levels.open_interest_time is not None:
        oi = levels.open_interest_time
        description = (f"Open interest as of {oi:%H:%M} New York, {oi:%a %d %b} (includes {taken:%a}'s full session); "
                       f"prices and vols from {source} options at {taken:%H:%M} New York, {taken:%a %d %b} "
                       f"({_price_then(levels)} then). Net gamma {levels.net_gamma / 1e9:+.2f}bn $ per 1% move.")
    return {
        "title": f"{levels.symbol} options levels",
        "description": description,
        "color": POSITIVE if levels.net_gamma > 0 else NEGATIVE,
        "fields": fields,
    }


def _busiest(levels: ZeroDteLevels, strikes: list[tuple[float, float]]) -> str:
    if not strikes:
        return "none"
    return ", ".join(f"{_points(levels, k).strip('*')} ({v:,.0f})" for k, v in strikes)


def zero_dte_embed(levels: ZeroDteLevels, live_price: float | None) -> dict:
    """The intraday card for a market: one expiry's levels, rest-of-day move and busiest strikes."""
    fields = _gamma_fields(levels, prefix="0DTE " if levels.is_today else "")
    if np.isfinite(levels.straddle):
        if np.isfinite(levels.ratio):
            price = live_price or levels.in_futures(levels.spot)
            sigma = levels.in_futures(levels.one_sigma)
        else:  # no futures conversion: stay in the underlying's own units
            price, sigma = levels.spot, levels.one_sigma
        value = f"Straddle ±{_size(levels, levels.straddle)} · 1σ ±{_size(levels, levels.one_sigma)}"
        value += f"\nFrom {price:,.0f}: **{price - sigma:,.0f} – {price + sigma:,.0f}** (1σ)"
        fields.append({"name": "Implied move to the close" if levels.is_today else "Implied move to expiry",
                       "value": value, "inline": False})
    fields.append({"name": "Busiest strikes today (contracts)",
                   "value": f"Calls: {_busiest(levels, levels.top_calls)}\nPuts: {_busiest(levels, levels.top_puts)}",
                   "inline": False})
    expiry = ("today's expiry (0DTE)" if levels.is_today
              else f"the nearest expiry, {pd.Timestamp(levels.expiry):%a %d %b} (none expires today)")
    taken = levels.snapshot_time
    return {
        "title": f"{levels.symbol} 0DTE gamma · {taken:%H:%M} New York",
        "description": f"From {levels.source.lstrip('^')} options, {expiry} ({_price_then(levels)}). "
                       f"Net gamma {levels.net_gamma / 1e9:+.2f}bn $ per 1% move.",
        "color": POSITIVE if levels.net_gamma > 0 else NEGATIVE,
        "fields": fields,
    }


DAILY_FOOTER = ("Call wall: most call gamma above price then; put wall: most put gamma below. Converted to futures at "
                "the futures/ETF price ratio when the options were read. Gamma assumes dealers are long customers' "
                "calls and short their puts. Context, not signals.")
ZERO_DTE_FOOTER = ("0DTE levels use this morning's open interest with live prices, so they move with price and time "
                   "left, but can't see positions opened today. Busiest strikes show volume, not direction. "
                   "Context, not signals.")


def build_message(all_levels: list[MarketLevels], live_prices: dict[str, float], today: dt.date) -> dict:
    """The full webhook payload: a heading plus one card per market."""
    zero_dte = bool(all_levels) and isinstance(all_levels[0], ZeroDteLevels)
    make = zero_dte_embed if zero_dte else market_embed
    embeds = [make(lv, live_prices.get(lv.symbol)) for lv in all_levels]
    if embeds:
        embeds[-1]["footer"] = {"text": ZERO_DTE_FOOTER if zero_dte else DAILY_FOOTER}
    if zero_dte:
        heading = f"**0DTE gamma update · {all_levels[0].snapshot_time:%H:%M} New York, {today:%a %d %b}**"
    else:
        heading = f"**Options levels for {today:%A %d %B %Y}**"
    return {"username": "Options levels", "content": heading, "embeds": embeds}


def post(webhook_url: str, payload: dict) -> None:
    request = urllib.request.Request(webhook_url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:  # raises on HTTP errors
        if response.status not in (200, 204):
            raise RuntimeError(f"Discord answered {response.status}")


# --- Command line ------------------------------------------------------------------

ZERO_DTE_RETRIES = 3  # Yahoo's quotes lag the open, so a read soon after it can be mostly blank
RETRY_WAIT_SECONDS = 120


def gather_with_retries(markets: list[str], zero_dte: bool, source: str,
                        before: dt.date | None) -> tuple[list, list[str]]:
    """Like :func:`gather`, but 0DTE reads retry the markets that had no usable quotes."""
    found, missing = gather(markets, zero_dte, source, before)
    for _ in range(ZERO_DTE_RETRIES if zero_dte else 0):
        if not missing:
            break
        print(f"No usable quotes yet for {', '.join(missing)}; trying again in {RETRY_WAIT_SECONDS} s.")
        time.sleep(RETRY_WAIT_SECONDS)
        more, missing = gather(missing, zero_dte, source, before)
        found += more
    found.sort(key=lambda lv: markets.index(lv.symbol))  # keep the cards in the requested order
    return found, missing


def gather(markets: list[str], zero_dte: bool, source: str, before: dt.date | None) -> tuple[list, list[str]]:
    found, missing = [], []
    for symbol in markets:
        try:
            levels = (live_zero_dte(symbol, source) if zero_dte
                      else levels_for(symbol, before=before, prefer=source, refresh_oi=True))
        except Exception as exc:
            print(f"{symbol}: {exc}", file=sys.stderr)
            levels = None
        if levels:
            found.append(levels)
        else:
            missing.append(symbol)
    return found, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post options levels for NQ, ES and YM to Discord.")
    parser.add_argument("--markets", nargs="+", default=list(DEFAULT_MARKETS), help="watchlist symbols")
    parser.add_argument("--source", choices=SOURCES, default="etf",
                        help="etf: QQQ/SPY/DIA options (default); index: NDX/SPX options")
    parser.add_argument("--zero-dte", action="store_true", help="today's 0DTE levels from live chains (US hours)")
    parser.add_argument("--post", action="store_true", help="send to the webhook in DISCORD_WEBHOOK_URL")
    parser.add_argument("--scheduled", "--at-open", dest="scheduled", action="store_true",
                        help="only send if a posting slot is due, and only once per slot")
    parser.add_argument("--include-today", action="store_true",
                        help="daily levels: also use today's snapshot (for testing after the close)")
    args = parser.parse_args(argv)

    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert(market_data.MARKET_TZ).date()
    marker = paths.DATA_DIR / "levels_posted.txt"
    slot = None
    if args.scheduled:
        slot, reason = due_slot(now, ZERO_DTE_SLOTS if args.zero_dte else (DAILY_SLOT,), marker)
        if slot is None:
            print(f"Not posting: {reason}.")
            return 0

    markets = [s.upper() for s in args.markets]
    all_levels, missing = gather_with_retries(markets, args.zero_dte, args.source,
                                              before=None if args.include_today else today)
    if missing:
        print(f"No usable data for: {', '.join(missing)}", file=sys.stderr)
    if not all_levels:
        print("Nothing to post.", file=sys.stderr)
        return 1

    live_prices = {}
    for lv in all_levels:
        try:
            live_prices[lv.symbol] = market_data.latest_price(instruments.resolve(lv.symbol).price_ticker)
        except Exception:
            pass  # the card just leaves out the range around the current price
    payload = build_message(all_levels, live_prices, today)

    if not args.post:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        print("Set DISCORD_WEBHOOK_URL to the channel's webhook URL to post.", file=sys.stderr)
        return 1
    post(webhook, payload)
    record_post(marker, today, slot.name if slot else manual_entry(now, args.zero_dte))
    print(f"Posted {'0DTE ' if args.zero_dte else ''}levels for {', '.join(lv.symbol for lv in all_levels)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
