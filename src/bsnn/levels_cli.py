"""Post the previous session's gamma levels and implied moves to a Discord channel.

    bsnn-levels                     # print the Discord message as JSON (nothing is sent)
    bsnn-levels --post              # send it to the webhook in DISCORD_WEBHOOK_URL
    bsnn-levels --post --at-open    # send it only once a day, between 08:30 and 11:00 New York time
    bsnn-levels --markets NQ ES
    bsnn-levels --source index      # read NDX / SPX options instead of QQQ / SPY (DIA stays for YM)

Levels use the newest saved snapshot from before today, so at the open they
reflect the previous session's open interest. The webhook URL is a secret: keep
it in an environment variable (or a GitHub Actions secret), never in a file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from bsnn import __version__, instruments, market_data, paths
from bsnn.levels import DEFAULT_MARKETS, SOURCES, MarketLevels, levels_for

POST_WINDOW = (dt.time(8, 30), dt.time(11, 0))  # New York time
POSITIVE, NEGATIVE = 0x2FA36B, 0xE0663A  # embed colours
USER_AGENT = f"bsnn-levels/{__version__} (+https://github.com/issyk0704/Black-Scholes-Neural-Networks)"


def _points(levels: MarketLevels, level: float) -> str:
    if not np.isfinite(level):
        return "none within ±15%"
    if np.isfinite(levels.ratio):
        return f"**{levels.in_futures(level):,.0f}**"
    return f"**{level:,.2f}** ({levels.source.lstrip('^')})"


def market_embed(levels: MarketLevels, live_price: float | None) -> dict:
    """One Discord embed (card) for a market."""
    positive = levels.net_gamma > 0
    regime = ("Positive gamma: dealers tend to sell rallies and buy dips, damping moves" if positive
              else "Negative gamma: dealers tend to chase moves, amplifying them")
    fields = [
        {"name": "Regime", "value": regime, "inline": False},
        {"name": "Gamma flip", "value": _points(levels, levels.flip), "inline": True},
        {"name": "Call wall", "value": _points(levels, levels.call_wall), "inline": True},
        {"name": "Put wall", "value": _points(levels, levels.put_wall), "inline": True},
    ]
    if np.isfinite(levels.one_day_move_pct):
        move = f"±{levels.one_day_move_pct:.2%}"
        if live_price:
            points = live_price * levels.one_day_move_pct
            move += (f" ≈ ±{points:,.0f} pts\nFrom {live_price:,.0f}: **{live_price - points:,.0f} – "
                     f"{live_price + points:,.0f}**")
        fields.append({"name": "1-day implied move (1σ)", "value": move, "inline": False})
    source = levels.source.lstrip("^")
    taken = levels.snapshot_time
    where = f"{source} {levels.spot:,.2f}"
    if np.isfinite(levels.ratio):
        where += f" ≈ {levels.symbol} {levels.in_futures(levels.spot):,.0f}"
    return {
        "title": f"{levels.symbol} options levels",
        "description": f"From {source} options at {taken:%H:%M} New York, {taken:%a %d %b} ({where} then). "
                       f"Net gamma {levels.net_gamma / 1e9:+.2f}bn $ per 1% move.",
        "color": POSITIVE if positive else NEGATIVE,
        "fields": fields,
    }


def build_message(all_levels: list[MarketLevels], live_prices: dict[str, float], today: dt.date) -> dict:
    """The full webhook payload: a heading plus one card per market."""
    embeds = [market_embed(lv, live_prices.get(lv.symbol)) for lv in all_levels]
    if embeds:
        embeds[-1]["footer"] = {"text": "Call wall: most call gamma above price then; put wall: most put gamma below. "
                                        "Converted to futures at the futures/ETF price ratio when the options were "
                                        "read. Gamma assumes dealers are long customers' calls and short their puts. "
                                        "Context, not signals."}
    return {"username": "Options levels", "content": f"**Options levels for {today:%A %d %B %Y}**",
            "embeds": embeds}


def post(webhook_url: str, payload: dict) -> None:
    request = urllib.request.Request(webhook_url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:  # raises on HTTP errors
        if response.status not in (200, 204):
            raise RuntimeError(f"Discord answered {response.status}")


def should_post_at_open(now: pd.Timestamp, marker: Path) -> tuple[bool, str]:
    """Post once per weekday, between 08:30 and 11:00 New York time."""
    ny = now.tz_convert(market_data.MARKET_TZ)
    if ny.weekday() >= 5:
        return False, "it's the weekend"
    if not POST_WINDOW[0] <= ny.time() < POST_WINDOW[1]:
        return False, f"it's {ny:%H:%M} in New York, outside 08:30-11:00"
    if marker.exists() and marker.read_text().strip() == f"{ny:%Y-%m-%d}":
        return False, "today's levels were already posted"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post the previous session's options levels to Discord.")
    parser.add_argument("--markets", nargs="+", default=list(DEFAULT_MARKETS), help="watchlist symbols")
    parser.add_argument("--source", choices=SOURCES, default="etf",
                        help="etf: QQQ/SPY/DIA options (default); index: NDX/SPX options")
    parser.add_argument("--post", action="store_true", help="send to the webhook in DISCORD_WEBHOOK_URL")
    parser.add_argument("--at-open", action="store_true", help="only send once a day, 08:30-11:00 New York")
    parser.add_argument("--include-today", action="store_true",
                        help="also use today's snapshot (for testing after the close; normally only earlier days)")
    args = parser.parse_args(argv)

    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert(market_data.MARKET_TZ).date()
    marker = paths.DATA_DIR / "levels_posted.txt"
    if args.at_open:
        ok, reason = should_post_at_open(now, marker)
        if not ok:
            print(f"Not posting: {reason}.")
            return 0

    all_levels, missing = [], []
    for symbol in (s.upper() for s in args.markets):
        levels = levels_for(symbol, before=None if args.include_today else today, prefer=args.source)
        if levels:
            all_levels.append(levels)
        else:
            missing.append(symbol)
    if missing:
        print(f"No usable snapshot (with open interest) for: {', '.join(missing)}", file=sys.stderr)
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
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{today:%Y-%m-%d}\n")
    print(f"Posted levels for {', '.join(lv.symbol for lv in all_levels)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
