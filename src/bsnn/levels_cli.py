"""Post gamma levels and implied moves for NQ, ES and YM to a Discord channel.

    bsnn-levels                         # print the message as JSON (nothing is sent)
    bsnn-levels --post                  # send it to the webhook in DISCORD_WEBHOOK_URL
    bsnn-levels --post --scheduled      # send only in the morning posting slot, once a day
    bsnn-levels --post --new-message    # post a separate card instead of updating the day's
    bsnn-levels --zero-dte              # today's 0DTE levels from live chains (US hours only)
    bsnn-levels --markets NQ ES
    bsnn-levels --source index          # state the levels in NDX / SPX prices instead of QQQ / SPY

The morning post gives each market's levels for today, this week and this month,
each from the options expiring within that horizon. The gamma levels add up the
index and ETF options (NDX + QQQ for NQ, SPX + SPY for ES, DIA for YM); each column
also lists the ETF's own levels, in its prices. They use the newest saved
snapshot from before today for prices and vols, with this morning's open interest,
which is published overnight and so includes the previous session's full trading.
0DTE levels read live chains, so they move with price and time left, but they
still rest on this morning's open interest.

The posting day follows the clock: before the 16:00 New York close it's today,
after it the next weekday. Before a morning card is posted it must pass the
checks in levels_checks; a market that fails is left out and the run fails.

The scheduled post starts each day's card. A manual post then edits that card in
place (its heading says when it was updated), so corrections and refreshes don't
add messages; --new-message posts a separate card instead.

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
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bsnn import __version__, instruments, levels_checks, levels_record, market_data, paths
from bsnn.levels import (DEFAULT_MARKETS, SOURCES, HorizonLevels, MarketLevels, ZeroDteLevels, levels_for,
                         live_zero_dte, session_for)

POSITIVE, NEGATIVE = 0x2FA36B, 0xE0663A  # embed colours
USER_AGENT = f"bsnn-levels/{__version__} (+https://github.com/issyk0704/Black-Scholes-Neural-Networks)"


@dataclass(frozen=True)
class Slot:
    """A posting window in New York time. GitHub can start scheduled jobs late, hence the width."""

    name: str
    start: dt.time
    end: dt.time


DAILY_SLOT = Slot("daily", dt.time(8, 30), dt.time(11, 0))  # scheduled for 09:15


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


HORIZON_NAMES = {"day": "Today", "week": "This week", "month": "This month"}


def _option_price(level: float) -> str:
    """A level in the option underlying's own prices: whole strikes without decimals."""
    if not np.isfinite(level):
        return "none"
    return f"{level:,.0f}" if float(level).is_integer() else f"{level:,.2f}"


def _own_line(levels: MarketLevels, h: HorizonLevels) -> str | None:
    """The source chain's own levels for the horizon, e.g. "QQQ alone: flip 748.31 · CW 760 · PW 750"."""
    own = next((o for o in levels.own_horizons if o.end == h.end), None)
    if own is None:
        return None
    name = levels.source.lstrip("^") + (" alone" if len(levels.sources) > 1 else "")
    return (f"{name}: flip {_option_price(own.flip)} · CW {_option_price(own.call_wall)} "
            f"· PW {_option_price(own.put_wall)}")


def _measured_at(levels: MarketLevels) -> str:
    """E.g. "NQ 30,882 (QQQ 745.74)"."""
    return (f"{levels.symbol} {levels.in_futures(levels.measured_at):,.0f} "
            f"({levels.source.lstrip('^')} {levels.measured_at:,.2f})")


def _sources(levels: MarketLevels) -> str:
    """The chains in the levels, e.g. "QQQ + NDX"."""
    return " + ".join(s.lstrip("^") for s in levels.sources or (levels.source,))


def _horizon_field(levels: MarketLevels, h: HorizonLevels, live_price: float | None) -> dict:
    """One column of the card: a horizon's regime, gamma levels, the source's own levels and 1σ range."""
    name = " & ".join(HORIZON_NAMES[c] for c in h.covers)
    if "day" not in h.covers:
        name += f" (to {h.end:%a %d %b})"
    lines = ["Positive gamma: damping" if h.net_gamma > 0 else "Negative gamma: amplifying",
             f"Flip {_points(levels, h.flip)}", f"Call wall {_points(levels, h.call_wall)}",
             f"Put wall {_points(levels, h.put_wall)}"]
    own = _own_line(levels, h)
    if own:
        lines.append(own)
    if np.isfinite(h.move_pct):
        if live_price:
            points = live_price * h.move_pct
            lines.append(f"1σ **{live_price - points:,.0f} – {live_price + points:,.0f}** (±{points:,.0f})")
        else:
            lines.append(f"1σ ±{h.move_pct:.2%}")
    if h.nearest_expiry:
        lines.append(f"Nothing expires by then: uses {pd.Timestamp(h.nearest_expiry):%a %d %b}")
    return {"name": name, "value": "\n".join(lines), "inline": True}


def market_embed(levels: MarketLevels, live_price: float | None) -> dict:
    """The morning card for a market: levels for today, this week and this month, side by side."""
    # Without horizons (nothing left to expire this month), fall back to every expiry's levels.
    fields = [_horizon_field(levels, h, live_price) for h in levels.horizons] or _gamma_fields(levels)
    taken, source = levels.snapshot_time, _sources(levels)
    description = (f"From {source} options at {taken:%H:%M} New York, {taken:%a %d %b} ({_price_then(levels)} "
                   f"then). Net gamma, all expiries: {levels.net_gamma / 1e9:+.2f}bn $ per 1% move.")
    if levels.open_interest_time is not None:
        oi = levels.open_interest_time
        description = (f"Open interest as of {oi:%H:%M} New York, {oi:%a %d %b} (includes {taken:%a}'s full session); "
                       f"prices and vols from {source} options at {taken:%H:%M} New York, {taken:%a %d %b} "
                       f"({_price_then(levels)} then). Net gamma, all expiries: {levels.net_gamma / 1e9:+.2f}bn $ per 1% move.")
    if np.isfinite(levels.measured_at):
        description += f" Levels measured at {_measured_at(levels)}, the price when posted."
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


DAILY_FOOTER = ("Each column uses only the options expiring by its end, adding up index and ETF options (NDX + QQQ, "
                "SPX + SPY; DIA for YM); the ETF line gives the ETF's own levels in its prices. Positive gamma: "
                "dealers tend to sell rallies and buy dips; negative: they chase moves. Call wall: most call gamma "
                "above the price the levels were measured at; put wall: most put gamma below. 1σ: the move the options price to the column's "
                "end, around the current futures price (about 68% of closes land inside). Converted to futures at "
                "the futures/ETF price ratio when the options were read. Gamma assumes dealers are long customers' "
                "calls and short their puts. Context, not signals.")
ZERO_DTE_FOOTER = ("0DTE levels use this morning's open interest with live prices, so they move with price and time "
                   "left, but can't see positions opened today. Busiest strikes show volume, not direction. "
                   "Context, not signals.")


def build_message(all_levels: list[MarketLevels], live_prices: dict[str, float], today: dt.date,
                  as_of: pd.Timestamp | None = None) -> dict:
    """The full webhook payload: a heading (with the New York time ``as_of``, if given) plus one card per market."""
    zero_dte = bool(all_levels) and isinstance(all_levels[0], ZeroDteLevels)
    make = zero_dte_embed if zero_dte else market_embed
    embeds = [make(lv, live_prices.get(lv.symbol)) for lv in all_levels]
    if embeds:
        embeds[-1]["footer"] = {"text": ZERO_DTE_FOOTER if zero_dte else DAILY_FOOTER}
    if zero_dte:
        heading = f"**0DTE gamma update · {all_levels[0].snapshot_time:%H:%M} New York, {today:%a %d %b}**"
    else:
        heading = f"**Options levels for {today:%A %d %B %Y}**"
        if as_of is not None:
            heading += f" · as of {as_of.tz_convert(market_data.MARKET_TZ):%H:%M} New York"
    return {"username": "Options levels", "content": heading, "embeds": embeds}


def webhook_endpoint(webhook_url: str, path: str = "", **query: str) -> str:
    """The webhook URL with ``path`` appended and ``query`` added, keeping any query it already has."""
    parts = urllib.parse.urlsplit(webhook_url)
    params = urllib.parse.parse_qsl(parts.query) + list(query.items())
    return urllib.parse.urlunsplit(parts._replace(path=parts.path.rstrip("/") + path,
                                                  query=urllib.parse.urlencode(params)))


def _send(url: str, payload: dict, method: str) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method=method,
                                     headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:  # raises on HTTP errors
        if response.status not in (200, 204):
            raise RuntimeError(f"Discord answered {response.status}")
        body = response.read()
    return json.loads(body) if body else {}


def post(webhook_url: str, payload: dict) -> str | None:
    """Send a new message; returns its ID, so the day's card can be edited later."""
    return _send(webhook_endpoint(webhook_url, wait="true"), payload, "POST").get("id")


def edit(webhook_url: str, message_id: str, payload: dict) -> None:
    """Replace a message this webhook sent (a webhook can't change its name on an edit)."""
    body = {key: value for key, value in payload.items() if key != "username"}
    _send(webhook_endpoint(webhook_url, f"/messages/{message_id}"), body, "PATCH")


MESSAGES_KEPT = 14  # days of message IDs to remember


def load_message_ids(path: Path) -> dict[str, str]:
    """Each posting day's card: {"2026-10-12": "<Discord message ID>"}."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_message_id(path: Path, day, message_id: str) -> None:
    ids = load_message_ids(path) | {f"{pd.Timestamp(day):%Y-%m-%d}": message_id}
    recent = dict(sorted(ids.items())[-MESSAGES_KEPT:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recent, indent=1) + "\n", encoding="utf-8")


def send(webhook: str, payload: dict, day, store: Path, update: bool, remember: bool = True) -> tuple[str, str | None]:
    """Edit ``day``'s card when ``update`` and there is one, otherwise post a new message
    (also when the card was deleted in Discord), which becomes the day's card if ``remember``.
    Returns what happened and the message ID."""
    key = f"{pd.Timestamp(day):%Y-%m-%d}"
    existing = load_message_ids(store).get(key) if update else None
    if existing:
        try:
            edit(webhook, existing, payload)
            return "edited", existing
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            print(f"The {key} card is gone from Discord; posting a new one.", file=sys.stderr)
    message_id = post(webhook, payload)
    if message_id and remember:
        save_message_id(store, day, message_id)
    return "posted", message_id


# --- Command line ------------------------------------------------------------------

ZERO_DTE_RETRIES = 3  # Yahoo's quotes lag the open, so a read soon after it can be mostly blank
RETRY_WAIT_SECONDS = 120


def gather_with_retries(markets: list[str], zero_dte: bool, source: str,
                        before: dt.date | None, prices: dict[str, float] | None = None) -> tuple[list, list[str]]:
    """Like :func:`gather`, but 0DTE reads retry the markets that had no usable quotes."""
    found, missing = gather(markets, zero_dte, source, before, prices)
    for _ in range(ZERO_DTE_RETRIES if zero_dte else 0):
        if not missing:
            break
        print(f"No usable quotes yet for {', '.join(missing)}; trying again in {RETRY_WAIT_SECONDS} s.")
        time.sleep(RETRY_WAIT_SECONDS)
        more, missing = gather(missing, zero_dte, source, before)
        found += more
    found.sort(key=lambda lv: markets.index(lv.symbol))  # keep the cards in the requested order
    return found, missing


def gather(markets: list[str], zero_dte: bool, source: str, before: dt.date | None,
           prices: dict[str, float] | None = None) -> tuple[list, list[str]]:
    """Each market's levels; daily levels are measured at the futures price in ``prices``."""
    found, missing = [], []
    for symbol in markets:
        try:
            levels = (live_zero_dte(symbol, source) if zero_dte
                      else levels_for(symbol, before=before, prefer=source, refresh_oi=True,
                                      price_now=(prices or {}).get(symbol)))
        except Exception as exc:
            print(f"{symbol}: {exc}", file=sys.stderr)
            levels = None
        if levels:
            found.append(levels)
        else:
            missing.append(symbol)
    return found, missing


def current_prices(markets: list[str]) -> dict[str, float]:
    """The latest futures price for each market; one that can't be fetched is left out
    (its card then measures levels at the snapshot's price and shows no range around price)."""
    prices = {}
    for symbol in markets:
        inst = instruments.resolve(symbol)
        try:
            prices[symbol] = market_data.latest_price(inst.price_ticker)
        except Exception as exc:
            print(f"{symbol}: no current price ({exc})", file=sys.stderr)
    return prices


def now_utc() -> pd.Timestamp:
    """The current time (a function so tests can fix the clock)."""
    return pd.Timestamp.now(tz="UTC")


def checked(all_levels: list[MarketLevels], session: pd.Timestamp, prefer: str) -> tuple[list[MarketLevels], list[str]]:
    """The cards that pass :func:`levels_checks.problems`, and the symbols left out (with the reasons printed)."""
    passed, failed = [], []
    for lv in all_levels:
        issues = levels_checks.problems(lv, session, prefer)
        if issues:
            print(f"{lv.symbol} left out: {'; '.join(issues)}", file=sys.stderr)
            failed.append(lv.symbol)
        else:
            passed.append(lv)
    return passed, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post options levels for NQ, ES and YM to Discord.")
    parser.add_argument("--markets", nargs="+", default=list(DEFAULT_MARKETS), help="watchlist symbols")
    parser.add_argument("--source", choices=SOURCES, default="etf",
                        help="whose prices the levels are in: etf, QQQ/SPY/DIA (default); index, NDX/SPX. "
                             "Both chains go into the gamma levels either way")
    parser.add_argument("--zero-dte", action="store_true", help="today's 0DTE levels from live chains (US hours)")
    parser.add_argument("--post", action="store_true", help="send to the webhook in DISCORD_WEBHOOK_URL")
    parser.add_argument("--scheduled", "--at-open", dest="scheduled", action="store_true",
                        help="only send in the morning posting slot, and only once a day")
    parser.add_argument("--new-message", action="store_true",
                        help="post a separate card instead of editing the day's (manual posts edit it by default)")
    parser.add_argument("--include-today", action="store_true",
                        help="no longer needed: the posting day follows the clock (kept for older workflow runs)")
    args = parser.parse_args(argv)

    now = now_utc()
    today = now.tz_convert(market_data.MARKET_TZ).date()
    session = session_for(now.tz_convert(market_data.MARKET_TZ))  # today, or the next weekday after the close
    marker = paths.DATA_DIR / "levels_posted.txt"
    slot = None
    if args.scheduled and args.zero_dte:
        parser.error("0DTE updates aren't scheduled any more; post them by hand without --scheduled")
    if args.scheduled:
        slot, reason = due_slot(now, (DAILY_SLOT,), marker)
        if slot is None:
            print(f"Not posting: {reason}.")
            return 0

    markets = [s.upper() for s in args.markets]
    live_prices = current_prices(markets)
    all_levels, missing = gather_with_retries(markets, args.zero_dte, args.source, before=session.date(),
                                              prices=live_prices)
    if not args.zero_dte:
        all_levels, failed = checked(all_levels, session, args.source)
        missing += failed
    if missing:
        print(f"No usable data for: {', '.join(missing)}", file=sys.stderr)
    if not all_levels:
        print("Nothing to post.", file=sys.stderr)
        return 1
    day = today if args.zero_dte else session.date()
    payload = build_message(all_levels, live_prices, day, as_of=now)

    status = 1 if missing else 0  # a market left out fails the run, after posting the others
    if not args.post:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return status
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        print("Set DISCORD_WEBHOOK_URL to the channel's webhook URL to post.", file=sys.stderr)
        return 1
    # The scheduled post and 0DTE updates are always new messages; a manual post edits the day's card.
    update = not (slot or args.zero_dte or args.new_message)
    action, message_id = send(webhook, payload, day, paths.DATA_DIR / "levels_messages.json", update,
                              remember=not args.zero_dte)  # a 0DTE update never becomes the day's card
    record_post(marker, today, slot.name if slot else manual_entry(now, args.zero_dte))
    kind = "zero-dte" if args.zero_dte else "scheduled" if slot else "manual-edit" if action == "edited" else "manual"
    levels_record.append(paths.DATA_DIR / "levels", day, now, kind, all_levels, live_prices, message_id)
    print(f"{action.capitalize()} {'0DTE ' if args.zero_dte else ''}levels for "
          f"{', '.join(lv.symbol for lv in all_levels)}.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
