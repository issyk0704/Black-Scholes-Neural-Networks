"""Post gamma levels and implied moves for NQ, ES and YM to a Discord channel.

    bsnn-levels                         # print the message as JSON (nothing is sent)
    bsnn-levels --post                  # send it to the webhook in DISCORD_WEBHOOK_URL
    bsnn-levels --post --scheduled      # send in the morning slot (09:15) or an update slot (10:45, 12:45, 14:45)
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

The scheduled post starts each day's card. Scheduled updates at 10:45, 12:45 and 14:45
New York (just before the 10:50 macro, the 13:30 NY PM session and the 14:50 macro) are each a NEW message with its own
copy box, headed "update at 10:45 New York / 15:45 UK". There is no 09:45 update (dropped
17 Nov 2026): it came half an hour after the morning card, from the thinnest live quotes
of the day. Last week's or month's scorecard
stays on the morning card only. A manual post edits the newest card in place (its
heading says when); --new-message posts a separate card instead. A card read from
live chains also lists the most-traded 0DTE strikes.

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

from bsnn import __version__, contracts, instruments, levels_checks, levels_record, levels_scorecard, market_data, paths
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


DAILY_SLOT = Slot("daily", dt.time(8, 30), dt.time(10, 30))  # scheduled for 09:15; closes before the first update
# Intraday refreshes of the day's card: before the 10:50 macro, before the 13:30 NY PM session (added 18 Nov 2026)
# and before the 14:50 macro (New York). No 09:45 update: too close to the 09:15 card and read from the thin quotes
# just after the open (17 Nov 2026).
UPDATE_SLOTS = (Slot("update-10:45", dt.time(10, 40), dt.time(11, 30)),
                Slot("update-12:45", dt.time(12, 40), dt.time(13, 30)),
                Slot("update-14:45", dt.time(14, 40), dt.time(15, 30)))
UK_TZ = "Europe/London"


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


CALM, FAST = "🟢 Calm: moves tend to snap back", "🔴 Fast: moves tend to run"


def _mood(net_gamma: float) -> str:
    """Positive gamma (dealers own the options) brakes moves; negative (they sold them) speeds them up."""
    return CALM if net_gamma > 0 else FAST


def _gamma_fields(levels: MarketLevels, prefix: str = "") -> list[dict]:
    return [
        {"name": f"Mood{f' ({prefix.strip()})' if prefix else ''}", "value": _mood(levels.net_gamma), "inline": False},
        {"name": "Ceiling (call wall)", "value": _points(levels, levels.call_wall), "inline": True},
        {"name": "Pivot (gamma flip)", "value": _points(levels, levels.flip), "inline": True},
        {"name": "Floor (put wall)", "value": _points(levels, levels.put_wall), "inline": True},
    ]


HORIZON_NAMES = {"day": "Today", "week": "This week", "month": "This month"}
STRADDLE_SHARE = levels_record.STRADDLE_SHARE  # the straddle: the average move, 0.8 of 1σ


def _futures_name(levels: MarketLevels) -> str:
    """E.g. "NQ Dec 26", or "NQ" when priced from the continuous symbol."""
    name = contracts.contract_name(levels.contract) if levels.contract else "front"
    return levels.symbol if name == "front" else f"{levels.symbol} {name}"


def _sources(levels: MarketLevels) -> str:
    """The chains in the levels, e.g. "QQQ + NDX"."""
    return " + ".join(s.lstrip("^") for s in levels.sources or (levels.source,))


def _ladder(levels: MarketLevels, h: HorizonLevels) -> list[str]:
    """Ceiling, pivot and floor ordered by price, highest first, as they sit on the chart
    (the pivot can be above the ceiling or below the floor); a missing level goes last."""
    rungs = [("Ceiling", h.call_wall), ("Pivot", h.flip), ("Floor", h.put_wall)]
    rungs.sort(key=lambda rung: -rung[1] if np.isfinite(rung[1]) else np.inf)
    return [f"{label} {_points(levels, value)}" for label, value in rungs]


# What each price on a column is, in the order labels share a line when they land on the same point.
RUNG_LABELS = {
    "pivot": "Pivot · above = calm, below = fast",
    "outer_high": "Outer high · rarely reached",
    "expected_high": "Expected high",
    "ceiling": "Ceiling · may stall here",
    "now": "◀ PRICE NOW",
    "busy": "Busy strike · pulls price into the close",
    "floor": "Floor · may bounce here",
    "expected_low": "Expected low",
    "outer_low": "Outer low · rarely reached",
}


def _rungs(levels: MarketLevels, h: HorizonLevels, live_price: float | None, busy: list[float]) -> list[tuple[float, str]]:
    """Every price on a column in futures points, with what it is: the levels, price now, the
    expected and outer ranges around it, and today's busiest strikes."""
    rungs = [(levels.in_futures(h.call_wall), "ceiling"), (levels.in_futures(h.flip), "pivot"),
             (levels.in_futures(h.put_wall), "floor")]
    if live_price:
        rungs.append((live_price, "now"))
        if np.isfinite(h.move_pct):
            for name, share in (("expected", STRADDLE_SHARE), ("outer", 1.0)):
                points = live_price * h.move_pct * share
                rungs += [(live_price + points, f"{name}_high"), (live_price - points, f"{name}_low")]
    rungs += [(levels.in_futures(k), "busy") for k in busy]
    return [(price, name) for price, name in rungs if np.isfinite(price)]


def _ladder_lines(rungs: list[tuple[float, str]]) -> list[str]:
    """One line per price, highest first, like the chart; things on the same point share a line."""
    by_point: dict[int, set[str]] = {}
    for price, name in rungs:
        by_point.setdefault(round(price), set()).add(name)
    order = list(RUNG_LABELS)

    def label(names: set[str]) -> str:
        others = [RUNG_LABELS[n] for n in sorted(names - {"busy"}, key=order.index)]
        if not others:
            return RUNG_LABELS["busy"]
        text = " + ".join(others)
        if "busy" in names:  # next to a level, a short tag instead of the full label
            text += " (busy strike)"
        return text

    return [f"**{point:,}**  {label(names)}" for point, names in sorted(by_point.items(), reverse=True)]


def _horizon_field(levels: MarketLevels, h: HorizonLevels, live_price: float | None) -> dict:
    """One column of the card: its mood, then every price top to bottom with what it is."""
    name = " & ".join(HORIZON_NAMES[c] for c in h.covers)
    if "day" not in h.covers:
        name += f" (to {h.end:%a %d %b})"
    lines = [_mood(h.net_gamma)]
    if not np.isfinite(h.flip):
        lines.append(f"No pivot within ±15%: {'fast' if h.net_gamma < 0 else 'calm'} at every price nearby")
    if np.isfinite(levels.ratio):
        busy = levels.busiest_calls + levels.busiest_puts if "day" in h.covers else []
        lines += _ladder_lines(_rungs(levels, h, live_price, busy))
    else:  # no futures conversion: the levels stay in the option underlying's prices
        lines += _ladder(levels, h)
    if np.isfinite(h.move_pct) and not (live_price and np.isfinite(levels.ratio)):
        lines.append(f"Expected ±{h.move_pct * STRADDLE_SHARE:.2%} · Outer ±{h.move_pct:.2%}")
    if h.nearest_expiry:
        lines.append(f"Nothing expires by then: uses {pd.Timestamp(h.nearest_expiry):%a %d %b}")
    return {"name": name, "value": "\n".join(lines), "inline": False}


HOW_TO_READ = "\n".join([
    "Read each list like the chart: higher prices at the top, **◀ PRICE NOW** is where price is.",
    "**Ceiling**: price often stalls here going up. **Floor**: often bounces here going down. "
    "A clean break through either can run.",
    "**Pivot**: above it the day is calm (moves snap back); below it, fast (moves run).",
    "**Expected**: the usual range to the column's end. **Outer**: rarely reached.",
    "**Busy strike**: today's most-traded options; price is often pulled there into the close.",
    "All prices are futures points. Where to look, not when to trade: the ICT setup decides.",
])


def market_embed(levels: MarketLevels, live_price: float | None) -> dict:
    """The card for a market: a price ladder for today, this week and this month."""
    # Without horizons (nothing left to expire this month), fall back to every expiry's levels.
    fields = [_horizon_field(levels, h, live_price) for h in levels.horizons] or _gamma_fields(levels)
    taken, source = levels.snapshot_time, _sources(levels)
    description = f"In {_futures_name(levels)} prices. {source} options read {taken:%a %d %b %H:%M} New York"
    if levels.open_interest_time is not None:
        description += f", open interest from {levels.open_interest_time:%a %H:%M}"
    description += "."
    for note in levels.oi_notes:                         # the OCC's file older than it should be, or unavailable
        description += f"\n⚠️ {note[0].upper()}{note[1:]}."
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


DAILY_FOOTER = ("Ceiling = call wall, floor = put wall, pivot = gamma flip, from the options expiring by each column's end (NDX + QQQ, SPX + SPY, DIA), assuming dealers own the calls and sold the puts. Expected = the straddle, outer = one standard deviation.")
ZERO_DTE_FOOTER = ("0DTE levels use this morning's open interest with live prices, so they move with price and time "
                   "left, but can't see positions opened today. Busiest strikes show volume, not direction. "
                   "Context, not signals.")


def _tv_number(value: float) -> str:
    """A price to the nearest quarter point, without trailing zeros: 28061, 7592.25; "-" when missing."""
    if not np.isfinite(value):
        return "-"
    return f"{round(value * 4) / 4:.2f}".rstrip("0").rstrip(".")


def tradingview_line(levels: MarketLevels, live_price: float | None, as_of: pd.Timestamp | None) -> str | None:
    """One line for the Options Levels Pine script, e.g.
    "NQ Z26 2026-11-13 09:45 | DW F 28002.00 28353.00 27796.00 27737.00 28023.00 27700.00 28059.00 | M F ...".

    After the market, contract code (- for the continuous symbol), session and New York time, one
    segment per column: the horizons it covers (D, W, M), the mood (C calm, F fast), then ceiling,
    pivot, floor, expected low / high, outer low / high in futures points (- when missing). None
    without a futures conversion.
    """
    if not np.isfinite(levels.ratio) or not levels.horizons:
        return None
    code = levels.contract.split(".")[0][-3:] if "." in levels.contract else "-"
    when = "-" if as_of is None else f"{as_of.tz_convert(market_data.MARKET_TZ):%H:%M}"
    session = "-" if levels.session is None else f"{levels.session:%Y-%m-%d}"
    segments = [f"{levels.symbol} {code} {session} {when}"]
    for h in levels.horizons:
        values = [levels.in_futures(h.call_wall), levels.in_futures(h.flip), levels.in_futures(h.put_wall)]
        if live_price and np.isfinite(h.move_pct):
            expected, outer = live_price * h.move_pct * STRADDLE_SHARE, live_price * h.move_pct
            values += [live_price - expected, live_price + expected, live_price - outer, live_price + outer]
        else:
            values += [np.nan] * 4
        letters = "".join(c[0].upper() for c in h.covers)
        segments.append(" ".join([letters, "C" if h.net_gamma > 0 else "F"] + [_tv_number(v) for v in values]))
    return " | ".join(segments)


def build_message(all_levels: list[MarketLevels], live_prices: dict[str, float], today: dt.date,
                  as_of: pd.Timestamp | None = None, review: dict | None = None, update: bool = False) -> dict:
    """The full webhook payload: a heading (with the New York time ``as_of``, if given; "update at" for an
    intraday update), one card per market and, on the week's or month's first card, the ``review`` scorecard last."""
    zero_dte = bool(all_levels) and isinstance(all_levels[0], ZeroDteLevels)
    make = zero_dte_embed if zero_dte else market_embed
    embeds = [make(lv, live_prices.get(lv.symbol)) for lv in all_levels]
    if embeds:
        embeds[-1]["footer"] = {"text": ZERO_DTE_FOOTER if zero_dte else DAILY_FOOTER}
        if not zero_dte:
            embeds[-1]["fields"].append({"name": "💡 How to read", "value": HOW_TO_READ, "inline": False})
    if review:
        embeds.append(review)
    if zero_dte:
        heading = f"**0DTE gamma update · {all_levels[0].snapshot_time:%H:%M} New York, {today:%a %d %b}**"
    else:
        heading = f"**Options levels for {today:%A %d %B %Y}**"
        if as_of is not None:
            heading += (f" · {'update at' if update else 'as of'} {as_of.tz_convert(market_data.MARKET_TZ):%H:%M} New York"
                        f" / {as_of.tz_convert(UK_TZ):%H:%M} UK")
    return {"username": "Options levels", "content": heading, "embeds": embeds}


def copy_message(all_levels: list[MarketLevels], live_prices: dict[str, float], as_of: pd.Timestamp | None) -> dict | None:
    """The small message posted under the card: just the lines for the Options Levels Pine script,
    one per market, in a code box to copy (long-press > Copy Text on the phone)."""
    lines = [ln for ln in (tradingview_line(lv, live_prices.get(lv.symbol), as_of) for lv in all_levels) if ln]
    if not lines:
        return None
    return {"username": "Options levels", "content": "📋 **Copy into TradingView** (Options Levels)\n```\n" + "\n".join(lines) + "\n```"}


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


MESSAGES_KEPT = 28  # message IDs to remember: two weeks of cards and their copy messages
COPY_SUFFIX = "-copy"  # store key of a day's copy message, e.g. "2026-11-16-copy"


def load_message_ids(path: Path) -> dict[str, str]:
    """Each posting day's card and copy message: {"2026-10-12": "<ID>", "2026-10-12-copy": "<ID>"}."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_message_id(path: Path, day, message_id: str, suffix: str = "") -> None:
    ids = load_message_ids(path) | {f"{pd.Timestamp(day):%Y-%m-%d}{suffix}": message_id}
    recent = dict(sorted(ids.items())[-MESSAGES_KEPT:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recent, indent=1) + "\n", encoding="utf-8")


def send(webhook: str, payload: dict, day, store: Path, update: bool, remember: bool = True,
         suffix: str = "") -> tuple[str, str | None]:
    """Edit ``day``'s card (its copy message with ``suffix``) when ``update`` and there is one, otherwise
    post a new message (also when it was deleted in Discord), which becomes the day's if ``remember``.
    Returns what happened and the message ID."""
    key = f"{pd.Timestamp(day):%Y-%m-%d}{suffix}"
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
        save_message_id(store, day, message_id, suffix)
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
                                      price_now=(prices or {}).get(symbol), contract=contract_for(symbol)))
        except Exception as exc:
            print(f"{symbol}: {exc}", file=sys.stderr)
            levels = None
        if levels:
            found.append(levels)
        else:
            missing.append(symbol)
    return found, missing


def contract_for(symbol: str) -> str:
    """The futures contract a market's card is priced in: ours by the open-interest rule (contracts.py)."""
    return contracts.futures_ticker(symbol)


def current_prices(markets: list[str]) -> dict[str, float]:
    """The latest futures price for each market, from :func:`contract_for`; one that can't be fetched
    is left out (its card then measures levels at the snapshot's price and shows no range around price)."""
    prices = {}
    for symbol in markets:
        try:
            prices[symbol] = market_data.latest_price(contract_for(symbol))
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


def scorecard(session: pd.Timestamp) -> dict | None:
    """Scores the finished morning cards (every day, so none age out of Yahoo's 60 days) and returns
    last week's or month's scorecard on the card that reviews it. A failure only leaves it out."""
    try:
        scores = levels_scorecard.update_scores(paths.DATA_DIR / "levels_scores.csv", paths.DATA_DIR / "levels")
        period = levels_scorecard.scorecard_period(session)
        return levels_scorecard.scorecard_embed(scores, *period) if period else None
    except Exception as exc:
        print(f"Scorecard left out: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post options levels for NQ, ES and YM to Discord.")
    parser.add_argument("--markets", nargs="+", default=list(DEFAULT_MARKETS), help="watchlist symbols")
    parser.add_argument("--source", choices=SOURCES, default="etf",
                        help="whose prices the levels are in: etf, QQQ/SPY/DIA (default); index, NDX/SPX. "
                             "Both chains go into the gamma levels either way")
    parser.add_argument("--zero-dte", action="store_true", help="today's 0DTE levels from live chains (US hours)")
    parser.add_argument("--post", action="store_true", help="send to the webhook in DISCORD_WEBHOOK_URL")
    parser.add_argument("--scheduled", "--at-open", dest="scheduled", action="store_true",
                        help="only send in a posting slot (morning card or intraday update), each once a day")
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
        slot, reason = due_slot(now, (DAILY_SLOT,) + UPDATE_SLOTS, marker)
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
    is_update = slot in UPDATE_SLOTS
    review = None if args.zero_dte or is_update else scorecard(session)  # the scorecard: morning card only
    payload = build_message(all_levels, live_prices, day, as_of=now, review=review, update=is_update)
    copy = None if args.zero_dte else copy_message(all_levels, live_prices, now)

    status = 1 if missing else 0  # a market left out fails the run, after posting the others
    if not args.post:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if copy:
            print("Copy message:\n" + copy["content"], file=sys.stderr)
        return status
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        print("Set DISCORD_WEBHOOK_URL to the channel's webhook URL to post.", file=sys.stderr)
        return 1
    # Scheduled posts (the morning card and each intraday update) and 0DTE updates are always new messages;
    # a scheduled one becomes the day's newest card, which a manual post edits.
    update = not (slot or args.zero_dte or args.new_message)
    store = paths.DATA_DIR / "levels_messages.json"
    action, message_id = send(webhook, payload, day, store, update,
                              remember=not args.zero_dte)  # a 0DTE update never becomes the day's card
    # The copy message follows its card: a new card gets a new one right under it, an edited card edits it.
    if copy:
        send(webhook, copy, day, store, update=action == "edited", suffix=COPY_SUFFIX)
    record_post(marker, today, slot.name if slot else manual_entry(now, args.zero_dte))
    kind = ("zero-dte" if args.zero_dte else "update" if is_update else "scheduled" if slot
            else "manual-edit" if action == "edited" else "manual")
    levels_record.append(paths.DATA_DIR / "levels", day, now, kind, all_levels, live_prices, message_id)
    print(f"{action.capitalize()} {'0DTE ' if args.zero_dte else ''}levels for "
          f"{', '.join(lv.symbol for lv in all_levels)}.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
