"""The morning post and a manual post, run through bsnn-levels from saved snapshots.

Only the network is faked: Yahoo (option chains and futures prices) and Discord.
These are the two ways the card is really made, so a bug in either path fails here.
"""

import json

import pandas as pd
import pytest

from bsnn import analytics, levels as lv, levels_cli, market_data
from bsnn.features import build_dataset

from conftest import make_chain

FRIDAY_CLOSE = "2026-10-02T20:30:00+00:00"  # 16:30 New York
MONDAY_0915 = "2026-10-05T13:15:00+00:00"  # before the open
MONDAY_1035 = "2026-10-05T14:35:00+00:00"  # in the session
RATIO = 41.5  # NQ / QQQ
EXPIRIES = ("2026-10-05", "2026-10-09", "2026-10-30")


def chains(snapshot, spot=100.0):
    """QQQ and NDX chains for the same moment; NDX is quoted 41 times higher."""
    qqq = make_chain(ticker="QQQ", snapshot=snapshot, spot=spot, expiries=EXPIRIES).assign(futuresPrice=RATIO * spot)
    ndx = make_chain(ticker="^NDX", snapshot=snapshot, spot=spot, expiries=EXPIRIES)
    for col in ("strike", "bid", "ask", "lastPrice", "underlyingPrice"):
        ndx[col] = ndx[col] * 41
    ndx["contractSymbol"] = "NDX" + ndx["contractSymbol"]
    return {"QQQ": qqq, "^NDX": ndx}


@pytest.fixture
def home(monkeypatch, tmp_path):
    """Friday's snapshots saved in a fresh data folder; Discord posts are captured."""
    monkeypatch.setattr(lv.market_data.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lv.market_data.paths, "OPTIONS_DIR", tmp_path / "options")
    for chain in chains(FRIDAY_CLOSE).values():
        market_data.save_option_snapshot(chain, tmp_path / "options")
    sent = []
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    monkeypatch.setattr(levels_cli, "post", lambda url, payload: sent.append(payload))
    return tmp_path, sent


def run(monkeypatch, now, live, nq_price, args):
    monkeypatch.setattr(levels_cli, "now_utc", lambda: pd.Timestamp(now))
    monkeypatch.setattr(lv.market_data, "fetch_option_chain", lambda source, save=False: live[source])
    monkeypatch.setattr(lv.market_data, "latest_price", lambda ticker: nq_price)
    return levels_cli.main(["--markets", "NQ", "--post"] + args)


def test_morning_post_uses_fridays_prices_with_this_mornings_open_interest(home, monkeypatch):
    tmp_path, sent = home
    morning = chains(MONDAY_0915)
    for chain in morning.values():  # quotes are blank before the open; open interest has been updated
        chain[["bid", "ask"]] = 0.0
        chain["openInterest"] = 2000.0
    assert run(monkeypatch, MONDAY_0915, morning, RATIO * 99.5, ["--scheduled"]) == 0
    (payload,) = sent
    assert payload["content"].startswith("**Options levels for Monday 05 October 2026**")
    card = payload["embeds"][0]
    assert card["description"].startswith("Open interest as of 09:15 New York, Mon 05 Oct")
    assert "prices and vols from QQQ + NDX options at 16:30 New York, Fri 02 Oct" in card["description"]
    assert "Levels measured at NQ 4,129 (QQQ 99.50)" in card["description"]
    assert [f["name"] for f in card["fields"]] == ["Today", "This week (to Fri 09 Oct)", "This month (to Fri 30 Oct)"]
    assert (tmp_path / "levels_posted.txt").read_text().strip() == "2026-10-05 daily"
    (record,) = [json.loads(line) for line in (tmp_path / "levels" / "2026-10-05.jsonl").read_text().splitlines()]
    assert record["kind"] == "scheduled" and record["posted_at"].startswith("2026-10-05T13:15")
    (nq,) = record["markets"]
    assert nq["sources"] == ["QQQ", "^NDX"] and nq["measured_at"] == pytest.approx(99.5)
    assert nq["open_interest_time"].startswith("2026-10-05T09:15")
    today = nq["horizons"][0]
    assert today["call_wall_futures"] == pytest.approx(today["call_wall"] * RATIO)
    assert today["range_low"] < RATIO * 99.5 < today["range_high"]


def test_manual_post_in_the_session_reads_the_live_chains_for_today(home, monkeypatch):
    tmp_path, sent = home
    live = chains(MONDAY_1035)  # on a strike, so today's expiry has an at-the-money pair with hours left
    price = RATIO * 100.0
    assert run(monkeypatch, MONDAY_1035, live, price, ["--include-today"]) == 0  # as the workflow runs it
    (payload,) = sent
    assert payload["content"].startswith("**Options levels for Monday 05 October 2026**")
    card = payload["embeds"][0]
    assert card["description"].startswith("From QQQ + NDX options at 10:35 New York, Mon 05 Oct")
    today = card["fields"][0]
    assert today["name"] == "Today" and "Nothing expires" not in today["value"]
    # Today's 1σ is the move today's expiry still prices to the close, not a whole day's.
    left_today = analytics.implied_moves(build_dataset(live["QQQ"], lv.FILTERS)).set_index("expiry")
    expected = price * left_today.at["2026-10-05", "move_pct"]
    assert f"(±{expected:,.0f})" in today["value"]
    assert "manual-daily-10:35" in (tmp_path / "levels_posted.txt").read_text()


def test_a_card_measured_at_a_stale_price_is_never_posted(home, monkeypatch, capsys):
    _, sent = home
    morning = chains(MONDAY_0915)
    for chain in morning.values():
        chain[["bid", "ask"]] = 0.0
    monkeypatch.setattr(lv.market_data, "latest_price", lambda ticker: (_ for _ in ()).throw(ConnectionError()))
    monkeypatch.setattr(levels_cli, "now_utc", lambda: pd.Timestamp(MONDAY_0915))
    monkeypatch.setattr(lv.market_data, "fetch_option_chain", lambda source, save=False: morning[source])
    assert levels_cli.main(["--markets", "NQ", "--post", "--scheduled"]) == 1
    assert sent == [] and "no current futures price" in capsys.readouterr().err
