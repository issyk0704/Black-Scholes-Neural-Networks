import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bsnn import levels as lv, levels_cli, market_data

from conftest import make_chain

INDEX_SNAPSHOT = "2026-10-05T18:30:00+00:00"  # Monday 14:30 New York


@pytest.fixture
def snapshots(tmp_path):
    """An NDX snapshot on Monday and a QQQ snapshot on Tuesday, in tmp_path."""
    market_data.save_option_snapshot(make_chain(ticker="^NDX", snapshot=INDEX_SNAPSHOT,
                                                expiries=("2026-10-16", "2026-11-20")), tmp_path)
    market_data.save_option_snapshot(make_chain(ticker="QQQ", snapshot="2026-10-06T18:30:00+00:00",
                                                expiries=("2026-10-16", "2026-11-20")), tmp_path)
    return tmp_path


def test_latest_snapshot_respects_before(snapshots):
    assert lv.latest_snapshot("QQQ", directory=snapshots).name == "QQQ_options_2026-10-06.csv.gz"
    assert lv.latest_snapshot("QQQ", before="2026-10-06", directory=snapshots) is None
    assert lv.latest_snapshot("^NDX", before="2026-10-06", directory=snapshots).name.startswith("^NDX")


def test_levels_for_reads_etf_options_by_default(snapshots):
    with patch.object(lv, "futures_ratio", return_value=41.5):
        levels = lv.levels_for("NQ", before="2026-10-07", directory=snapshots)
    assert levels.source == "QQQ" and levels.symbol == "NQ"
    assert levels.snapshot_time.date() == pd.Timestamp("2026-10-06").date()
    assert levels.in_futures(100.0) == pytest.approx(4150.0)
    assert np.isfinite(levels.one_day_move_pct) and levels.one_day_move_pct > 0


def test_levels_for_can_prefer_index_options(snapshots):
    with patch.object(lv, "futures_ratio", return_value=1.3):
        levels = lv.levels_for("NQ", before="2026-10-07", directory=snapshots, prefer="index")
    assert levels.source == "^NDX" and levels.snapshot_time.date() == pd.Timestamp("2026-10-05").date()
    with pytest.raises(ValueError, match="Unknown source"):
        lv.levels_for("NQ", prefer="futures")


def test_levels_for_falls_back_when_first_source_has_no_open_interest(tmp_path):
    no_oi = make_chain(ticker="QQQ", snapshot=INDEX_SNAPSHOT, expiries=("2026-10-16",)).assign(openInterest=0.0)
    market_data.save_option_snapshot(no_oi, tmp_path)
    market_data.save_option_snapshot(make_chain(ticker="^NDX", snapshot=INDEX_SNAPSHOT, expiries=("2026-10-16",)),
                                     tmp_path)
    with patch.object(lv, "futures_ratio", side_effect=ConnectionError("offline")):
        levels = lv.levels_for("NQ", before="2026-10-06", directory=tmp_path)
    assert levels.source == "^NDX" and np.isnan(levels.ratio)


def test_ym_has_no_index_options():
    from bsnn import instruments
    assert lv.option_sources(instruments.resolve("YM"), "index") == ["DIA"]


def test_futures_ratio_uses_price_recorded_with_snapshot():
    from bsnn import instruments
    raw = make_chain(ticker="QQQ", spot=760.0).assign(futuresPrice=31_464.0)
    with patch.object(lv.market_data, "get_history", side_effect=AssertionError("shouldn't download")):
        assert lv.futures_ratio(instruments.resolve("NQ"), raw) == pytest.approx(31_464.0 / 760.0)


def test_levels_for_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        lv.levels_for("AAPL")


BASE = dict(symbol="NQ", source="^NDX", snapshot_time=pd.Timestamp("2026-10-05 14:30", tz="America/New_York"),
            spot=24_000.0, ratio=1.3, net_gamma=5.1e9, flip=23_700.0, call_wall=24_200.0, put_wall=22_300.0,
            one_day_move_pct=0.0102)


def sample_levels(**changes):
    return lv.MarketLevels(**(BASE | changes))


def sample_zero_dte(**changes):
    extra = dict(expiry="2026-10-05", is_today=True, straddle=40.0, one_sigma=50.0,
                 top_calls=[(24_200.0, 51_000.0), (24_300.0, 20_000.0)], top_puts=[(23_800.0, 64_000.0)])
    return lv.ZeroDteLevels(**(BASE | extra | changes))


def test_message_is_webhook_ready_json():
    payload = levels_cli.build_message([sample_levels(), sample_levels(symbol="ES", net_gamma=-1e9)],
                                       {"NQ": 31_400.0}, pd.Timestamp("2026-10-06").date())
    json.dumps(payload)  # serialisable
    nq, es = payload["embeds"]
    assert nq["color"] == levels_cli.POSITIVE and es["color"] == levels_cli.NEGATIVE
    assert "at 14:30 New York, Mon 05 Oct (NDX 24,000.00 ≈ NQ 31,200 then)" in nq["description"]
    fields = {f["name"]: f["value"] for f in nq["fields"]}
    assert fields["Gamma flip"] == "**30,810**" and fields["Call wall"] == "**31,460**"
    assert "±1.02% ≈ ±320 pts" in fields["1-day implied move (1σ)"]
    assert "31,080 – 31,720" in fields["1-day implied move (1σ)"]
    assert "Range" not in str(es["fields"]) and "footer" in es  # no live price for ES; footer on the last card
    assert len(json.dumps(payload)) < 6000  # Discord's limit for a message's embeds


def test_card_without_ratio_shows_underlying_levels():
    card = levels_cli.market_embed(sample_levels(ratio=np.nan, flip=np.nan), None)
    fields = {f["name"]: f["value"] for f in card["fields"]}
    assert fields["Call wall"] == "**24,200.00** (NDX)" and fields["Gamma flip"] == "none within ±15%"


def test_zero_dte_message():
    payload = levels_cli.build_message([sample_zero_dte(), sample_zero_dte(symbol="YM", is_today=False,
                                                                           expiry="2026-10-09")],
                                       {"NQ": 31_210.0}, pd.Timestamp("2026-10-05").date())
    json.dumps(payload)
    assert payload["content"] == "**0DTE gamma update · 14:30 New York, Mon 05 Oct**"
    nq, ym = payload["embeds"]
    assert nq["title"] == "NQ 0DTE gamma · 14:30 New York" and "today's expiry (0DTE)" in nq["description"]
    fields = {f["name"]: f["value"] for f in nq["fields"]}
    assert fields["Regime"].startswith("Positive 0DTE gamma")
    assert fields["Implied move to the close"] == "Straddle ±52 pts · 1σ ±65 pts\nFrom 31,210: **31,145 – 31,275** (1σ)"
    assert fields["Busiest strikes today (contracts)"] == "Calls: 31,460 (51,000), 31,590 (20,000)\nPuts: 30,940 (64,000)"
    assert "the nearest expiry, Fri 09 Oct (none expires today)" in ym["description"]
    assert "Implied move to expiry" in {f["name"] for f in ym["fields"]}
    assert ym["footer"]["text"] == levels_cli.ZERO_DTE_FOOTER


@pytest.mark.parametrize("when, posted, expected", [
    ("2026-10-06 13:15", "", "daily"),                  # Tuesday 09:15 New York (summer time)
    ("2026-10-06 12:15", "", None),                     # 08:15: too early
    ("2026-10-06 14:15", "2026-10-06 daily", None),     # second scheduled run: already posted
    ("2026-12-08 14:15", "2026-12-07 daily", "daily"),  # 09:15 New York in winter; yesterday's record
    ("2026-10-10 13:15", "", None),                     # Saturday
])
def test_daily_slot(tmp_path, when, posted, expected):
    marker = tmp_path / "levels_posted.txt"
    marker.write_text(posted + "\n")
    slot, _ = levels_cli.due_slot(pd.Timestamp(when, tz="UTC"), (levels_cli.DAILY_SLOT,), marker)
    assert (slot.name if slot else None) == expected


@pytest.mark.parametrize("when, posted, expected", [
    ("2026-10-06 13:45", "", "0dte-09:45"),             # 09:45 New York in summer
    ("2026-10-06 14:45", "2026-10-06 0dte-09:45", "0dte-10:45"),
    ("2026-10-06 15:45", "", None),                     # 11:45: between slots
    ("2026-10-06 18:45", "", "0dte-14:45"),
    ("2026-12-08 14:45", "", "0dte-09:45"),             # 09:45 New York in winter
    ("2026-12-08 18:30", "", "0dte-13:30"),
    ("2026-12-08 18:45", "2026-12-08 0dte-13:30", None),  # the summer-time run lands in the same slot
])
def test_zero_dte_slots(tmp_path, when, posted, expected):
    marker = tmp_path / "levels_posted.txt"
    marker.write_text(posted + "\n")
    slot, _ = levels_cli.due_slot(pd.Timestamp(when, tz="UTC"), levels_cli.ZERO_DTE_SLOTS, marker)
    assert (slot.name if slot else None) == expected


def test_record_post_keeps_today_only(tmp_path):
    marker = tmp_path / "levels_posted.txt"
    marker.write_text("2026-10-05 daily\n2026-10-06 daily\n")
    levels_cli.record_post(marker, pd.Timestamp("2026-10-06").date(), levels_cli.ZERO_DTE_SLOTS[0])
    assert marker.read_text() == "2026-10-06 0dte-09:45\n2026-10-06 daily\n"


def test_include_today_lifts_the_date_limit(monkeypatch):
    seen = []
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer: seen.append(before))
    levels_cli.main(["--markets", "NQ"])
    levels_cli.main(["--markets", "NQ", "--include-today"])
    assert seen[0] is not None and seen[1] is None


def test_post_requires_webhook(monkeypatch, capsys):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer: sample_levels(symbol=symbol))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 31_400.0)
    assert levels_cli.main(["--markets", "NQ", "--post"]) == 1
    assert "DISCORD_WEBHOOK_URL" in capsys.readouterr().err


@pytest.fixture
def fake_post(monkeypatch, tmp_path):
    sent = {}
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    monkeypatch.setattr(levels_cli.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer: sample_levels(symbol=symbol))
    monkeypatch.setattr(levels_cli, "live_zero_dte", lambda symbol, prefer: sample_zero_dte(symbol=symbol))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 31_400.0)
    monkeypatch.setattr(levels_cli, "post", lambda url, payload: sent.update(url=url, payload=payload))
    return sent, tmp_path / "levels_posted.txt"


def test_manual_post_does_not_block_scheduled_ones(fake_post):
    sent, marker = fake_post
    assert levels_cli.main(["--markets", "NQ", "ES", "--post"]) == 0
    assert sent["url"] == "https://discord.example/webhook" and len(sent["payload"]["embeds"]) == 2
    assert not marker.exists()


def test_scheduled_zero_dte_post_records_its_slot(fake_post, monkeypatch):
    sent, marker = fake_post
    monkeypatch.setattr(levels_cli, "due_slot", lambda now, slots, marker: (levels_cli.ZERO_DTE_SLOTS[1], ""))
    assert levels_cli.main(["--zero-dte", "--post", "--scheduled"]) == 0
    assert sent["payload"]["content"].startswith("**0DTE gamma update")
    assert marker.read_text().strip().endswith("0dte-10:45")


def test_compute_zero_dte_uses_todays_expiry():
    raw = make_chain(ticker="QQQ", expiries=("2026-01-05", "2026-01-16")).assign(futuresPrice=4140.0)
    levels = lv.compute_zero_dte("NQ", raw)
    assert levels.is_today and levels.expiry == "2026-01-05"
    assert levels.ratio == pytest.approx(41.4) and np.isfinite(levels.straddle)
    for busiest in (levels.top_calls, levels.top_puts):
        volumes = [v for _, v in busiest]
        assert 0 < len(busiest) <= 3 and volumes == sorted(volumes, reverse=True)
    assert levels.top_calls[0][0] == 100  # the synthetic volume peaks at the money


def test_compute_zero_dte_falls_back_to_nearest_expiry():
    raw = make_chain(ticker="DIA", expiries=("2026-01-09", "2026-01-16"))
    with patch.object(lv, "futures_ratio", return_value=100.0):
        levels = lv.compute_zero_dte("YM", raw)
    assert not levels.is_today and levels.expiry == "2026-01-09"

