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
    assert levels.snapshot_date == pd.Timestamp("2026-10-06")
    assert levels.in_futures(100.0) == pytest.approx(4150.0)
    assert np.isfinite(levels.one_day_move_pct) and levels.one_day_move_pct > 0


def test_levels_for_can_prefer_index_options(snapshots):
    with patch.object(lv, "futures_ratio", return_value=1.3):
        levels = lv.levels_for("NQ", before="2026-10-07", directory=snapshots, prefer="index")
    assert levels.source == "^NDX" and levels.snapshot_date == pd.Timestamp("2026-10-05")
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


def test_levels_for_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        lv.levels_for("AAPL")


def sample_levels(**changes):
    base = dict(symbol="NQ", source="^NDX", snapshot_date=pd.Timestamp("2026-10-05"), spot=24_000.0, ratio=1.3,
                net_gamma=5.1e9, flip=23_700.0, call_wall=24_200.0, put_wall=22_300.0, one_day_move_pct=0.0102)
    return lv.MarketLevels(**(base | changes))


def test_message_is_webhook_ready_json():
    payload = levels_cli.build_message([sample_levels(), sample_levels(symbol="ES", net_gamma=-1e9)],
                                       {"NQ": 31_400.0}, pd.Timestamp("2026-10-06").date())
    json.dumps(payload)  # serialisable
    nq, es = payload["embeds"]
    assert nq["color"] == levels_cli.POSITIVE and es["color"] == levels_cli.NEGATIVE
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


@pytest.mark.parametrize("when, posted, expected", [
    ("2026-10-06 13:15", None, True),           # Tuesday 09:15 New York (summer time)
    ("2026-10-06 12:15", None, False),          # 08:15: too early
    ("2026-10-06 14:15", "2026-10-06", False),  # second run of the day: already posted
    ("2026-12-08 14:15", "2026-12-07", True),   # 09:15 New York in winter, yesterday's marker
    ("2026-10-10 13:15", None, False),          # Saturday
])
def test_should_post_at_open(tmp_path, when, posted, expected):
    marker = tmp_path / "levels_posted.txt"
    if posted:
        marker.write_text(posted + "\n")
    assert levels_cli.should_post_at_open(pd.Timestamp(when, tz="UTC"), marker)[0] is expected


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


def test_post_sends_and_marks_the_day(monkeypatch, tmp_path):
    sent = {}
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    monkeypatch.setattr(levels_cli.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer: sample_levels(symbol=symbol))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 31_400.0)
    monkeypatch.setattr(levels_cli, "post", lambda url, payload: sent.update(url=url, payload=payload))
    assert levels_cli.main(["--markets", "NQ", "ES", "--post"]) == 0
    assert sent["url"] == "https://discord.example/webhook" and len(sent["payload"]["embeds"]) == 2
    assert (tmp_path / "levels_posted.txt").exists()
