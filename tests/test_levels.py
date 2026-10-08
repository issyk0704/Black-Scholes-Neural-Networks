import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bsnn import analytics, levels as lv, levels_cli, market_data

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


WEDNESDAY = "2026-10-07T18:30:00+00:00"  # 14:30 New York
THURSDAY_MORNING = "2026-10-08T13:15:00+00:00"  # 09:15 New York, before the open


def test_refresh_open_interest_takes_the_morning_chain():
    saved = make_chain(ticker="QQQ", snapshot=WEDNESDAY, expiries=("2026-10-07", "2026-10-16"))
    morning = make_chain(ticker="QQQ", snapshot=THURSDAY_MORNING, expiries=("2026-10-16",))
    morning["openInterest"] = np.arange(len(morning), dtype=float)
    raw, when = lv.refresh_open_interest(saved, morning)
    assert set(raw["ExpiryDate"]) == {"2026-10-16"}  # Wednesday's expiry has gone
    expected = morning.set_index("contractSymbol").loc[raw["contractSymbol"], "openInterest"]
    assert (raw["openInterest"].to_numpy() == expected.to_numpy()).all()
    assert (raw["bid"].to_numpy() == saved.loc[raw.index, "bid"].to_numpy()).all()  # quotes stay the snapshot's
    assert when == pd.Timestamp("2026-10-08 09:15", tz="America/New_York")


def test_refresh_open_interest_waits_for_the_overnight_update():
    saved = make_chain(ticker="QQQ", snapshot=WEDNESDAY, expiries=("2026-10-16",))
    morning = make_chain(ticker="QQQ", snapshot=THURSDAY_MORNING, expiries=("2026-10-16",))  # same open interest
    raw, when = lv.refresh_open_interest(saved, morning)
    assert raw is saved and when is None
    assert lv.refresh_open_interest(saved, morning.iloc[0:0]) == (saved, None)


def test_levels_for_refreshes_open_interest_when_asked(snapshots):
    morning = make_chain(ticker="QQQ", snapshot=THURSDAY_MORNING, expiries=("2026-10-16", "2026-11-20"))
    morning["openInterest"] = 5000.0
    with patch.object(lv, "futures_ratio", return_value=41.5), \
         patch.object(lv.market_data, "fetch_option_chain", return_value=morning) as fetch:
        fresh = lv.levels_for("NQ", before="2026-10-08", directory=snapshots, refresh_oi=True)
        plain = lv.levels_for("NQ", before="2026-10-08", directory=snapshots)
    fetch.assert_called_once_with("QQQ", save=False)
    assert fresh.open_interest_time == pd.Timestamp("2026-10-08 09:15", tz="America/New_York")
    assert fresh.net_gamma == pytest.approx(plain.net_gamma * 5)  # same chain, five times the open interest
    assert plain.open_interest_time is None


def test_levels_for_keeps_the_snapshot_if_the_refresh_fails(snapshots):
    with patch.object(lv, "futures_ratio", return_value=41.5), \
         patch.object(lv.market_data, "fetch_option_chain", side_effect=ConnectionError("offline")):
        levels = lv.levels_for("NQ", before="2026-10-08", directory=snapshots, refresh_oi=True)
    assert levels is not None and levels.open_interest_time is None


def test_levels_for_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        lv.levels_for("AAPL")


BASE = dict(symbol="NQ", source="^NDX", snapshot_time=pd.Timestamp("2026-10-05 14:30", tz="America/New_York"),
            spot=24_000.0, ratio=1.3, net_gamma=5.1e9, flip=23_700.0, call_wall=24_200.0, put_wall=22_300.0,
            one_day_move_pct=0.0102)


HORIZONS = [
    lv.HorizonLevels(covers=("day",), end=pd.Timestamp("2026-10-06"), net_gamma=2e9, flip=23_700.0,
                     call_wall=24_200.0, put_wall=23_800.0, move_pct=0.0102),
    lv.HorizonLevels(covers=("week",), end=pd.Timestamp("2026-10-09"), net_gamma=-1e9, flip=24_100.0,
                     call_wall=24_500.0, put_wall=23_500.0, move_pct=0.0204),
    lv.HorizonLevels(covers=("month",), end=pd.Timestamp("2026-10-30"), net_gamma=4e9, flip=23_600.0,
                     call_wall=25_000.0, put_wall=22_300.0, move_pct=0.0442, nearest_expiry="2026-11-06"),
]


def sample_levels(**changes):
    return lv.MarketLevels(**(BASE | {"horizons": HORIZONS} | changes))


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
    assert "Net gamma, all expiries: +5.10bn" in nq["description"]
    today, week, month = nq["fields"]
    assert all(f["inline"] for f in nq["fields"])  # side by side
    assert today["name"] == "Today" and week["name"] == "This week (to Fri 09 Oct)"
    assert month["name"] == "This month (to Fri 30 Oct)"
    assert today["value"].split("\n") == ["Positive gamma: damping", "Flip **30,810**", "Call wall **31,460**",
                                           "Put wall **30,940**", "1σ **31,080 – 31,720** (±320)"]
    assert week["value"].startswith("Negative gamma: amplifying")
    assert month["value"].endswith("Nothing expires by then: uses Fri 06 Nov")
    assert "1σ ±2.04%" in es["fields"][1]["value"] and "footer" in es  # no live price for ES; footer on the last card
    assert len(json.dumps(payload)) < 6000  # Discord's limit for a message's embeds


def test_three_markets_fit_in_one_message():
    payload = levels_cli.build_message([sample_levels(symbol=s) for s in ("NQ", "ES", "YM")],
                                       {"NQ": 31_400.0, "ES": 6_800.0, "YM": 46_000.0},
                                       pd.Timestamp("2026-10-06").date())
    assert len(json.dumps(payload, ensure_ascii=False)) < 6000


def test_horizons_ending_together_share_a_column():
    card = levels_cli.market_embed(sample_levels(horizons=[
        lv.HorizonLevels(covers=("day", "week"), end=pd.Timestamp("2026-10-09"), net_gamma=1e9, flip=np.nan,
                         call_wall=24_200.0, put_wall=23_800.0, move_pct=0.01)]), None)
    assert card["fields"][0]["name"] == "Today & This week"


def test_card_without_horizons_shows_every_expiry():
    card = levels_cli.market_embed(sample_levels(horizons=[]), None)
    assert [f["name"] for f in card["fields"]] == ["Regime", "Gamma flip", "Call wall", "Put wall"]


def test_card_says_when_open_interest_was_refreshed():
    morning = pd.Timestamp("2026-10-06 09:15", tz="America/New_York")
    card = levels_cli.market_embed(sample_levels(open_interest_time=morning), None)
    assert card["description"].startswith("Open interest as of 09:15 New York, Tue 06 Oct (includes Mon's full session)")
    assert "prices and vols from NDX options at 14:30 New York, Mon 05 Oct" in card["description"]


def test_card_without_ratio_shows_underlying_levels():
    today = HORIZONS[0]
    card = levels_cli.market_embed(sample_levels(ratio=np.nan, horizons=[
        lv.HorizonLevels(**(vars(today) | {"flip": np.nan}))]), None)
    lines = card["fields"][0]["value"].split("\n")
    assert "Call wall **24,200.00** (NDX)" in lines and "Flip none within ±15%" in lines


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


def test_record_post_keeps_today_only(tmp_path):
    marker = tmp_path / "levels_posted.txt"
    marker.write_text("2026-10-05 daily\n2026-10-06 daily\n")
    levels_cli.record_post(marker, pd.Timestamp("2026-10-06").date(), "manual-0dte-13:35")
    assert marker.read_text() == "2026-10-06 daily\n2026-10-06 manual-0dte-13:35\n"


def test_include_today_lifts_the_date_limit(monkeypatch):
    seen = []
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer, **kw: seen.append(before))
    levels_cli.main(["--markets", "NQ"])
    levels_cli.main(["--markets", "NQ", "--include-today"])
    assert seen[0] is not None and seen[1] is None


def test_post_requires_webhook(monkeypatch, capsys):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer, **kw: sample_levels(symbol=symbol))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 31_400.0)
    assert levels_cli.main(["--markets", "NQ", "--post"]) == 1
    assert "DISCORD_WEBHOOK_URL" in capsys.readouterr().err


@pytest.fixture
def fake_post(monkeypatch, tmp_path):
    sent = {}
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    monkeypatch.setattr(levels_cli.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer, **kw: sample_levels(symbol=symbol))
    monkeypatch.setattr(levels_cli, "live_zero_dte", lambda symbol, prefer: sample_zero_dte(symbol=symbol))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 31_400.0)
    monkeypatch.setattr(levels_cli, "post", lambda url, payload: sent.update(url=url, payload=payload))
    return sent, tmp_path / "levels_posted.txt"


def test_manual_entry_names_the_kind_and_new_york_time():
    now = pd.Timestamp("2026-10-06 19:20", tz="UTC")
    assert levels_cli.manual_entry(now, zero_dte=True) == "manual-0dte-15:20"
    assert levels_cli.manual_entry(now, zero_dte=False) == "manual-daily-15:20"


def test_manual_post_is_recorded(fake_post):
    sent, marker = fake_post
    assert levels_cli.main(["--markets", "NQ", "ES", "--zero-dte", "--post"]) == 0
    assert sent["url"] == "https://discord.example/webhook" and len(sent["payload"]["embeds"]) == 2
    assert "manual-0dte-" in marker.read_text()


def test_manual_entries_do_not_block_scheduled_slots(tmp_path):
    marker = tmp_path / "levels_posted.txt"
    marker.write_text("2026-10-06 manual-0dte-13:35\n2026-10-06 manual-daily-09:00\n")
    slot, _ = levels_cli.due_slot(pd.Timestamp("2026-10-06 13:15", tz="UTC"), (levels_cli.DAILY_SLOT,), marker)
    assert slot.name == "daily"


def test_scheduled_daily_post_records_its_slot(fake_post, monkeypatch):
    sent, marker = fake_post
    monkeypatch.setattr(levels_cli, "due_slot", lambda now, slots, marker: (levels_cli.DAILY_SLOT, ""))
    assert levels_cli.main(["--post", "--scheduled"]) == 0
    assert sent["payload"]["content"].startswith("**Options levels for")
    assert marker.read_text().strip().endswith(" daily")


def test_zero_dte_is_no_longer_scheduled(fake_post):
    with pytest.raises(SystemExit):
        levels_cli.main(["--zero-dte", "--post", "--scheduled"])
    assert fake_post[0] == {}


def test_zero_dte_retries_markets_without_quotes(monkeypatch):
    attempts = {"ES": 0}

    def flaky(symbol, prefer):  # ES has no usable quotes on the first read
        if symbol == "ES":
            attempts["ES"] += 1
            if attempts["ES"] == 1:
                return None
        return sample_zero_dte(symbol=symbol)

    sleeps = []
    monkeypatch.setattr(levels_cli, "live_zero_dte", flaky)
    monkeypatch.setattr(levels_cli.time, "sleep", sleeps.append)
    found, missing = levels_cli.gather_with_retries(["NQ", "ES", "YM"], True, "etf", None)
    assert [lv.symbol for lv in found] == ["NQ", "ES", "YM"] and missing == []
    assert sleeps == [levels_cli.RETRY_WAIT_SECONDS]


def test_zero_dte_gives_up_after_the_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(levels_cli, "live_zero_dte", lambda symbol, prefer: None)
    monkeypatch.setattr(levels_cli.time, "sleep", sleeps.append)
    found, missing = levels_cli.gather_with_retries(["NQ"], True, "etf", None)
    assert found == [] and missing == ["NQ"] and len(sleeps) == levels_cli.ZERO_DTE_RETRIES


def test_daily_levels_do_not_retry(monkeypatch):
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer, **kw: None)
    monkeypatch.setattr(levels_cli.time, "sleep", lambda s: pytest.fail("daily levels read files; waiting won't help"))
    assert levels_cli.gather_with_retries(["NQ"], False, "etf", None) == ([], ["NQ"])


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



@pytest.mark.parametrize("session, week, month", [
    ("2026-10-07", "2026-10-09", "2026-10-30"),  # Wednesday
    ("2026-10-09", "2026-10-09", "2026-10-30"),  # Friday: the week ends today
    ("2026-10-30", "2026-10-30", "2026-10-30"),  # the month's last weekday
    ("2026-05-29", "2026-05-29", "2026-05-29"),  # Friday 29 May; 31 May is a Sunday
])
def test_horizon_ends(session, week, month):
    ends = lv.horizon_ends(session)
    assert ends == {"day": pd.Timestamp(session), "week": pd.Timestamp(week), "month": pd.Timestamp(month)}


def test_next_session_skips_the_weekend():
    assert lv.next_session("2026-10-08") == pd.Timestamp("2026-10-09")
    assert lv.next_session("2026-10-09") == pd.Timestamp("2026-10-12")


FRIDAY_CLOSE = "2026-10-02T20:30:00+00:00"  # 16:30 New York, after the close


def horizon_chain():
    """A Friday-evening chain with expiries today, during next week, at its end, at month end and later."""
    raw = make_chain(ticker="QQQ", snapshot=FRIDAY_CLOSE, spot=100.0,
                     expiries=("2026-10-05", "2026-10-07", "2026-10-09", "2026-10-30", "2026-11-20"))
    return raw.assign(futuresPrice=4150.0)


def test_each_horizon_uses_the_options_expiring_within_it():
    from bsnn.features import build_dataset
    dataset = build_dataset(horizon_chain(), lv.FILTERS)
    by_expiry = {e: analytics.gamma_levels(dataset[dataset["expiry"] <= e])["net"]
                 for e in ("2026-10-05", "2026-10-09", "2026-10-30")}
    day, week, month = lv.compute_horizons(dataset, "2026-10-05")
    assert (day.covers, week.covers, month.covers) == (("day",), ("week",), ("month",))
    assert day.net_gamma == pytest.approx(by_expiry["2026-10-05"])
    assert week.net_gamma == pytest.approx(by_expiry["2026-10-09"])
    assert month.net_gamma == pytest.approx(by_expiry["2026-10-30"])
    assert day.move_pct < week.move_pct < month.move_pct  # 1, 5 and 20 trading days
    assert not (day.nearest_expiry or week.nearest_expiry or month.nearest_expiry)


def test_one_day_horizon_matches_the_one_day_move():
    levels = lv.compute_levels("NQ", horizon_chain())
    assert levels.session == pd.Timestamp("2026-10-05")  # the Monday after the snapshot
    assert levels.horizons[0].move_pct == pytest.approx(levels.one_day_move_pct)


def test_friday_shares_a_column_for_the_day_and_week():
    from bsnn.features import build_dataset
    dataset = build_dataset(horizon_chain(), lv.FILTERS)
    horizons = lv.compute_horizons(dataset, "2026-10-09")
    assert [h.covers for h in horizons] == [("day", "week"), ("month",)]


def test_horizon_without_an_expiry_uses_the_nearest():
    from bsnn.features import build_dataset
    raw = make_chain(ticker="DIA", snapshot=FRIDAY_CLOSE, expiries=("2026-10-07", "2026-11-20"))
    day, week, month = lv.compute_horizons(build_dataset(raw, lv.FILTERS), "2026-10-05")
    assert day.nearest_expiry == "2026-10-07" and week.nearest_expiry == "" and month.nearest_expiry == ""


def test_levels_for_dates_the_horizons_from_the_posting_day(snapshots):
    with patch.object(lv, "futures_ratio", return_value=41.5):
        levels = lv.levels_for("NQ", before="2026-10-10", directory=snapshots)  # a Saturday
    assert levels.session == pd.Timestamp("2026-10-12")
    assert [h.covers for h in levels.horizons][-1] == ("month",)
