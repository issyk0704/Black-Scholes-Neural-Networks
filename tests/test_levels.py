import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from conftest import fits_discord, is_copy

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
    assert nq["description"] == "In NQ prices. NDX options read Mon 05 Oct 14:30 New York."
    today, week, month = nq["fields"]
    assert not any(f["inline"] for f in nq["fields"])  # one ladder under another
    assert today["name"] == "Today" and week["name"] == "This week (to Fri 09 Oct)"
    assert month["name"] == "This month (to Fri 30 Oct)"
    # Every price top to bottom, like the chart, with what it is.
    assert today["value"].split("\n") == [levels_cli.CALM,
                                           "**31,720**  Outer high · rarely reached",
                                           "**31,656**  Expected high",
                                           "**31,460**  Ceiling · may stall here",
                                           "**31,400**  ◀ PRICE NOW",
                                           "**31,144**  Expected low",
                                           "**31,080**  Outer low · rarely reached",
                                           "**30,940**  Floor · may bounce here",
                                           "**30,810**  Pivot · above = calm, below = fast"]
    assert week["value"].startswith(levels_cli.FAST)
    assert month["value"].endswith("Nothing expires by then: uses Fri 06 Nov")
    assert "Expected ±1.63% · Outer ±2.04%" in es["fields"][1]["value"]  # no live price for ES: sizes only
    assert "footer" in es and es["fields"][-1]["name"] == "💡 How to read"  # once, on the last card
    assert all(f["name"] != "💡 How to read" for f in nq["fields"])
    assert fits_discord(payload)


OWN_HORIZONS = [lv.HorizonLevels(**(vars(h) | {"flip": 748.31, "call_wall": 760.0, "put_wall": 750.0}))
                for h in HORIZONS]


def test_card_shows_only_futures_prices():
    """The ETF's own levels (e.g. QQQ 680) are a different unit and stay off the card."""
    card = levels_cli.market_embed(sample_levels(source="QQQ", sources=("QQQ", "^NDX"), own_horizons=OWN_HORIZONS),
                                   31_400.0)
    assert card["description"].startswith("In NQ prices. QQQ + NDX options read Mon 05 Oct 14:30 New York")
    assert all("748.31" not in f["value"] and "CW" not in f["value"] for f in card["fields"])


def test_levels_on_the_same_point_share_a_line():
    lines = levels_cli._ladder_lines([(100.2, "ceiling"), (99.8, "busy"), (90.0, "busy"), (95.0, "now")])
    assert lines == ["**100**  Ceiling · may stall here (busy strike)", "**95**  ◀ PRICE NOW",
                     "**90**  Busy strike · pulls price into the close"]


def test_a_column_without_a_pivot_says_so():
    no_flip = lv.HorizonLevels(covers=("day",), end=pd.Timestamp("2026-10-06"), net_gamma=-1e9, flip=np.nan,
                               call_wall=24_200.0, put_wall=23_800.0, move_pct=0.0102)
    lines = levels_cli.market_embed(sample_levels(horizons=[no_flip]), 31_400.0)["fields"][0]["value"].split("\n")
    assert lines[:2] == [levels_cli.FAST, "No pivot within ±15%: fast at every price nearby"]
    assert not any("Pivot ·" in line for line in lines)


def test_daily_levels_are_measured_at_the_futures_price(monkeypatch):
    seen = {}
    monkeypatch.setattr(levels_cli, "levels_for",
                        lambda symbol, before, prefer, **kw: seen.update({symbol: kw["price_now"]}))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 30_882.25)
    levels_cli.main(["--markets", "NQ"])
    assert seen == {"NQ": 30_882.25}


def test_three_markets_fit_in_one_message():
    payload = levels_cli.build_message([sample_levels(symbol=s, sources=("QQQ", "^NDX"), own_horizons=OWN_HORIZONS,
                                                      busiest_calls=[24_200.0, 24_300.0], busiest_puts=[23_800.0, 23_700.0])
                                        for s in ("NQ", "ES", "YM")],
                                       {"NQ": 31_400.0, "ES": 6_800.0, "YM": 46_000.0},
                                       pd.Timestamp("2026-10-06").date())
    assert fits_discord(payload)


def test_horizons_ending_together_share_a_column():
    card = levels_cli.market_embed(sample_levels(horizons=[
        lv.HorizonLevels(covers=("day", "week"), end=pd.Timestamp("2026-10-09"), net_gamma=1e9, flip=np.nan,
                         call_wall=24_200.0, put_wall=23_800.0, move_pct=0.01)]), None)
    assert card["fields"][0]["name"] == "Today & This week"


def test_card_without_horizons_shows_every_expiry():
    card = levels_cli.market_embed(sample_levels(horizons=[]), None)
    assert [f["name"] for f in card["fields"]] == ["Mood", "Ceiling (call wall)", "Pivot (gamma flip)", "Floor (put wall)"]


def test_card_says_when_open_interest_was_refreshed():
    morning = pd.Timestamp("2026-10-06 09:15", tz="America/New_York")
    card = levels_cli.market_embed(sample_levels(open_interest_time=morning), None)
    assert card["description"] == "In NQ prices. NDX options read Mon 05 Oct 14:30 New York, open interest from Tue 09:15."


def test_card_without_ratio_shows_underlying_levels():
    today = HORIZONS[0]
    card = levels_cli.market_embed(sample_levels(ratio=np.nan, horizons=[
        lv.HorizonLevels(**(vars(today) | {"flip": np.nan}))]), None)
    lines = card["fields"][0]["value"].split("\n")
    assert "Ceiling **24,200.00** (NDX)" in lines and "Pivot none within ±15%" in lines


def test_zero_dte_message():
    payload = levels_cli.build_message([sample_zero_dte(), sample_zero_dte(symbol="YM", is_today=False,
                                                                           expiry="2026-10-09")],
                                       {"NQ": 31_210.0}, pd.Timestamp("2026-10-05").date())
    json.dumps(payload)
    assert payload["content"] == "**0DTE gamma update · 14:30 New York, Mon 05 Oct**"
    nq, ym = payload["embeds"]
    assert nq["title"] == "NQ 0DTE gamma · 14:30 New York" and "today's expiry (0DTE)" in nq["description"]
    fields = {f["name"]: f["value"] for f in nq["fields"]}
    assert fields["Mood (0DTE)"] == levels_cli.CALM
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


TUESDAY_0915 = pd.Timestamp("2026-10-06 13:15", tz="UTC")  # 09:15 New York


def passing_levels(symbol="NQ", **changes):
    """A card for Tuesday 06 Oct that passes every check: all its chains, measured at 24,000."""
    from bsnn import instruments
    sources = tuple(lv.option_sources(instruments.resolve(symbol), "etf"))
    return sample_levels(**({"symbol": symbol, "sources": sources, "measured_at": 24_000.0,
                             "session": pd.Timestamp("2026-10-06")} | changes))


@pytest.mark.parametrize("now, posting_day", [
    ("2026-10-06 13:15", "2026-10-06"),  # 09:15 New York: today
    ("2026-10-06 21:00", "2026-10-07"),  # 17:00, after the close: tomorrow
    ("2026-10-10 15:00", "2026-10-12"),  # Saturday: Monday
])
def test_posting_day_follows_the_clock(monkeypatch, now, posting_day):
    seen = []
    monkeypatch.setattr(levels_cli, "now_utc", lambda: pd.Timestamp(now, tz="UTC"))
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer, **kw: seen.append(before))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 31_400.0)
    levels_cli.main(["--markets", "NQ"])
    levels_cli.main(["--markets", "NQ", "--include-today"])  # still accepted, no longer changes anything
    assert seen == [pd.Timestamp(posting_day).date()] * 2


def test_post_requires_webhook(monkeypatch, capsys):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(levels_cli, "now_utc", lambda: TUESDAY_0915)
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer, **kw: passing_levels(symbol))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 31_400.0)
    assert levels_cli.main(["--markets", "NQ", "--post"]) == 1
    assert "DISCORD_WEBHOOK_URL" in capsys.readouterr().err


@pytest.fixture
def fake_post(monkeypatch, tmp_path):
    sent = {}
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    monkeypatch.setattr(levels_cli.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(levels_cli, "now_utc", lambda: TUESDAY_0915)
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer, **kw: passing_levels(symbol))
    monkeypatch.setattr(levels_cli, "live_zero_dte", lambda symbol, prefer: sample_zero_dte(symbol=symbol))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 31_400.0)
    monkeypatch.setattr(levels_cli, "post", lambda url, payload: None if is_copy(payload) else sent.update(url=url, payload=payload))
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


def test_card_failing_a_check_is_left_out_and_fails_the_run(fake_post, monkeypatch, capsys):
    sent, _ = fake_post
    wrong_side = passing_levels("ES", measured_at=23_000.0)  # below every put wall
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer, **kw:
                        wrong_side if symbol == "ES" else passing_levels(symbol))
    assert levels_cli.main(["--markets", "NQ", "ES", "--post"]) == 1
    assert [e["title"] for e in sent["payload"]["embeds"]] == ["NQ options levels"]
    assert "ES left out: day put wall 23,800.00 isn't below price 23,000.00" in capsys.readouterr().err


def test_nothing_is_posted_when_every_card_fails(fake_post, monkeypatch):
    sent, _ = fake_post
    monkeypatch.setattr(levels_cli, "levels_for",
                        lambda symbol, before, prefer, **kw: passing_levels(symbol, sources=(symbol,)))
    assert levels_cli.main(["--markets", "NQ", "--post"]) == 1 and sent == {}


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


@pytest.mark.parametrize("taken, session", [
    ("2026-10-12 10:35", "2026-10-12"),  # a live chain on Monday morning: Monday's levels
    ("2026-10-09 16:30", "2026-10-12"),  # Friday after the close: Monday's
    ("2026-10-08 16:00", "2026-10-09"),  # at the close: the next day's
    ("2026-10-10 11:00", "2026-10-12"),  # Saturday: Monday's
])
def test_session_for(taken, session):
    assert lv.session_for(pd.Timestamp(taken, tz="America/New_York")) == pd.Timestamp(session)


def test_live_chain_without_a_date_gives_todays_levels():
    raw = make_chain(ticker="QQQ", snapshot="2026-10-05T14:35:00+00:00", spot=100.0,  # Monday 10:35 New York
                     expiries=("2026-10-05", "2026-10-09")).assign(futuresPrice=4150.0)
    levels = lv.compute_levels("NQ", raw)
    assert levels.session == pd.Timestamp("2026-10-05")
    assert levels.horizons[0].covers == ("day",) and not levels.horizons[0].nearest_expiry


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


def test_each_horizon_prices_the_move_to_its_expiry():
    from bsnn.features import build_dataset
    levels = lv.compute_levels("NQ", horizon_chain())
    assert levels.session == pd.Timestamp("2026-10-05")  # the Monday after the snapshot
    moves = analytics.implied_moves(build_dataset(horizon_chain(), lv.FILTERS)).set_index("expiry")
    # Friday evening to Monday's close is about three calendar days, weekend included, as the options price it.
    day, week, month = levels.horizons
    assert day.move_pct == pytest.approx(moves.at["2026-10-05", "move_pct"])
    assert week.move_pct == pytest.approx(moves.at["2026-10-09", "move_pct"])
    assert month.move_pct == pytest.approx(moves.at["2026-10-30", "move_pct"])


def two_chains(snapshot=FRIDAY_CLOSE):
    """QQQ and NDX chains of the same market, NDX quoted 41 times higher with a tenth of the contracts."""
    expiries = ("2026-10-05", "2026-10-09", "2026-10-30")
    qqq = make_chain(ticker="QQQ", snapshot=snapshot, spot=100.0, expiries=expiries).assign(futuresPrice=4150.0)
    ndx = make_chain(ticker="^NDX", snapshot=snapshot, spot=100.0, expiries=expiries)
    for col in ("strike", "bid", "ask", "lastPrice", "underlyingPrice"):
        ndx[col] = ndx[col] * 41
    ndx["openInterest"] = ndx["openInterest"] / 10
    return qqq, ndx


def test_levels_add_up_every_chain_in_the_first_chains_prices():
    qqq, ndx = two_chains()
    alone = lv.compute_levels("NQ", qqq)
    both = lv.compute_levels("NQ", qqq, others=[ndx])
    assert both.source == "QQQ" and both.sources == ("QQQ", "^NDX") and both.spot == pytest.approx(100.0)
    # The same smile with a tenth of the contracts at 41x the notional: about 4.1x QQQ's gamma, in the same places.
    from bsnn.features import build_dataset
    own, other = (build_dataset(chain, lv.FILTERS) for chain in (qqq, ndx))
    call_gamma = lambda dataset: analytics.gamma_by_strike(dataset)["call"].sum()  # noqa: E731
    ndx_gamma = call_gamma(analytics.in_units_of(other, 100.0))
    assert ndx_gamma == pytest.approx(call_gamma(own) * 4.1, rel=0.05)
    assert call_gamma(lv.combine(own, [other])) == pytest.approx(call_gamma(own) + ndx_gamma, rel=1e-9)
    assert both.horizons[1].call_wall == pytest.approx(alone.horizons[1].call_wall)
    # This chain's net gamma only crosses zero falling (calm below, fast above), so neither has a pivot
    # (analytics._pivot); adding NDX must not invent one.
    assert both.horizons[1].flip == pytest.approx(alone.horizons[1].flip, rel=1e-3, nan_ok=True)
    assert both.horizons[1].put_wall == pytest.approx(alone.horizons[1].put_wall)
    assert [h.net_gamma for h in both.own_horizons] == [h.net_gamma for h in alone.horizons]
    assert alone.sources == ("QQQ",) and alone.own_horizons is alone.horizons


def test_levels_are_measured_at_the_current_price():
    at_close = lv.compute_levels("NQ", horizon_chain())  # QQQ 100, NQ 4,150 at the snapshot
    lower = lv.compute_levels("NQ", horizon_chain(), price_now=4150.0 * 0.98)  # NQ has since fallen 2%
    assert np.isnan(at_close.measured_at) and lower.measured_at == pytest.approx(98.0)
    assert lower.spot == pytest.approx(100.0)  # the snapshot's price is still reported as such
    for h in lower.horizons + lower.own_horizons:
        assert not h.put_wall >= 98.0 and not h.call_wall <= 98.0  # a wall may be "none" (NaN), never on the wrong side
    assert lower.horizons[1].call_wall < at_close.horizons[1].call_wall  # the call wall follows price down


def test_live_chain_with_quotes_replaces_the_snapshot(snapshots):
    live = make_chain(ticker="QQQ", snapshot="2026-10-08T15:30:00+00:00", spot=103.0,  # 11:30 New York
                      expiries=("2026-10-16", "2026-11-20"))
    with patch.object(lv, "futures_ratio", return_value=41.5), \
         patch.object(lv.market_data, "fetch_option_chain", return_value=live):
        levels = lv.levels_for("NQ", before="2026-10-08", directory=snapshots, refresh_oi=True)
    assert levels.spot == pytest.approx(103.0) and levels.open_interest_time is None
    assert levels.snapshot_time == pd.Timestamp("2026-10-08 11:30", tz="America/New_York")


def test_levels_for_adds_the_index_chain_from_the_same_day(tmp_path):
    qqq, ndx = two_chains()
    for chain in (qqq, ndx):
        market_data.save_option_snapshot(chain, tmp_path)
    market_data.save_option_snapshot(two_chains("2026-10-01T20:30:00+00:00")[1], tmp_path)  # an older NDX
    with patch.object(lv, "futures_ratio", return_value=41.5):
        levels = lv.levels_for("NQ", before="2026-10-05", directory=tmp_path)
    assert levels.sources == ("QQQ", "^NDX")


def test_levels_for_leaves_out_a_chain_from_another_day(snapshots, capsys):
    with patch.object(lv, "futures_ratio", return_value=41.5):
        levels = lv.levels_for("NQ", before="2026-10-07", directory=snapshots)  # QQQ Tuesday, NDX Monday
    assert levels.sources == ("QQQ",)
    assert "no ^NDX snapshot from 2026-10-06" in capsys.readouterr().err


def test_friday_shares_a_column_for_the_day_and_week():
    from bsnn.features import build_dataset
    dataset = build_dataset(horizon_chain(), lv.FILTERS)
    horizons = lv.compute_horizons(dataset, "2026-10-09")
    assert [h.covers for h in horizons] == [("day", "week"), ("month",)]


def test_horizon_without_an_expiry_uses_the_nearest():
    from bsnn.features import build_dataset
    raw = make_chain(ticker="DIA", snapshot=FRIDAY_CLOSE, expiries=("2026-10-07", "2026-11-20"))
    dataset = build_dataset(raw, lv.FILTERS)
    day, week, month = lv.compute_horizons(dataset, "2026-10-05")
    assert day.nearest_expiry == "2026-10-07" and week.nearest_expiry == "" and month.nearest_expiry == ""
    wednesday = analytics.implied_moves(dataset).set_index("expiry").at["2026-10-07", "move_pct"]
    assert day.move_pct == pytest.approx(wednesday * np.sqrt(1 / 3))  # Monday is one of its three sessions


def test_levels_for_dates_the_horizons_from_the_posting_day(snapshots):
    with patch.object(lv, "futures_ratio", return_value=41.5):
        levels = lv.levels_for("NQ", before="2026-10-10", directory=snapshots)  # a Saturday
    assert levels.session == pd.Timestamp("2026-10-12")
    assert [h.covers for h in levels.horizons][-1] == ("month",)


def test_levels_are_listed_top_to_bottom_like_the_chart():
    above_ceiling = lv.HorizonLevels(covers=("day",), end=pd.Timestamp("2026-10-05"), net_gamma=-1e9, flip=24_300.0,
                                     call_wall=24_200.0, put_wall=np.nan, move_pct=0.01)
    lines = levels_cli.market_embed(sample_levels(ratio=np.nan, horizons=[above_ceiling]), None)["fields"][0]["value"]
    assert lines.split("\n")[1:4] == ["Pivot **24,300.00** (NDX)", "Ceiling **24,200.00** (NDX)", "Floor none within ±15%"]


ALL_SLOTS = (levels_cli.DAILY_SLOT,) + levels_cli.UPDATE_SLOTS


@pytest.mark.parametrize("when, posted, expected", [
    ("2026-10-06 13:39", "", "daily"),                                 # 09:39 New York: still the morning card
    ("2026-10-06 14:15", "", "daily"),                                 # 10:15: a late morning card still posts
    ("2026-10-06 13:45", "2026-10-06 daily", None),                    # no 09:45 update any more
    ("2026-10-06 14:45", "2026-10-06 daily", "update-10:45"),          # just before the 10:50 macro
    ("2026-10-06 18:45", "", "update-14:45"),                          # 14:45, after the PM macro window opens
    ("2026-10-06 18:50", "2026-10-06 update-14:45", None),             # already updated
    ("2026-10-06 16:00", "", None),                                    # 12:00: between slots
    ("2026-10-06 16:45", "2026-10-06 update-10:45", "update-12:45"),   # 12:45, before the NY PM session
    ("2026-10-06 17:35", "", None),                                    # 13:35: the 12:45 window has closed
    ("2026-12-08 15:45", "", "update-10:45"),                          # winter: 10:45 New York = 15:45 UTC
])
def test_update_slots(tmp_path, when, posted, expected):
    marker = tmp_path / "levels_posted.txt"
    marker.write_text(posted + "\n")
    slot, _ = levels_cli.due_slot(pd.Timestamp(when, tz="UTC"), ALL_SLOTS, marker)
    assert (slot.name if slot else None) == expected


def test_slots_never_overlap():
    slots = sorted(ALL_SLOTS, key=lambda s: s.start)
    assert all(a.end <= b.start for a, b in zip(slots, slots[1:]))


def test_todays_busiest_strikes_sit_in_todays_ladder():
    card = levels_cli.market_embed(sample_levels(busiest_calls=[24_200.0, 24_300.0], busiest_puts=[23_900.0]), 31_400.0)
    today, week, month = card["fields"]
    assert "**31,590**  Busy strike · pulls price into the close" in today["value"].split("\n")
    assert "**31,460**  Ceiling · may stall here (busy strike)" in today["value"].split("\n")
    assert "**31,070**  Busy strike · pulls price into the close" in today["value"].split("\n")
    assert "Busy" not in week["value"] + month["value"]  # 0DTE volume says nothing about later expiries


def _chain(volume_call, volume_put, expiry="2026-10-06"):
    return pd.DataFrame({"K": [100.0, 101.0, 99.0, 98.0], "is_call": [1, 1, 0, 0],
                         "volume": [volume_call, 5.0, volume_put, 0.0],
                         "expiry": [expiry] * 4, "snapshot": ["2026-10-06 10:40"] * 4})


def test_busiest_strikes_come_only_from_a_chain_read_that_session():
    live = pd.Timestamp("2026-10-06 10:40", tz="America/New_York")
    assert lv.busiest_today(_chain(50.0, 70.0), live, pd.Timestamp("2026-10-06")) == ([100.0, 101.0], [99.0])
    yesterday = pd.Timestamp("2026-10-05 16:30", tz="America/New_York")  # its volume is Monday's
    assert lv.busiest_today(_chain(50.0, 70.0), yesterday, pd.Timestamp("2026-10-06")) == ([], [])
    no_expiry_today = _chain(50.0, 70.0, expiry="2026-10-07")
    assert lv.busiest_today(no_expiry_today, live, pd.Timestamp("2026-10-06")) == ([], [])


def test_tradingview_line_has_every_column_in_futures_points():
    as_of = pd.Timestamp("2026-10-06 13:45", tz="UTC")  # 09:45 New York
    line = levels_cli.tradingview_line(sample_levels(contract="NQZ26.CME", session=pd.Timestamp("2026-10-06")),
                                       31_400.0, as_of)
    head, day, week, month = line.split(" | ")
    assert head == "NQ Z26 2026-10-06 09:45"
    tokens = day.split()
    assert tokens[:2] == ["D", "C"]
    ceiling, pivot, floor, exp_lo, exp_hi, out_lo, out_hi = map(float, tokens[2:])
    assert (ceiling, pivot, floor) == (31_460.0, 30_810.0, 30_940.0)  # NDX levels x 1.3
    assert out_hi - 31_400 == pytest.approx(31_400 * 0.0102, abs=0.25)  # to the nearest quarter point
    assert exp_hi - 31_400 == pytest.approx((out_hi - 31_400) * 0.79788, abs=0.25)
    assert exp_lo < 31_400 < exp_hi and out_lo < exp_lo
    assert week.split()[:2] == ["W", "F"] and month.startswith("M ")


def test_tradingview_line_marks_missing_values_and_merged_columns():
    merged = lv.HorizonLevels(covers=("day", "week"), end=pd.Timestamp("2026-10-09"), net_gamma=-1e9, flip=np.nan,
                              call_wall=24_200.0, put_wall=23_800.0, move_pct=0.01)
    line = levels_cli.tradingview_line(sample_levels(horizons=[merged]), None, None)
    assert line == "NQ - - - | DW F 31460 - 30940 - - - -"
    assert levels_cli.tradingview_line(sample_levels(ratio=np.nan), 31_400.0, None) is None


def test_copy_message_holds_one_line_per_market_and_the_card_none():
    levels = [sample_levels(symbol=s) for s in ("NQ", "ES", "YM")]
    prices = {"NQ": 31_400.0, "ES": 6_800.0, "YM": 46_000.0}
    as_of = pd.Timestamp("2026-10-06 13:45", tz="UTC")
    copy = levels_cli.copy_message(levels, prices, as_of)
    block = copy["content"].split("```")[1].strip().split("\n")
    assert [ln.split()[0] for ln in block] == ["NQ", "ES", "YM"] and fits_discord(copy)
    assert "```" not in levels_cli.build_message(levels, prices, pd.Timestamp("2026-10-06").date(), as_of=as_of)["content"]
    assert levels_cli.copy_message([sample_levels(ratio=np.nan)], prices, as_of) is None


def test_refreshed_open_interest_carries_whose_occ_file_it_is():
    saved = make_chain(ticker="^NDX", snapshot=WEDNESDAY, expiries=("2026-10-16",))
    saved[market_data.OCC_DAY], saved[market_data.OCC_READ] = "2026-10-06", "2026-10-07T13:15:00+00:00"
    morning = make_chain(ticker="^NDX", snapshot=THURSDAY_MORNING, expiries=("2026-10-16",))
    morning["openInterest"] = np.arange(len(morning), dtype=float)
    morning[market_data.OCC_DAY], morning[market_data.OCC_READ] = "2026-10-07", "2026-10-08T13:15:00+00:00"
    raw, _ = lv.refresh_open_interest(saved, morning)
    assert set(raw[market_data.OCC_DAY]) == {"2026-10-07"} and set(raw[market_data.OCC_READ]) == {"2026-10-08T13:15:00+00:00"}


def test_card_warns_when_index_open_interest_is_not_the_latest():
    note = "NDX open interest may be a day old (the OCC's latest file was for Thu 19 Nov)"
    card = levels_cli.market_embed(sample_levels(oi_notes=(note,)), None)
    assert card["description"].endswith("New York.\n⚠️ " + note + ".")
    assert "⚠️" not in levels_cli.market_embed(sample_levels(), None)["description"]
