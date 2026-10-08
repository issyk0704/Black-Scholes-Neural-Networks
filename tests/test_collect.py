import pandas as pd
import pytest

from bsnn import collect_cli, market_data

from conftest import make_chain


@pytest.mark.parametrize("when, expected", [
    ("2026-10-06 14:30", True),   # Tuesday 10:30 New York (UTC-4)
    ("2026-10-06 13:30", False),  # 09:30: too close to the open
    ("2026-10-06 20:30", False),  # 16:30: after the close
    ("2026-10-10 15:00", False),  # Saturday
])
def test_us_session_open(when, expected):
    assert market_data.us_session_open(pd.Timestamp(when, tz="UTC")) is expected


@pytest.mark.parametrize("when, expected", [
    ("2026-10-06 20:30", True),   # 16:30 New York: after the close, quotes hold their closing values
    ("2026-10-06 13:30", False),  # 09:30: too close to the open
    ("2026-10-07 00:30", False),  # 20:30: SPX's overnight session has started
    ("2026-10-10 20:30", False),  # Saturday
])
def test_quotes_reliable(when, expected):
    assert market_data.quotes_reliable(pd.Timestamp(when, tz="UTC")) is expected


def test_collect_saves_live_in_session_chains_only(tmp_path):
    def fake_fetch(ticker, save):
        assert save is False
        if ticker == "BAD":
            raise ConnectionError("timeout")
        if ticker == "NIGHT":  # well quoted, but overnight (SPX trades round the clock on Cboe)
            return make_chain(ticker=ticker, snapshot="2026-01-05T10:30:00+00:00")
        chain = make_chain(ticker=ticker)  # 10:00 New York
        return chain.assign(bid=0.0) if ticker == "STALE" else chain

    outcomes = collect_cli.collect(["GOOD", "STALE", "NIGHT", "BAD"], tmp_path, fetch=fake_fetch)
    assert outcomes["GOOD"].startswith("saved")
    assert outcomes["STALE"] == "skipped: only 0% of 168 contracts had a live quote"
    assert outcomes["NIGHT"].startswith("skipped: it was taken outside US trading hours")
    assert outcomes["BAD"] == "failed: timeout"
    assert [p.name for p in market_data.list_option_snapshots(tmp_path)] == ["GOOD_options_2026-01-05.csv.gz"]


def test_main_does_nothing_outside_market_hours(monkeypatch, tmp_path):
    monkeypatch.setattr(collect_cli.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(market_data, "quotes_reliable", lambda: False)
    monkeypatch.setattr(collect_cli, "collect", lambda *a, **k: pytest.fail("should not collect"))
    assert collect_cli.main([]) == 0
    assert "nothing collected" in (tmp_path / "collect.log").read_text()
