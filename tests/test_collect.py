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
    assert collect_cli.us_session_open(pd.Timestamp(when, tz="UTC")) is expected


def test_collect_saves_live_chains_and_skips_blank_ones(tmp_path):
    def fake_fetch(ticker, save):
        assert save is False
        if ticker == "BAD":
            raise ConnectionError("timeout")
        chain = make_chain(ticker=ticker)
        return chain.assign(bid=0.0) if ticker == "STALE" else chain

    outcomes = collect_cli.collect(["GOOD", "STALE", "BAD"], tmp_path, fetch=fake_fetch)
    assert outcomes["GOOD"].startswith("saved")
    assert outcomes["STALE"].startswith("skipped")
    assert outcomes["BAD"] == "failed: timeout"
    assert [p.name for p in market_data.list_option_snapshots(tmp_path)] == ["GOOD_options_2026-01-05.csv.gz"]


def test_main_does_nothing_outside_market_hours(monkeypatch, tmp_path):
    monkeypatch.setattr(collect_cli.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(collect_cli, "us_session_open", lambda: False)
    monkeypatch.setattr(collect_cli, "collect", lambda *a, **k: pytest.fail("should not collect"))
    assert collect_cli.main([]) == 0
    assert "closed" in (tmp_path / "collect.log").read_text()
