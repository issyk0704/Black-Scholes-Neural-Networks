from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bsnn import market_data as md


def test_market_dates_handles_offsets_and_dst():
    idx = md.market_dates(["2024-03-08 00:00:00-05:00", "2024-03-11 00:00:00-04:00"])
    assert list(idx) == [pd.Timestamp("2024-03-08"), pd.Timestamp("2024-03-11")]
    assert list(md.market_dates(["2024-03-08", "2024-03-11"])) == list(idx)


def test_unadjusted_close_undoes_later_splits(history):
    before_split = history.index[150]
    assert md.unadjusted_close(history, before_split) == pytest.approx(history["Close"].iloc[150] * 2)
    assert md.unadjusted_close(history) == pytest.approx(history["Close"].iloc[-1])


def test_realized_vol_is_annualised(history):
    vol = md.realized_vol(history["Close"]).dropna()
    assert vol.median() == pytest.approx(0.16, abs=0.02)  # daily sd 1% -> about 16% a year


def test_dividend_yield_uses_trailing_year(history):
    asof = history.index[150]
    assert md.dividend_yield(history, asof) == pytest.approx(2.0 / history["Close"].iloc[150])
    assert md.dividend_yield(history, history.index[50]) == 0.0


def test_history_round_trips_through_csv(tmp_path, history):
    fake = MagicMock()
    fake.return_value.history.return_value = history.tz_localize("America/New_York")
    with patch.object(md.yf, "Ticker", fake):
        fetched = md.fetch_history("abc", directory=tmp_path)
    loaded = md.load_history("ABC", directory=tmp_path)
    pd.testing.assert_frame_equal(fetched, loaded, check_freq=False, check_dtype=False)


def test_get_history_falls_back_to_cache(tmp_path, history):
    history.to_csv(md.history_path("ABC", tmp_path))
    with patch.object(md.yf, "Ticker", side_effect=ConnectionError("offline")):
        df = md.get_history("ABC", refresh=True, directory=tmp_path)
    assert len(df) == len(history)


def test_risk_free_rate_falls_back_when_offline():
    md._risk_free_rate.cache_clear()
    with patch.object(md.yf, "Ticker", side_effect=ConnectionError("offline")):
        assert md.risk_free_rate("2025-06-02") == md.DEFAULT_RATE
    md._risk_free_rate.cache_clear()


def test_enrich_snapshot_adds_market_inputs(history, chain):
    snapshot = history.index[220].tz_localize("America/New_York") + pd.Timedelta(hours=12)
    out = md.enrich_snapshot(chain.drop(columns=md.SNAPSHOT_COLUMNS), "abc", snapshot, spot=50.0,
                             history=history, rate=0.03)
    assert set(md.SNAPSHOT_COLUMNS) <= set(out.columns)
    assert (out["ticker"] == "ABC").all()
    assert out["histVol"].iloc[0] == pytest.approx(md.realized_vol(history["Close"]).iloc[220])
    assert md.option_snapshot_path("abc", snapshot, "x").name == f"ABC_options_{history.index[220]:%Y-%m-%d}.csv"


def test_load_option_snapshots_requires_market_inputs(tmp_path, chain):
    good, bad = tmp_path / "A_options_2026-01-05.csv", tmp_path / "B_options_2026-01-05.csv"
    chain.to_csv(good, index=False)
    chain.drop(columns=["histVol"]).to_csv(bad, index=False)
    assert len(md.load_option_snapshots([good])) == len(chain)
    with pytest.raises(ValueError, match="histVol"):
        md.load_option_snapshots([bad])
    assert md.list_option_snapshots(tmp_path) == [good, bad]


def test_committed_snapshots_are_loadable():
    files = md.list_option_snapshots()
    if not files:
        pytest.skip("no snapshots in data/options")
    df = md.load_option_snapshots(files)
    assert np.isfinite(df["underlyingPrice"]).all()
