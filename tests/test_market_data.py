from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bsnn import market_data as md

from conftest import make_chain


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


def test_fred_series_become_price_histories(tmp_path, monkeypatch):
    csv = "observation_date,DGS2\n2026-09-29,4.89\n2026-09-30,.\n2026-10-01,4.78\n2026-10-02,4.83\n"
    monkeypatch.setattr(md, "FRED_URL", str(tmp_path / "{series}.csv"))
    (tmp_path / "DGS2.csv").write_text(csv)
    df = md.fetch_history("FRED:DGS2", start="2026-09-30", directory=tmp_path)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-10-01", "2026-10-02"]  # missing values dropped
    assert df["Close"].tolist() == [4.78, 4.83] and (df["High"] == df["Close"]).all()
    assert md.history_path("FRED:DGS2", tmp_path).name == "FRED_DGS2_data.csv"
    assert md.load_history("FRED:DGS2", tmp_path)["Close"].tolist() == [4.78, 4.83]


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
    assert md.option_snapshot_path("abc", snapshot, "x").name == f"ABC_options_{history.index[220]:%Y-%m-%d}.csv.gz"


def test_load_option_snapshots_requires_market_inputs(tmp_path, chain):
    good, bad = tmp_path / "A_options_2026-01-05.csv", tmp_path / "B_options_2026-01-05.csv"
    chain.to_csv(good, index=False)
    chain.drop(columns=["histVol"]).to_csv(bad, index=False)
    assert len(md.load_option_snapshots([good])) == len(chain)
    with pytest.raises(ValueError, match="histVol"):
        md.load_option_snapshots([bad])
    assert md.list_option_snapshots(tmp_path) == [good, bad]


def test_compressed_snapshots_round_trip(tmp_path, chain):
    path = md.save_option_snapshot(chain, tmp_path)
    assert path.name == "TEST_options_2026-01-05.csv.gz"
    assert path.stat().st_size < len(chain.to_csv(index=False)) / 2
    pd.testing.assert_frame_equal(md.load_option_snapshots(md.list_option_snapshots(tmp_path)), chain)
    assert md.snapshot_label(path) == "TEST  2026-01-05"
    assert md.snapshot_label(tmp_path / "SPY_options_2026-10-05.csv") == "SPY  2026-10-05"


def test_fetch_does_not_save_chains_with_blank_quotes(tmp_path, chain):
    fake = MagicMock()
    fake.return_value.options = ("2026-02-20",)
    blank = chain[chain["ExpiryDate"] == "2026-02-20"].assign(bid=0.0, ask=0.0)
    fake.return_value.option_chain.return_value = MagicMock(calls=blank[blank["OptionType"] == "Call"],
                                                            puts=blank[blank["OptionType"] == "Put"])
    history = pd.DataFrame({"Close": 100 * np.exp(np.linspace(0, 0.1, 60))},
                           index=pd.bdate_range("2025-10-01", periods=60, tz="America/New_York"))
    fake.return_value.history.return_value = history
    fake.return_value.fast_info = {"lastPrice": 100.0}
    with patch.object(md.yf, "Ticker", fake), patch.object(md, "risk_free_rate", return_value=0.04):
        out = md.fetch_option_chain("TEST", directory=tmp_path)
    assert len(out) == len(blank)
    assert md.list_option_snapshots(tmp_path) == []


def test_snapshots_in_subfolders_are_found_once(tmp_path, chain):
    local = md.save_option_snapshot(chain, tmp_path)
    cloud = tmp_path / "cloud" / "data" / "options"
    md.save_option_snapshot(chain, cloud)  # same snapshot, collected again in the cloud
    only_cloud = md.save_option_snapshot(chain.assign(ticker="CLD"), cloud)
    assert md.list_option_snapshots(tmp_path) == [only_cloud, local]


@pytest.mark.parametrize("snapshot, ok", [
    ("2026-10-05T14:30:00+00:00", True),    # Monday 10:30 New York
    ("2026-10-05T23:30:00+00:00", True),    # Monday 19:30 New York: closing quotes, still fine
    ("2026-10-06T00:29:00+00:00", False),   # Monday 20:29 New York: SPX's overnight session is about to start
    ("2026-10-06T10:33:00+00:00", False),   # Tuesday 06:33 New York, before the open
    ("2026-10-10T15:00:00+00:00", False),   # Saturday
])
def test_snapshot_problem_checks_trading_hours(chain, snapshot, ok):
    assert (md.snapshot_problem(chain.assign(snapshotTime=snapshot)) is None) is ok


def test_live_quote_share(chain):
    assert md.live_quote_share(chain) == 1.0
    assert md.live_quote_share(chain.assign(bid=0.0)) == 0.0


def test_parity_dividend_yield_recovers_index_yield():
    chain = make_chain(q=0.013)
    assert md.parity_dividend_yield(chain, 100.0, 0.04, chain["snapshotTime"].iloc[0]) == pytest.approx(0.013, abs=1e-4)
    assert md.parity_dividend_yield(chain.assign(bid=0.0), 100.0, 0.04, chain["snapshotTime"].iloc[0]) == 0.0


def test_cash_index_snapshot_takes_yield_from_parity(history):
    snapshot = history.index[220].tz_localize("America/New_York") + pd.Timedelta(hours=12)
    chain = make_chain(snapshot=snapshot.isoformat(), q=0.013,
                       expiries=("2025-12-19", "2026-01-16")).drop(columns=md.SNAPSHOT_COLUMNS)
    out = md.enrich_snapshot(chain, "^SPX", snapshot, spot=100.0, history=history, rate=0.04)
    assert out["dividendYield"].iloc[0] == pytest.approx(0.013, abs=1e-4)
    assert md.is_cash_index("^SPX") and not md.is_cash_index("SPY")


def test_foreign_rate_from_futures():
    # EUR futures above spot means euro rates are below dollar rates.
    rate = md.foreign_rate_from_futures(spot=1.1241, futures_price=1.1271, years_to_expiry=0.19, domestic_rate=0.04)
    assert rate == pytest.approx(0.04 - np.log(1.1271 / 1.1241) / 0.19)
    assert rate < 0.04
    with pytest.raises(ValueError):
        md.foreign_rate_from_futures(1.12, 1.13, 0.0, 0.04)


def test_committed_snapshots_are_loadable():
    files = md.list_option_snapshots()
    if not files:
        pytest.skip("no snapshots in data/options")
    df = md.load_option_snapshots(files)
    assert np.isfinite(df["underlyingPrice"]).all()
