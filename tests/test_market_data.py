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


def test_a_thin_proxy_chain_is_kept_but_a_blanked_one_is_not(chain):
    inside = chain.assign(snapshotTime="2026-10-20T20:30:00+00:00")    # Tuesday 16:30 New York
    thin = inside.copy()
    thin.loc[thin.index[: int(len(thin) * 0.55)], "bid"] = 0.0           # 45% quoted, as UUP on 20 Oct 2026
    assert md.snapshot_problem(thin) is None
    assert "live quote" in md.snapshot_problem(inside.assign(bid=0.0))


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


OCC_TEXT = """Series Search Results for SPX

\t\tSeries/contract\t\tStrike\t\t\tOpen Interest\t\t\t
ProductSymbol\tyear\tMonth\tDay\tInteger\tDec\tC/P\tCall\tPut\tPosition Limit\t
SPXW  \t\t2026\t11\t20\t7400\t000\tC P \t1200\t50\t0
SPX  \t\t2026\t11\t20\t7300\t000\tC P \t10\t3400\t0
2SPX  \t\t2026\t11\t20\t7400\t000\t  P \t0\t9\t0
"""


def test_the_occ_report_reads_into_open_interest_per_contract():
    oi = md.occ_open_interest("^SPX", get_text=lambda symbol: OCC_TEXT)
    assert oi[("SPXW", "2026-11-20", "Call", 7400.0)] == 1200.0
    assert oi[("SPX", "2026-11-20", "Put", 7300.0)] == 3400.0
    assert not any(k[0] == "2SPX" for k in oi)                                   # adjusted contracts left out


def test_an_occ_report_without_the_index_options_says_so():
    with pytest.raises(ValueError, match=r"no \^NDX options"):
        md.occ_open_interest("^NDX", get_text=lambda symbol: "Series Search Results for NDX\n")


def test_yahoos_index_chain_takes_the_occs_open_interest():
    yahoo = pd.DataFrame({"contractSymbol": ["SPXW261120C07400000", "SPX261120P07300000", "SPXW261120P07000000"],
                          "strike": [7400.0, 7300.0, 7000.0], "OptionType": ["Call", "Put", "Put"],
                          "ExpiryDate": "2026-11-20", "openInterest": [0.0, 4.0, 7.0], "bid": [10.0, 8.0, 1.0]})
    oi = md.occ_open_interest("^SPX", get_text=lambda symbol: OCC_TEXT)
    filled, n = md.fill_open_interest(yahoo, oi)
    assert n == 2 and list(filled["openInterest"]) == [1200.0, 3400.0, 7.0]       # unmatched keeps Yahoo's
    assert list(filled["bid"]) == [10.0, 8.0, 1.0]                                  # prices untouched


def test_parity_yield_copes_with_two_roots_at_one_strike():
    rows = []
    for root_shift in (0.0, 0.1):                              # SPX (AM) and SPXW (PM) at the same strikes
        for k in (98.0, 100.0, 102.0):
            rows.append({"OptionType": "Call", "ExpiryDate": "2026-12-18", "strike": k, "bid": 3.0 + root_shift,
                         "ask": 3.2 + root_shift})
            rows.append({"OptionType": "Put", "ExpiryDate": "2026-12-18", "strike": k, "bid": 2.0 + root_shift,
                         "ask": 2.2 + root_shift})
    q = md.parity_dividend_yield(pd.DataFrame(rows), 100.0, 0.04, "2026-11-18T15:00:00+00:00")
    assert np.isfinite(q)


def test_the_morning_settled_monthlies_leave_the_chain_at_the_open_on_their_expiry_day():
    chain = pd.DataFrame({
        "contractSymbol": ["SPX261120C07000000", "SPXW261120C07000000", "SPX261218C07000000", "SPX261120P06900000"],
        "ExpiryDate": ["2026-11-20", "2026-11-20", "2026-12-18", "2026-11-20"],
        "OptionType": ["Call", "Call", "Call", "Put"], "strike": [7000.0, 7000.0, 7000.0, 6900.0]})
    before = md.drop_settled_at_open(chain, pd.Timestamp("2026-11-20 14:29", tz="UTC"))     # 09:29 NY
    after = md.drop_settled_at_open(chain, pd.Timestamp("2026-11-20 14:30", tz="UTC"))      # 09:30 NY
    day_before = md.drop_settled_at_open(chain, pd.Timestamp("2026-11-19 20:00", tz="UTC"))
    assert len(before) == 4 and len(day_before) == 4
    assert list(after["contractSymbol"]) == ["SPXW261120C07000000", "SPX261218C07000000"]   # SPXW and December stay


NY = "America/New_York"


@pytest.mark.parametrize("now, due", [
    ("2026-11-20 15:00", "2026-11-19"),      # Friday afternoon: Thursday evening's file
    ("2026-11-20 21:30", "2026-11-20"),      # after Friday's build
    ("2026-11-22 12:00", "2026-11-20"),      # the weekend: Friday's
    ("2026-11-23 09:15", "2026-11-20"),      # Monday's 09:15 card: Friday's
    ("2026-11-27 09:15", "2026-11-25"),      # the day after Thanksgiving: Wednesday's (no file for the holiday)
])
def test_the_occ_file_due_is_the_last_trading_day_built(now, due):
    assert md.occ_due(pd.Timestamp(now, tz=NY)) == pd.Timestamp(due).date()


def test_the_occ_file_day_is_its_earliest_expiry():
    oi = md.occ_open_interest("^SPX", get_text=lambda symbol: OCC_TEXT)
    assert md.occ_file_day(oi) == pd.Timestamp("2026-11-20").date()


def occ_chain(day: str, read: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["^NDX", "^NDX"], md.OCC_DAY: day, md.OCC_READ: pd.Timestamp(read, tz=NY).isoformat()})


def test_the_card_note_flags_a_stale_or_missing_occ_file_only():
    assert md.occ_note(occ_chain("2026-11-20", "2026-11-23 09:15")) is None            # Friday's file on Monday
    stale = md.occ_note(occ_chain("2026-11-19", "2026-11-23 09:15"))                   # Thursday's on Monday
    assert stale == "NDX open interest may be a day old (the OCC's latest file was for Thu 19 Nov)"
    assert "OCC was unavailable" in md.occ_note(occ_chain("", "2026-11-23 09:15"))
    assert md.occ_note(pd.DataFrame({"ticker": ["QQQ"]})) is None                        # an ETF / an old snapshot


def test_the_occ_columns_survive_a_saved_snapshot(tmp_path):
    for day, expect in (("2026-11-19", "a day old"), ("", "unavailable")):
        path = tmp_path / f"x{len(day)}.csv.gz"
        occ_chain(day, "2026-11-23 09:15").to_csv(path, index=False)
        assert expect in md.occ_note(pd.read_csv(path))
