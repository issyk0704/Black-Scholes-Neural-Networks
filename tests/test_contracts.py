import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bsnn import contracts, levels, levels_cli, levels_record, levels_scorecard


def test_expiry_is_the_third_friday():
    assert contracts.expiry(2026, 12) == dt.date(2026, 12, 18)
    assert contracts.expiry(2027, 3) == dt.date(2027, 3, 19)


@pytest.mark.parametrize("day, expected", [
    ("2026-11-13", ["NQZ26.CME", "NQH27.CME"]),
    ("2026-12-18", ["NQZ26.CME", "NQH27.CME"]),  # expiry day: Dec still listed (it settles at the open)
    ("2026-12-19", ["NQH27.CME", "NQM27.CME"]),
])
def test_quarterly_contracts(day, expected):
    assert contracts.quarterly_contracts("NQ", pd.Timestamp(day).date()) == expected


def test_dow_contracts_trade_on_cbot():
    assert contracts.quarterly_contracts("YM", dt.date(2026, 11, 13)) == ["YMZ26.CBT", "YMH27.CBT"]


@pytest.mark.parametrize("front_oi, next_oi, expected", [
    (276_207, 4_731, "NQZ26.CME"),     # today: Dec holds almost all the open interest
    (150_000, 150_000, "NQZ26.CME"),   # level: stay until the next one is higher
    (120_000, 180_000, "NQH27.CME"),   # Mar has overtaken Dec: switch
])
def test_switch_only_once_the_next_contracts_open_interest_is_higher(monkeypatch, front_oi, next_oi, expected):
    oi = {"NQZ26.CME": front_oi, "NQH27.CME": next_oi}
    monkeypatch.setattr(contracts, "fetch_open_interest", lambda t: oi[t])
    assert contracts.active_contract("NQ", "2026-12-10") == expected


def test_without_open_interest_or_a_rule_use_the_continuous_symbol():
    assert contracts.futures_ticker("NQ", "2026-12-10") == "NQ=F"  # tests can't read open interest
    assert contracts.active_contract("GC", "2026-12-10") is None


@pytest.mark.parametrize("ticker, name", [("NQZ26.CME", "Dec 26"), ("YMH27.CBT", "Mar 27"), ("NQ=F", "front")])
def test_contract_name(ticker, name):
    assert contracts.contract_name(ticker) == name


def snapshot(ticker=None):
    raw = pd.DataFrame({"underlyingPrice": [100.0], "futuresPrice": [4_150.0], "snapshotTime": ["2026-12-09T21:30:00Z"]})
    return raw.assign(futuresTicker=ticker) if ticker else raw


def test_ratio_is_unchanged_while_the_contract_is_the_same():
    nq = levels.instruments.resolve("NQ")
    assert levels.futures_ratio(nq, snapshot("NQZ26.CME"), "NQZ26.CME") == pytest.approx(41.5)
    assert levels.futures_ratio(nq, snapshot("NQZ26.CME")) == pytest.approx(41.5)


def test_ratio_moves_to_the_new_contract_on_the_switch_day(monkeypatch):
    prices = {"NQZ26.CME": 28_000.0, "NQH27.CME": 28_330.0, "NQ=F": 28_000.0}
    monkeypatch.setattr(levels.market_data, "latest_price", lambda t: prices[t])
    nq = levels.instruments.resolve("NQ")
    switched = levels.futures_ratio(nq, snapshot("NQZ26.CME"), "NQH27.CME")
    assert switched == pytest.approx(41.5 * 28_330 / 28_000)  # levels shift up with the contract, ~1.2%
    old = levels.futures_ratio(nq, snapshot(), "NQZ26.CME")  # an older snapshot priced NQ=F, the same contract
    assert old == pytest.approx(41.5)


def test_a_failed_switch_price_leaves_the_market_out(monkeypatch):
    def down(ticker):
        raise OSError("Yahoo is down")

    monkeypatch.setattr(levels.market_data, "latest_price", down)
    with pytest.raises(OSError):
        levels.futures_ratio(levels.instruments.resolve("NQ"), snapshot("NQZ26.CME"), "NQH27.CME")


def test_card_and_record_name_the_contract():
    from test_levels import sample_levels

    card_levels = sample_levels(source="QQQ", ratio=41.4, measured_at=745.74, contract="NQH27.CME")
    assert levels_cli.market_embed(card_levels, 30_900.0)["description"].startswith("In NQ Mar 27 prices. ")
    assert levels_record.market_record(card_levels, 30_900.0)["futures_ticker"] == "NQH27.CME"
    continuous = sample_levels(source="QQQ", ratio=41.4, measured_at=745.74)
    assert levels_cli.market_embed(continuous, 30_900.0)["description"].startswith("In NQ prices. ")


def test_prices_come_from_our_contract(monkeypatch):
    monkeypatch.setattr(contracts, "fetch_open_interest", {"NQZ26.CME": 1.0, "NQH27.CME": 2.0}.get)
    monkeypatch.setattr(levels_cli.market_data, "latest_price", {"NQH27.CME": 28_330.0}.get)
    monkeypatch.setattr(contracts.pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2026-12-10", tz=tz)))
    assert levels_cli.current_prices(["NQ"]) == {"NQ": 28_330.0}


def test_each_card_is_scored_against_the_contract_it_was_posted_in(tmp_path):
    import json

    (tmp_path / "2026-12-10.jsonl").write_text(json.dumps({"kind": "scheduled", "markets": [{
        "symbol": "NQ", "session": "2026-12-10", "futures_price": 28_330.0, "futures_ticker": "NQH27.CME",
        "horizons": [{"covers": ["day"], "end": "2026-12-10", "move_pct": 0.01, "call_wall_futures": 28_500.0,
                      "put_wall_futures": 28_100.0}]}]}) + "\n", encoding="utf-8")
    assert levels_scorecard.morning_cards(tmp_path)["ticker"].tolist() == ["NQH27.CME"]
