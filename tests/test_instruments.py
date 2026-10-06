import pytest

from bsnn import instruments, pricing

WATCHLIST = ["NQ", "ES", "YM", "RTY", "DXY", "EU", "GU", "XAUUSD", "XAGUSD", "GC", "SI", "ZB", "ZT", "ZN",
             "US02Y", "US10Y", "US30Y"]


def test_whole_watchlist_is_registered():
    assert [i.symbol for i in instruments.INSTRUMENTS] == WATCHLIST


@pytest.mark.parametrize("inst", instruments.INSTRUMENTS, ids=lambda i: i.symbol)
def test_instrument_is_consistent(inst):
    assert inst.asset_class in instruments.ASSET_CLASSES
    assert inst.model is None or inst.model in pricing.MODELS
    assert (inst.model is None) == inst.is_yield  # yields are the only markets without an option model
    assert (inst.model == "garman_kohlhagen") == (inst.futures_ticker is not None)
    assert inst.options_proxy and inst.price_ticker


def test_resolve_is_case_insensitive_and_ignores_plain_tickers():
    assert instruments.resolve(" nq ").price_ticker == "NQ=F"
    assert instruments.resolve("AAPL") is None


def test_option_proxies_are_distinct_and_ordered():
    proxies = instruments.option_proxies()
    assert len(proxies) == len(set(proxies)) == 12
    assert proxies[:4] == ["QQQ", "SPY", "DIA", "IWM"]


def test_asset_class_of_proxies_and_stocks():
    assert instruments.asset_class_of("spy") == instruments.EQUITY_INDEX
    assert instruments.asset_class_of("TLT") == instruments.RATES
    assert instruments.asset_class_of("GLD") == instruments.METALS
    assert instruments.asset_class_of("FXE") == instruments.FX
    assert instruments.asset_class_of("AAPL") == instruments.SINGLE_STOCK
