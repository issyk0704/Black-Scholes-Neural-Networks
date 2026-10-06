"""The markets we trade, and where the app gets prices and option data for each.

Yahoo Finance has prices for futures, FX and yields but no option chains for
them, so each instrument names a listed ETF whose options stand in for its own
("options proxy"). The proxy's smile is close to the real one for index futures
(SPY vs ES) and only indicative for bonds (TLT vs ZB differ in duration).
"""

from __future__ import annotations

from dataclasses import dataclass

EQUITY_INDEX, SINGLE_STOCK, RATES, METALS, FX = "Equity index", "Single stock", "Rates", "Metals", "FX"
ASSET_CLASSES = (EQUITY_INDEX, SINGLE_STOCK, RATES, METALS, FX)

BSM, BLACK76, GARMAN_KOHLHAGEN = "bsm", "black76", "garman_kohlhagen"


@dataclass(frozen=True)
class Instrument:
    symbol: str  # as on the TradingView watchlist
    name: str
    asset_class: str
    price_ticker: str  # Yahoo symbol for the price history
    model: str | None  # option pricing model; None for yields, which aren't option underlyings here
    options_proxy: str  # Yahoo symbol with listed options
    proxy_is_thin: bool = False  # too few strikes/expiries to learn much from
    vol_index: str | None = None  # Yahoo symbol of the matching implied-vol index
    futures_ticker: str | None = None  # FX only: futures used to infer the foreign interest rate
    is_yield: bool = False  # quoted in percent; changes are measured in basis points
    note: str = ""


INSTRUMENTS = [
    Instrument("NQ", "Nasdaq-100 E-mini", EQUITY_INDEX, "NQ=F", BLACK76, "QQQ", vol_index="^VXN"),
    Instrument("ES", "S&P 500 E-mini", EQUITY_INDEX, "ES=F", BLACK76, "SPY", vol_index="^VIX"),
    Instrument("YM", "Dow E-mini", EQUITY_INDEX, "YM=F", BLACK76, "DIA", vol_index="^VXD"),
    Instrument("RTY", "Russell 2000 E-mini", EQUITY_INDEX, "RTY=F", BLACK76, "IWM"),
    Instrument("DXY", "US Dollar Index", FX, "DX-Y.NYB", BLACK76, "UUP", proxy_is_thin=True,
               note="Yahoo has the index, not DX futures; the index level stands in for the futures price."),
    Instrument("EU", "EUR/USD", FX, "EURUSD=X", GARMAN_KOHLHAGEN, "FXE", proxy_is_thin=True, futures_ticker="6E=F"),
    Instrument("GU", "GBP/USD", FX, "GBPUSD=X", GARMAN_KOHLHAGEN, "FXB", proxy_is_thin=True, futures_ticker="6B=F"),
    Instrument("XAUUSD", "Gold spot", METALS, "GC=F", BLACK76, "GLD", vol_index="^GVZ",
               note="Yahoo has no spot gold feed; front-month GC futures are used instead."),
    Instrument("XAGUSD", "Silver spot", METALS, "SI=F", BLACK76, "SLV",
               note="Yahoo has no spot silver feed; front-month SI futures are used instead."),
    Instrument("GC", "Gold futures", METALS, "GC=F", BLACK76, "GLD", vol_index="^GVZ"),
    Instrument("SI", "Silver futures", METALS, "SI=F", BLACK76, "SLV"),
    Instrument("ZB", "30-year T-Bond futures", RATES, "ZB=F", BLACK76, "TLT", vol_index="^MOVE"),
    Instrument("ZT", "2-year T-Note futures", RATES, "ZT=F", BLACK76, "SHY", proxy_is_thin=True, vol_index="^MOVE"),
    Instrument("ZN", "10-year T-Note futures", RATES, "ZN=F", BLACK76, "IEF", vol_index="^MOVE"),
    # Yahoo's only 2-year series (2YY=F yield futures) barely trades and has stale prints, so this
    # comes from the Federal Reserve's FRED database instead: official, but one business day behind.
    Instrument("US02Y", "US 2-year yield", RATES, "FRED:DGS2", None, "SHY", proxy_is_thin=True, vol_index="^MOVE",
               is_yield=True, note="From FRED (DGS2), published one business day late."),
    Instrument("US10Y", "US 10-year yield", RATES, "^TNX", None, "IEF", vol_index="^MOVE", is_yield=True),
    Instrument("US30Y", "US 30-year yield", RATES, "^TYX", None, "TLT", vol_index="^MOVE", is_yield=True),
]

_BY_SYMBOL = {i.symbol: i for i in INSTRUMENTS}
_CLASS_BY_PROXY = {i.options_proxy: i.asset_class for i in INSTRUMENTS}


def resolve(symbol: str) -> Instrument | None:
    """The watchlist instrument for ``symbol``, or None for a plain Yahoo ticker such as AAPL."""
    return _BY_SYMBOL.get(symbol.strip().upper())


def option_proxies() -> list[str]:
    """Every distinct options proxy, in watchlist order."""
    return list(dict.fromkeys(i.options_proxy for i in INSTRUMENTS))


def asset_class_of(ticker: str) -> str:
    """Asset class of an option underlying: a known proxy's class, otherwise a single stock."""
    return _CLASS_BY_PROXY.get(ticker.upper(), SINGLE_STOCK)


def label(instrument: Instrument) -> str:
    return f"{instrument.symbol} — {instrument.name}"
