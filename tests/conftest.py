import numpy as np
import pandas as pd
import pytest

from bsnn import contracts, levels_scorecard, pricing


@pytest.fixture(autouse=True)
def no_scorecard_downloads(monkeypatch):
    """Tests never download futures bars for the scorecard; a test that needs bars passes its own."""
    monkeypatch.setattr(levels_scorecard, "fetch_bars", lambda symbol: pd.DataFrame(columns=["High", "Low", "Close"]))


@pytest.fixture(autouse=True)
def no_open_interest_downloads(monkeypatch):
    """Tests never ask Yahoo for contract open interest, so futures are priced from the continuous
    symbol (NQ=F) unless a test supplies open interest itself."""
    def unavailable(ticker):
        raise OSError("no network in tests")

    monkeypatch.setattr(contracts, "fetch_open_interest", unavailable)
    contracts._active.cache_clear()
    yield
    contracts._active.cache_clear()


@pytest.fixture
def history():
    """A year of synthetic daily prices with one dividend and a 2-for-1 split."""
    dates = pd.bdate_range("2025-01-01", periods=260)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                       "Volume": 1_000_000, "Dividends": 0.0, "Stock Splits": 0.0}, index=dates)
    df.index.name = "Date"
    df.loc[dates[100], "Dividends"] = 2.0
    df.loc[dates[200], "Stock Splits"] = 2.0
    return df


def make_chain(snapshot="2026-01-05T15:00:00+00:00", spot=100.0, ticker="TEST",
               expiries=("2026-02-20", "2026-03-20", "2026-06-18", "2026-09-18"), rate=0.04, q=0.01,
               hist_vol=0.2, skew=0.15):
    """A synthetic option chain whose quotes come from a skewed Black-Scholes vol."""
    rows = []
    for expiry in expiries:
        T = (pd.Timestamp(expiry + " 16:00", tz="America/New_York") - pd.Timestamp(snapshot)).total_seconds() \
            / (365 * 24 * 3600)
        for K in np.arange(75, 126, 2.5):
            vol = hist_vol * 1.1 - skew * np.log(spot / K) * -1
            for kind in ("Call", "Put"):
                mid = pricing.price(spot, K, T, rate, vol, q, kind == "Call")
                rows.append({"contractSymbol": f"{ticker}{expiry}{kind[0]}{K}", "strike": K,
                             "bid": mid * 0.98, "ask": mid * 1.02, "lastPrice": mid, "impliedVolatility": vol,
                             "openInterest": 1000.0, "volume": float(1000 - abs(K - spot) * 10),
                             "OptionType": kind, "ExpiryDate": expiry})
    return pd.DataFrame(rows).assign(ticker=ticker, snapshotTime=snapshot, underlyingPrice=spot,
                                     riskFreeRate=rate, dividendYield=q, histVol=hist_vol)


@pytest.fixture
def chain():
    return make_chain()


def fits_discord(payload: dict) -> bool:
    """Within Discord's limits: 2,000 characters of message text, 6,000 of embed text (titles,
    descriptions, field names and values, footers), 10 embeds."""
    def text(e):
        return (len(e.get("title", "")) + len(e.get("description", "")) + len(e.get("footer", {}).get("text", ""))
                + sum(len(f["name"]) + len(f["value"]) for f in e.get("fields", [])))
    embeds = payload.get("embeds", [])
    return len(payload.get("content", "")) <= 2000 and sum(text(e) for e in embeds) <= 6000 and len(embeds) <= 10


def is_copy(payload: dict) -> bool:
    """The small message under the card with the TradingView lines (no embeds)."""
    return payload.get("content", "").startswith("📋") and not payload.get("embeds")
