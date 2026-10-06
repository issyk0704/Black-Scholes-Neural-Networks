import numpy as np
import pandas as pd
import pytest

from bsnn import pricing


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
