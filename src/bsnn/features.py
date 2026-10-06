"""Turn option-chain snapshots into a modelling dataset.

Each row is one quoted contract. The target is the bid/ask mid divided by the
strike, which keeps it on a similar scale across tickers and price levels; the
features are scale-free for the same reason.

Implied volatility is solved here from the mid using the snapshot's own rate and
dividend yield. Yahoo's ``impliedVolatility`` column is not used: it assumes a
zero rate and zero dividends, which skews calls high and puts low.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bsnn import pricing
from bsnn.instruments import ASSET_CLASSES, SINGLE_STOCK, asset_class_of
from bsnn.market_data import years_to_expiry  # noqa: F401  (re-exported; it used to live here)

FEATURES = ["log_moneyness", "time_to_expiry", "risk_free_rate", "dividend_yield", "hist_vol", "is_call"]
ATM_IV_FEATURE = "atm_iv"
ASSET_CLASS_FEATURES = {cls: "class_" + cls.lower().replace(" ", "_") for cls in ASSET_CLASSES}


def feature_columns(use_atm_iv: bool) -> list[str]:
    return FEATURES + [ATM_IV_FEATURE] if use_atm_iv else list(FEATURES)


def add_asset_class_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One 0/1 column per asset class, from ``df["asset_class"]``."""
    return df.assign(**{col: (df["asset_class"] == cls).astype(float) for cls, col in ASSET_CLASS_FEATURES.items()})


@dataclass(frozen=True)
class Filters:
    """Which quotes are clean enough to learn from."""

    min_days: float = 1.0
    max_days: float = 730.0
    min_moneyness: float = 0.7  # spot / strike
    max_moneyness: float = 1.3
    max_relative_spread: float = 0.5  # (ask - bid) / mid
    min_price: float = 0.05


def single_contract(S, K, T, r, q, hist_vol, is_call, atm_iv=np.nan, asset_class=SINGLE_STOCK) -> pd.DataFrame:
    """A one-row dataset for pricing a contract that isn't in a snapshot."""
    return add_asset_class_columns(pd.DataFrame({
        "S": [S], "K": [K], "log_moneyness": [np.log(S / K)], "time_to_expiry": [T],
        "risk_free_rate": [r], "dividend_yield": [q], "hist_vol": [hist_vol],
        "is_call": [float(is_call)], ATM_IV_FEATURE: [atm_iv], "asset_class": [asset_class],
    }))


def at_the_money_iv(df: pd.DataFrame) -> pd.Series:
    """For each (ticker, snapshot, expiry), the mean implied vol at the strike nearest spot."""
    groups = ["ticker", "snapshot", "expiry"]
    distance = df["log_moneyness"].abs()
    nearest = distance == distance.groupby([df[g] for g in groups]).transform("min")
    atm = df[nearest].groupby(groups)["market_iv"].mean().rename(ATM_IV_FEATURE)
    return df[groups].join(atm, on=groups)[ATM_IV_FEATURE]


def build_dataset(chains: pd.DataFrame, filters: Filters = Filters()) -> pd.DataFrame:
    """One row per usable contract: features, benchmark inputs, ``market_iv`` and ``target``."""
    bid = chains["bid"].to_numpy(dtype=float)
    ask = chains["ask"].to_numpy(dtype=float)
    S = chains["underlyingPrice"].to_numpy(dtype=float)
    K = chains["strike"].to_numpy(dtype=float)
    df = pd.DataFrame({
        "ticker": chains["ticker"].to_numpy(),
        "asset_class": chains["ticker"].map(asset_class_of).to_numpy(),
        "snapshot": chains["snapshotTime"].to_numpy(),
        "expiry": chains["ExpiryDate"].astype(str).to_numpy(),
        "contract": chains["contractSymbol"].to_numpy() if "contractSymbol" in chains else "",
        "S": S,
        "K": K,
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
        "log_moneyness": np.log(S / K),
        "time_to_expiry": years_to_expiry(chains["snapshotTime"], chains["ExpiryDate"]),
        "risk_free_rate": chains["riskFreeRate"].to_numpy(dtype=float),
        "dividend_yield": chains["dividendYield"].to_numpy(dtype=float),
        "hist_vol": chains["histVol"].to_numpy(dtype=float),
        "is_call": (chains["OptionType"] == "Call").to_numpy(dtype=float),
        "open_interest": chains["openInterest"].fillna(0).to_numpy(dtype=float) if "openInterest" in chains else 0.0,
    })

    days = df["time_to_expiry"] * 365
    moneyness = df["S"] / df["K"]
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = (
            (df["bid"] > 0)
            & (df["ask"] >= df["bid"])
            & (df["mid"] >= filters.min_price)
            & ((df["ask"] - df["bid"]) / df["mid"] <= filters.max_relative_spread)
            & days.between(filters.min_days, filters.max_days)
            & moneyness.between(filters.min_moneyness, filters.max_moneyness)
            & np.isfinite(df[FEATURES + ["mid"]]).all(axis=1)
            & (df["hist_vol"] > 0)
        )
    df = df[keep].reset_index(drop=True)

    df["market_iv"] = pricing.implied_volatility(
        df["mid"], df["S"], df["K"], df["time_to_expiry"], df["risk_free_rate"], df["dividend_yield"],
        df["is_call"] > 0)
    # Mids with no implied vol break no-arbitrage bounds (often early-exercise premium on
    # American contracts); a European model can't fit them, so they're dropped.
    df = df[np.isfinite(df["market_iv"])].reset_index(drop=True)
    df[ATM_IV_FEATURE] = at_the_money_iv(df).to_numpy()
    df["target"] = df["mid"] / df["K"]
    return add_asset_class_columns(df)
