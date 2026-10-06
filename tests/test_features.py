import numpy as np
import pandas as pd
import pytest

from bsnn.features import ATM_IV_FEATURE, FEATURES, Filters, build_dataset, feature_columns, years_to_expiry

from conftest import make_chain


def test_years_to_expiry_runs_to_4pm_new_york():
    years = years_to_expiry(["2026-01-05T21:00:00+00:00"], ["2026-01-06"])
    assert years[0] * 365 * 24 == pytest.approx(24.0)


def test_dataset_recovers_true_implied_vol(chain):
    ds = build_dataset(chain)
    assert len(ds) > 0
    true_vol = 0.2 * 1.1 + 0.15 * ds["log_moneyness"]
    np.testing.assert_allclose(ds["market_iv"], true_vol, rtol=1e-5)
    np.testing.assert_allclose(ds["target"], ds["mid"] / ds["K"])
    assert np.isfinite(ds[feature_columns(True)]).all().all()


def test_atm_iv_is_shared_within_an_expiry(chain):
    ds = build_dataset(chain)
    per_expiry = ds.groupby("expiry")[ATM_IV_FEATURE].nunique()
    assert (per_expiry == 1).all()
    # The strike nearest spot is 100, where the synthetic smile is exactly 0.22.
    assert ds[ATM_IV_FEATURE].iloc[0] == pytest.approx(0.22, rel=1e-5)


def test_filters_drop_bad_quotes(chain):
    chain.loc[0, "bid"] = 0
    chain.loc[1, "ask"] = chain.loc[1, "bid"] * 3
    ds = build_dataset(chain)
    assert chain.loc[0, "contractSymbol"] not in set(ds["contract"])
    assert chain.loc[1, "contractSymbol"] not in set(ds["contract"])
    narrow = build_dataset(chain, Filters(min_moneyness=0.95, max_moneyness=1.05))
    assert (narrow["S"] / narrow["K"]).between(0.95, 1.05).all()


def test_multiple_snapshots_and_tickers():
    chains = pd.concat([make_chain(ticker="AAA"), make_chain(ticker="BBB", spot=110.0,
                                                             snapshot="2026-02-02T15:00:00+00:00")])
    ds = build_dataset(chains)
    assert set(ds["ticker"]) == {"AAA", "BBB"}
    assert list(FEATURES) == feature_columns(False)
