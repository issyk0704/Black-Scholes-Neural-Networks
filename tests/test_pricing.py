import numpy as np
import pytest

from bsnn import pricing

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def test_textbook_values():
    # Hull, Options Futures and Other Derivatives: S=K=100, T=1, r=5%, sigma=20%
    assert pricing.price(S, K, T, R, SIGMA, is_call=True) == pytest.approx(10.4506, abs=1e-4)
    assert pricing.price(S, K, T, R, SIGMA, is_call=False) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    call = pricing.price(S, 110, 0.5, R, SIGMA, q, is_call=True)
    put = pricing.price(S, 110, 0.5, R, SIGMA, q, is_call=False)
    assert call - put == pytest.approx(S * np.exp(-q * 0.5) - 110 * np.exp(-R * 0.5))


def test_vectorised_matches_scalar():
    strikes = np.array([80.0, 100.0, 120.0])
    is_call = np.array([True, False, True])
    vector = pricing.price(S, strikes, T, R, SIGMA, 0.01, is_call)
    scalar = [pricing.price(S, k, T, R, SIGMA, 0.01, c) for k, c in zip(strikes, is_call)]
    np.testing.assert_allclose(vector, scalar)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call):
    q, h = 0.02, 1e-4
    g = pricing.greeks(S, 105, 0.75, R, SIGMA, q, is_call)

    def p(s=S, t=0.75, r=R, vol=SIGMA):
        return pricing.price(s, 105, t, r, vol, q, is_call)

    assert g["price"] == pytest.approx(p())
    assert g["delta"] == pytest.approx((p(s=S + h) - p(s=S - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((p(s=S + 0.01) - 2 * p() + p(s=S - 0.01)) / 0.01**2, rel=1e-3)
    assert g["vega"] == pytest.approx((p(vol=SIGMA + h) - p(vol=SIGMA - h)) / (2 * h) / 100, rel=1e-5)
    assert g["rho"] == pytest.approx((p(r=R + h) - p(r=R - h)) / (2 * h) / 100, rel=1e-5)
    # theta is the change as one day passes, i.e. time to expiry shrinks
    assert g["theta"] == pytest.approx(-(p(t=0.75 + h) - p(t=0.75 - h)) / (2 * h) / 365, rel=1e-5)


def test_implied_volatility_round_trip():
    strikes = np.array([70.0, 90.0, 100.0, 115.0, 140.0])
    vols = np.array([0.45, 0.3, 0.22, 0.18, 0.25])
    is_call = np.array([False, False, True, True, True])
    prices = pricing.price(S, strikes, 0.4, R, vols, 0.01, is_call)
    solved = pricing.implied_volatility(prices, S, strikes, 0.4, R, 0.01, is_call)
    np.testing.assert_allclose(solved, vols, rtol=1e-6)


def test_implied_volatility_scalar_and_impossible_prices():
    price = pricing.price(S, K, T, R, 0.3)
    assert pricing.implied_volatility(price, S, K, T, R) == pytest.approx(0.3)
    # Below intrinsic value / above the spot: no volatility fits.
    assert np.isnan(pricing.implied_volatility(1.0, S, 50, T, R))
    assert np.isnan(pricing.implied_volatility(S + 1, S, K, T, R))


@pytest.mark.parametrize("bad", [{"S": 0}, {"K": -1}, {"T": 0}, {"sigma": 0}, {"sigma": float("nan")}])
def test_rejects_invalid_inputs(bad):
    args = {"S": S, "K": K, "T": T, "r": R, "sigma": SIGMA} | bad
    with pytest.raises(ValueError):
        pricing.price(**args)


def test_payoff():
    np.testing.assert_allclose(pricing.payoff([90, 100, 110], 100, True), [0, 0, 10])
    np.testing.assert_allclose(pricing.payoff([90, 100, 110], 100, False), [10, 0, 0])
