import numpy as np
import pandas as pd
import pytest

from bsnn import analytics, pricing
from bsnn.features import Filters, build_dataset

from conftest import make_chain

LOOSE = Filters(min_days=0.05, min_moneyness=0.5, max_moneyness=2.0, max_relative_spread=1.0, min_price=0.001)


@pytest.fixture
def flat():
    """A chain with no smile: every contract priced at 22% vol."""
    return build_dataset(make_chain(skew=0.0), LOOSE)


def test_implied_moves_from_flat_vol(flat):
    moves = analytics.implied_moves(flat)
    assert list(moves["expiry"]) == sorted(moves["expiry"]) and len(moves) == 4
    first = moves.iloc[0]
    T = first["days"] / 365
    assert first["atm_strike"] == 100 and first["atm_iv"] == pytest.approx(0.22, rel=1e-4)
    assert first["move"] == pytest.approx(100 * 0.22 * np.sqrt(T), rel=1e-4)
    call = pricing.price(100, 100, T, 0.04, 0.22, 0.01, True)
    put = pricing.price(100, 100, T, 0.04, 0.22, 0.01, False)
    assert first["straddle"] == pytest.approx(call + put, rel=1e-6)
    # The straddle is about 0.8 of the one-sigma move (sqrt(2/pi) for a normal distribution).
    assert first["straddle"] / first["move"] == pytest.approx(np.sqrt(2 / np.pi), abs=0.03)
    assert first["low"] == pytest.approx(100 - first["move"]) and first["high"] == pytest.approx(100 + first["move"])


def test_implied_moves_skip_expiries_without_a_strike_near_spot(flat):
    far_only = flat[(flat["K"] <= 90) | (flat["K"] >= 110)]
    assert analytics.implied_moves(far_only).empty


def test_one_day_move_skips_same_day_expiry(flat):
    moves = analytics.implied_moves(flat).assign(days=[0.5, 30, 60, 90])
    day = analytics.one_day_move(moves)
    assert day["expiry"] == moves["expiry"].iloc[1]
    assert day["move"] == pytest.approx(100 * moves["atm_iv"].iloc[1] / np.sqrt(252))


SNAPSHOT_DAY = "2026-01-05"  # make_chain's default snapshot, a Monday


def test_horizon_move_on_an_expiry_is_that_expirys_move(flat):
    moves = analytics.implied_moves(flat)
    first = moves.iloc[0]
    move = analytics.horizon_move(moves, SNAPSHOT_DAY, first["expiry"])
    assert move["expiry"] == first["expiry"] and move["move"] == pytest.approx(first["move"])


def test_horizon_move_takes_the_horizons_share_of_the_sessions(flat):
    moves = analytics.implied_moves(flat)
    first = moves.iloc[0]  # Friday 20 February
    friday_before = pd.Timestamp(first["expiry"]) - pd.Timedelta(days=7)  # no expiry that early: uses the first
    move = analytics.horizon_move(moves, SNAPSHOT_DAY, friday_before)
    assert move["expiry"] == first["expiry"]
    assert move["move"] == pytest.approx(first["move"] * np.sqrt(30 / 35))  # 30 of the expiry's 35 sessions


def test_sessions_through_skips_weekends():
    assert analytics.sessions_through("2026-10-12", "2026-10-16") == 5  # Monday to Friday
    assert analytics.sessions_through("2026-10-09", "2026-10-12") == 2  # Friday and Monday
    assert analytics.sessions_through("2026-10-10", "2026-10-10") == 1  # a Saturday still counts as one


def test_horizon_move_uses_todays_expiry_until_its_last_half_hour(flat):
    moves = analytics.implied_moves(flat).assign(days=[0.25, 30, 60, 90])  # six hours to today's close
    assert analytics.horizon_move(moves, SNAPSHOT_DAY, moves["expiry"].iloc[0])["expiry"] == moves["expiry"].iloc[0]
    moves = moves.assign(days=[0.01, 30, 60, 90])  # a quarter of an hour left
    assert analytics.horizon_move(moves, SNAPSHOT_DAY, moves["expiry"].iloc[0])["expiry"] == moves["expiry"].iloc[1]
    assert analytics.horizon_move(moves, SNAPSHOT_DAY, moves["expiry"].iloc[3])["expiry"] == moves["expiry"].iloc[3]


def test_in_units_of_keeps_dollar_gamma(flat):
    scaled = analytics.in_units_of(flat, 4100.0)
    assert scaled["S"].iloc[0] == pytest.approx(4100.0) and scaled["K"].max() == pytest.approx(flat["K"].max() * 41)
    assert analytics._gex(scaled, 4100.0) == pytest.approx(analytics._gex(flat, 100.0), rel=1e-9)
    back = analytics.in_units_of(scaled, 100.0)
    assert back["contract_size"].to_numpy() == pytest.approx(analytics.CONTRACT_SIZE)


def test_gamma_by_strike_matches_formula(flat):
    one = flat[(flat["K"] == 100) & (flat["expiry"] == flat["expiry"].min())]
    by_strike = analytics.gamma_by_strike(one)
    call = one[one["is_call"] == 1].iloc[0]
    gamma = pricing.greeks(100, 100, call["time_to_expiry"], 0.04, call["market_iv"], 0.01, True)["gamma"]
    assert by_strike.loc[100, "call"] == pytest.approx(gamma * 1000 * 100 * 100**2 * 0.01, rel=1e-6)
    assert by_strike.loc[100, "put"] == pytest.approx(-by_strike.loc[100, "call"], rel=1e-6)
    assert by_strike.loc[100, "net"] == pytest.approx(0, abs=1e-6)


def test_gamma_levels_find_walls_and_flip(flat):
    # Call open interest only above spot, put open interest only below: dealers are long
    # gamma up there and short gamma down here, so net gamma flips sign in between.
    oi = np.where(flat["is_call"] == 1, np.where(flat["K"] >= 105, 5000.0, 0.0),
                  np.where(flat["K"] <= 95, 5000.0, 0.0))
    df = flat.assign(open_interest=oi)
    levels = analytics.gamma_levels(df)
    assert levels["call_wall"] >= 105 and levels["put_wall"] <= 95
    assert 95 < levels["flip"] < 105
    spots, curve = levels["curve"]
    assert curve[0] < 0 < curve[-1]


def test_walls_sit_above_and_below_spot(flat):
    # Heavy open interest at the money (100) would dominate gamma; the walls must skip it.
    oi = np.where(flat["K"] == 100, 50_000.0, 1000.0)
    levels = analytics.gamma_levels(flat.assign(open_interest=oi))
    assert levels["call_wall"] > 100 and levels["put_wall"] < 100
    assert analytics.gamma_by_strike(flat.assign(open_interest=oi))["call"].idxmax() == 100


def test_no_flip_when_gamma_never_changes_sign(flat):
    calls_only = flat.assign(open_interest=np.where(flat["is_call"] == 1, 1000.0, 0.0))
    levels = analytics.gamma_levels(calls_only)
    assert levels["net"] > 0 and np.isnan(levels["flip"]) and np.isnan(levels["put_wall"])


def test_zero_crossing_picks_nearest_root():
    x = np.linspace(0, 10, 101)
    y = np.sin(x)  # roots at pi, 2pi, 3pi
    assert analytics._zero_crossing_nearest(x, y, 6.0) == pytest.approx(2 * np.pi, abs=0.01)


def test_front_expiry_prefers_today():
    today = build_dataset(make_chain(expiries=("2026-01-05", "2026-01-16")), LOOSE)
    front, is_today = analytics.front_expiry(today)
    assert is_today and set(front["expiry"]) == {"2026-01-05"}
    later = build_dataset(make_chain(expiries=("2026-01-09", "2026-01-16")), LOOSE)
    front, is_today = analytics.front_expiry(later)
    assert not is_today and set(front["expiry"]) == {"2026-01-09"}


def test_volume_by_strike(flat):
    one = flat[flat["expiry"] == flat["expiry"].min()]
    volume = analytics.volume_by_strike(one)
    assert volume["call"].idxmax() == 100 and volume["put"].idxmax() == 100
    assert volume.loc[100, "call"] == one[(one["K"] == 100) & (one["is_call"] == 1)]["volume"].sum()


def test_within_days(flat):
    assert analytics.within_days(flat, None) is flat
    assert (analytics.within_days(flat, 50)["time_to_expiry"] * 365 <= 50).all()
