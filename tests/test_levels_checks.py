import numpy as np
import pandas as pd
import pytest

from bsnn import levels as lv
from bsnn.levels_checks import problems

SESSION = pd.Timestamp("2026-10-06")  # Tuesday


def horizon(covers, end, call_wall=24_200.0, put_wall=23_800.0, move_pct=0.01):
    return lv.HorizonLevels(covers=covers, end=pd.Timestamp(end), net_gamma=1e9, flip=23_900.0,
                            call_wall=call_wall, put_wall=put_wall, move_pct=move_pct)


def good(**changes):
    """NQ levels for Tuesday that pass every check."""
    fields = dict(symbol="NQ", source="QQQ", snapshot_time=pd.Timestamp("2026-10-05 16:30", tz="America/New_York"),
                  spot=24_100.0, ratio=1.0, net_gamma=1e9, flip=23_900.0, call_wall=24_200.0, put_wall=23_800.0,
                  one_day_move_pct=0.01, session=SESSION, sources=("QQQ", "^NDX"), measured_at=24_000.0,
                  horizons=[horizon(("day",), "2026-10-06"), horizon(("week",), "2026-10-09", move_pct=0.02),
                            horizon(("month",), "2026-10-30", call_wall=np.nan, move_pct=0.04)])
    return lv.MarketLevels(**(fields | changes))


def test_good_levels_pass():
    assert problems(good(), SESSION) == []


def test_levels_for_another_day_fail():
    assert problems(good(), pd.Timestamp("2026-10-07")) == ["levels are for Tue 06 Oct, not Wed 07 Oct"]


def test_old_snapshot_fails():
    old = good(snapshot_time=pd.Timestamp("2026-09-30 16:30", tz="America/New_York"))
    assert problems(old, SESSION) == ["the options were read 6 days before Tue 06 Oct"]


def test_missing_chain_fails():
    assert problems(good(sources=("QQQ",)), SESSION) == ["NDX options are missing"]
    assert problems(good(source="^NDX", sources=("^NDX", "QQQ")), SESSION, prefer="index") == []


def test_levels_must_be_measured_at_the_current_price():
    # The walls are then judged against the snapshot's price (24,100), which sits between them.
    assert problems(good(measured_at=np.nan), SESSION) == ["no current futures price to measure the levels at"]


@pytest.mark.parametrize("price, expected", [
    (24_250.0, "day call wall 24,200.00 isn't above price 24,250.00"),
    (23_700.0, "day put wall 23,800.00 isn't below price 23,700.00"),
])
def test_walls_on_the_wrong_side_of_price_fail(price, expected):
    assert expected in problems(good(measured_at=price), SESSION)


def test_the_etfs_own_walls_are_checked_too():
    own = [horizon(("day",), "2026-10-06", put_wall=24_050.0)]
    assert problems(good(own_horizons=own), SESSION) == ["QQQ day put wall 24,050.00 isn't below price 24,000.00"]


def test_a_move_out_of_bounds_or_shrinking_fails():
    levels = good()
    levels.horizons[0].move_pct = 0.03  # wider today than this week
    assert problems(levels, SESSION) == ["1σ doesn't widen from day to week to month"]
    levels.horizons[0].move_pct = np.nan
    assert problems(levels, SESSION) == ["day 1σ of nan% is outside 0.05%-25%"]


def test_no_horizons_fail():
    assert "no day, week or month levels" in problems(good(horizons=[]), SESSION)
