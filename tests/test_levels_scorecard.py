import json

import numpy as np
import pandas as pd
import pytest

from conftest import fits_discord
from scipy.stats import norm

from bsnn import levels_scorecard as sc


def bars(days: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    """5-minute cash-session bars: each day's (high, low, close), with the extremes at midday."""
    frames = []
    for day, (high, low, close) in days.items():
        index = pd.date_range(f"{day} 09:30", f"{day} 15:55", freq="5min", tz="America/New_York")
        frame = pd.DataFrame({"High": close, "Low": close, "Close": close}, index=index)
        frame.iloc[30, 0], frame.iloc[40, 1] = high, low
        frames.append(frame)
    return pd.concat(frames)


def card(**changes) -> pd.Series:
    """Posted at 1,000 with a 1% move (σ = 10): ceiling 1,010 (1σ up), floor 995 (½σ down)."""
    return pd.Series({"session": "2026-11-09", "symbol": "NQ", "covers": "day", "end": "2026-11-09", "price": 1000.0,
                      "move_pct": 0.01, "ceiling": 1010.0, "floor": 995.0} | changes)


def test_ceiling_held_floor_broke_and_chance_for_each():
    result = sc.score_card(card(), bars({"2026-11-09": (1008.0, 990.0, 1003.0)}))
    assert result["ceiling_held"] == 1.0 and result["floor_held"] == 0.0
    assert result["ceiling_chance"] == pytest.approx(2 * norm.cdf(1.0) - 1)  # 68%: a 1σ level holds that often anyway
    assert result["floor_chance"] == pytest.approx(2 * norm.cdf(0.5) - 1)
    assert result["inside_expected"] == 1.0  # closed 3 away, inside ±8


def test_close_outside_the_expected_range_and_a_missing_wall():
    result = sc.score_card(card(ceiling=np.nan), bars({"2026-11-09": (1009.0, 999.0, 1009.0)}))
    assert result["inside_expected"] == 0.0 and np.isnan(result["ceiling_held"]) and np.isnan(result["ceiling_chance"])


def test_an_unfinished_horizon_is_not_scored():
    week = card(covers="week", end="2026-11-13")
    assert sc.score_card(week, bars({"2026-11-09": (1008.0, 990.0, 1003.0)})) is None
    midday = bars({"2026-11-09": (1008.0, 990.0, 1003.0)}).iloc[:40]  # read before the close
    assert sc.score_card(card(), midday) is None


def test_a_week_column_uses_every_session_to_its_end():
    week = card(covers="week", end="2026-11-10")
    result = sc.score_card(week, bars({"2026-11-09": (1005.0, 997.0, 1001.0), "2026-11-10": (1012.0, 999.0, 1011.0)}))
    assert result["ceiling_held"] == 0.0 and result["floor_held"] == 1.0


def write_record(directory, day, kinds_and_ceilings):
    lines = []
    for kind, ceiling in kinds_and_ceilings:
        lines.append(json.dumps({"kind": kind, "markets": [{
            "symbol": "NQ", "session": day, "futures_price": 1000.0, "horizons": [
                {"covers": ["day"], "end": day, "move_pct": 0.01, "call_wall_futures": ceiling,
                 "put_wall_futures": 995.0}]}]}))
    (directory / f"{day}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_only_the_days_first_scheduled_card_is_scored(tmp_path):
    write_record(tmp_path, "2026-11-09", [("manual", 1100.0), ("scheduled", 1010.0), ("update", 1020.0),
                                          ("scheduled", 1030.0)])
    cards = sc.morning_cards(tmp_path)
    assert cards["ceiling"].tolist() == [1010.0]


def test_scores_are_saved_once_and_kept(tmp_path):
    records, store = tmp_path / "levels", tmp_path / "levels_scores.csv"
    records.mkdir()
    write_record(records, "2026-11-09", [("scheduled", 1010.0)])
    calls = []

    def bars_for(symbol):
        calls.append(symbol)
        return bars({"2026-11-09": (1008.0, 990.0, 1003.0), "2026-11-10": (1000.0, 1000.0, 1000.0)})

    first = sc.update_scores(store, records, bars_for)
    assert len(first) == 1 and first["ceiling_held"].iloc[0] == 1.0 and store.exists()
    again = sc.update_scores(store, records, bars_for)  # nothing new to score: no download
    assert len(again) == 1 and calls == ["NQ=F"]  # no contract recorded: the continuous symbol


def test_a_failed_download_leaves_the_scores_as_they_were(tmp_path):
    records = tmp_path / "levels"
    records.mkdir()
    write_record(records, "2026-11-09", [("scheduled", 1010.0)])

    def broken(symbol):
        raise OSError("Yahoo is down")

    assert sc.update_scores(tmp_path / "scores.csv", records, broken).empty


@pytest.mark.parametrize("session, expected", [
    ("2026-11-09", ("week", "2026-11-02", "2026-11-06")),   # Monday: last week
    ("2026-11-02", ("month", "2026-10-01", "2026-10-31")),  # first weekday of November: last month wins
    ("2026-12-01", ("month", "2026-11-01", "2026-11-30")),  # a Tuesday that starts the month
    ("2026-11-10", None),
])
def test_scorecard_period(session, expected):
    period = sc.scorecard_period(session)
    assert (period if period is None else (period[0], f"{period[1]:%Y-%m-%d}", f"{period[2]:%Y-%m-%d}")) == expected


def scores(rows):
    return pd.DataFrame(rows, columns=sc.COLUMNS)


def test_weekly_card_counts_against_chance_and_reports_mondays_week_levels():
    week = [("2026-11-0" + d, s, "day", "2026-11-0" + d, 1.0, 0.6, 0.0, 0.5, 1.0) for d in "23456" for s in ("NQ", "ES")]
    week.append(("2026-11-02", "NQ", "week", "2026-11-06", 0.0, 0.5, 1.0, 0.5, 0.0))
    embed = sc.scorecard_embed(scores(week), "week", pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-06"))
    assert embed["title"] == "Last week's scorecard (Mon 02 Nov – Fri 06 Nov)"
    lines = embed["description"].split("\n")
    assert lines[0] == "Ceiling held: **10 of 10** · chance alone: about 6"
    assert lines[1] == "Floor held: **0 of 10** · chance alone: about 5"
    assert lines[2] == "Closed inside expected range: **10 of 10** · chance alone: about 6"
    assert lines[3] == "Week levels from the Mon 02 Nov card: NQ ceiling broke, floor held"
    assert lines[4].startswith("Since 02 Nov, the day levels: ")


def test_no_scorecard_without_scored_days():
    assert sc.scorecard_embed(scores([]), "week", pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-06")) is None


@pytest.mark.parametrize("held, expected", [
    (1.0, "better than chance"), (0.0, "worse than chance"), (None, "not proven yet")])
def test_verdict_needs_a_clear_gap_after_allowing_for_markets_moving_together(held, expected):
    n = 60  # 20 days x 3 markets
    rng = np.random.default_rng(1)
    hits = [held if held is not None else float(rng.random() < 0.5) for _ in range(n)]
    days = scores([(f"d{i // 3}", "NQ ES YM".split()[i % 3], "day", "", h, 0.5, np.nan, np.nan, 1.0)
                   for i, h in enumerate(hits)])
    assert expected in sc.verdict(days)


def test_mondays_card_ends_with_last_weeks_scorecard_and_fits_discord(tmp_path, monkeypatch):
    from bsnn import levels_cli
    from test_levels import OWN_HORIZONS, sample_levels

    week = [("2026-11-0" + d, s, "day", "2026-11-0" + d, 1.0, 0.6, 0.0, 0.5, 1.0) for d in "23456" for s in ("NQ", "ES", "YM")]
    week += [("2026-11-02", s, "week", "2026-11-06", 0.0, 0.5, 1.0, 0.5, 0.0) for s in ("NQ", "ES", "YM")]
    scores(week).to_csv(tmp_path / "levels_scores.csv", index=False)
    monkeypatch.setattr(levels_cli.paths, "DATA_DIR", tmp_path)
    review = levels_cli.scorecard(pd.Timestamp("2026-11-09"))
    assert review["title"].startswith("Last week's scorecard")
    assert levels_cli.scorecard(pd.Timestamp("2026-11-10")) is None  # Tuesday: no review
    payload = levels_cli.build_message(
        [sample_levels(symbol=s, sources=("QQQ", "^NDX"), own_horizons=OWN_HORIZONS,
                       busiest_calls=[24_200.0, 24_300.0], busiest_puts=[23_800.0, 23_700.0]) for s in ("NQ", "ES", "YM")],
        {"NQ": 31_400.0, "ES": 6_800.0, "YM": 46_000.0}, pd.Timestamp("2026-11-09").date(), review=review)
    assert payload["embeds"][-1] is review and "footer" in payload["embeds"][-2]
    assert fits_discord(payload)


def test_a_broken_scorecard_never_stops_the_post(tmp_path, monkeypatch):
    from bsnn import levels_cli

    monkeypatch.setattr(levels_cli.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(levels_cli.levels_scorecard, "update_scores", lambda *a, **k: 1 / 0)
    assert levels_cli.scorecard(pd.Timestamp("2026-11-09")) is None
