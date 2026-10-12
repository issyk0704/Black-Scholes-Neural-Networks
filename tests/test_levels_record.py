import json

import numpy as np
import pandas as pd

from bsnn import levels as lv, levels_record


def test_each_post_appends_a_line_with_missing_values_as_null(tmp_path):
    day = lv.HorizonLevels(covers=("day",), end=pd.Timestamp("2026-10-06"), net_gamma=1e9, flip=np.nan,
                           call_wall=760.0, put_wall=np.nan, move_pct=0.01)
    levels = lv.MarketLevels(symbol="NQ", source="QQQ", snapshot_time=pd.Timestamp("2026-10-05 16:30", tz="America/New_York"),
                             spot=750.0, ratio=41.4, net_gamma=1e9, flip=np.nan, call_wall=760.0, put_wall=np.nan,
                             one_day_move_pct=0.01, session=pd.Timestamp("2026-10-06"), horizons=[day],
                             sources=("QQQ", "^NDX"), measured_at=745.0)
    when = pd.Timestamp("2026-10-06 13:15", tz="UTC")
    for kind in ("scheduled", "manual"):
        path = levels_record.append(tmp_path, "2026-10-06", when, kind, [levels], {"NQ": 30_843.0})
    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.name == "2026-10-06.jsonl" and (first["kind"], second["kind"]) == ("scheduled", "manual")
    (column,) = first["markets"][0]["horizons"]
    assert column["flip"] is None and column["put_wall"] is None and column["put_wall_futures"] is None
    assert column["call_wall_futures"] == 760.0 * 41.4
    assert (column["range_low"], column["range_high"]) == (30_843.0 - 308.43, 30_843.0 + 308.43)
