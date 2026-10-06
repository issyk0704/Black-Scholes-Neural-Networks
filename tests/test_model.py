import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras")

from bsnn.features import build_dataset, single_contract  # noqa: E402
from bsnn.model import BS_ATM, NN, OptionPricer, TrainConfig, split_dataset, train  # noqa: E402

from conftest import make_chain  # noqa: E402

EXPIRIES = ("2026-01-16", "2026-02-20", "2026-03-20", "2026-04-17", "2026-06-18", "2026-09-18", "2026-12-18")


@pytest.fixture(scope="module")
def dataset():
    chains = pd.concat([
        make_chain(ticker="AAA", expiries=EXPIRIES),
        make_chain(ticker="BBB", spot=105.0, hist_vol=0.3, snapshot="2026-01-06T15:00:00+00:00", expiries=EXPIRIES),
    ])
    return build_dataset(chains)


def test_expiry_split_keeps_expiries_whole(dataset):
    train_df, test_df = split_dataset(dataset, "expiry", 0.3)
    key = ["ticker", "expiry"]
    overlap = train_df[key].drop_duplicates().merge(test_df[key].drop_duplicates())
    assert overlap.empty and len(train_df) + len(test_df) == len(dataset)


def test_date_split_holds_out_newest_day(dataset):
    train_df, test_df = split_dataset(dataset, "date")
    assert set(test_df["ticker"]) == {"BBB"} and set(train_df["ticker"]) == {"AAA"}


def test_split_errors(dataset):
    with pytest.raises(ValueError, match="two different days"):
        split_dataset(dataset[dataset["ticker"] == "AAA"], "date")
    with pytest.raises(ValueError, match="Unknown split"):
        split_dataset(dataset, "weekday")


@pytest.mark.parametrize("kind", ["smile", "price"])
def test_train_evaluate_save_load(dataset, tmp_path, kind):
    epochs = []
    result = train(dataset, TrainConfig(kind=kind, epochs=40, patience=40, hidden_layers=(32, 32)),
                   on_epoch=lambda e, total, logs: epochs.append(e))
    assert epochs == list(range(1, 41))
    assert list(result.metrics.index) == [NN, "Black-Scholes (30d hist. vol)", BS_ATM]
    assert np.isfinite(result.predictions[NN]).all() and (result.predictions[NN] >= 0).all()

    path = result.pricer.save(tmp_path / "m")
    loaded = OptionPricer.load(path)
    assert loaded.kind == kind and loaded.metadata["rows"] == result.pricer.metadata["rows"]
    row = single_contract(S=100, K=105, T=0.25, r=0.04, q=0.01, hist_vol=0.2, is_call=True, atm_iv=0.22)
    assert loaded.predict(row) == pytest.approx(result.pricer.predict(row), rel=1e-5)


def test_training_can_be_stopped(dataset):
    result = train(dataset, TrainConfig(epochs=50, hidden_layers=(8,)), should_stop=lambda: True)
    assert result.pricer.metadata["epochs_run"] == 1
