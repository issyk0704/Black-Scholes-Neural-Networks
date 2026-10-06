import pandas as pd
import pytest

pytest.importorskip("pytestqt")

from bsnn.gui import model_tab  # noqa: E402
from bsnn.gui.app import MainWindow  # noqa: E402

from conftest import make_chain  # noqa: E402


class FakePricer:
    """Stands in for a trained model: prices every contract at its mid."""

    kind, uses_atm_iv, description, metadata = "smile", True, "Fake model", {}

    def predict(self, df):
        return df["mid"].to_numpy() if "mid" in df else df["S"].to_numpy() * 0 + 1.23


@pytest.fixture
def window(qtbot):
    w = MainWindow(load_latest_model=False)
    qtbot.addWidget(w)
    return w


def cell(table, row_label, column=0):
    for r in range(table.rowCount()):
        if table.verticalHeaderItem(r).text() == row_label:
            return table.item(r, column).text()
    raise KeyError(row_label)


def set_contract(tab, S=100, K=100, days=365, r=5, q=0, vol=20):
    for box, value in ((tab.spot, S), (tab.strike, K), (tab.days, days), (tab.rate, r), (tab.dividend, q),
                       (tab.vol, vol)):
        box.setValue(value)


def test_pricer_recalculates_as_inputs_change(window):
    tab = window.pricer_tab
    set_contract(tab, days=365)
    # 365 days = 1 year: the textbook S=K=100, r=5%, sigma=20% values.
    assert cell(tab.table, "Price", 0) == "10.4506"
    assert cell(tab.table, "Price", 1) == "5.5735"
    tab.vol.setValue(30)
    assert cell(tab.table, "Price", 0) != "10.4506"


def test_implied_vol_solver_round_trips(window):
    tab = window.pricer_tab
    set_contract(tab)
    tab.market_price.setValue(10.4506)
    assert tab.iv_result.text() == "20.00%"
    tab.vol.setValue(35)
    tab.use_implied_vol()
    assert tab.vol.value() == pytest.approx(20.0, abs=0.01)


def test_pricer_shows_neural_network_price(window):
    window.state.set_pricer(FakePricer())
    assert cell(window.pricer_tab.table, "Neural-network price", 0) == "1.2300"
    assert "Fake model" in window.statusBar().currentMessage()


def test_chain_tab_lists_contracts_and_model_prices(window):
    tab = window.chain_tab
    tab.set_chain(make_chain(), "test")
    assert tab.expiry.count() == 4
    assert tab.table.rowCount() > 0
    window.state.set_pricer(FakePricer())
    headers = [tab.table.horizontalHeaderItem(c).text() for c in range(tab.table.columnCount())]
    assert headers[-2:] == ["Neural net", "NN − mid"]
    tab.side.setCurrentText("Puts")
    assert {tab.table.item(r, 1).text() for r in range(tab.table.rowCount())} == {"Put"}


def test_market_tab_summarises_history(window, history):
    tab = window.market_tab
    tab.on_loaded(("TEST", "1y"), history)
    assert tab.stats["Last close"].text() == f"{history['Close'].iloc[-1]:,.2f}"
    for i in range(tab.chart.count()):
        tab.chart.setCurrentIndex(i)


def test_model_tab_trains_in_background(window, qtbot, tmp_path, monkeypatch):
    expiries = ("2026-01-16", "2026-02-20", "2026-03-20", "2026-04-17", "2026-06-18", "2026-09-18")
    files = []
    for ticker in ("AAA", "BBB"):
        path = tmp_path / f"{ticker}_options_2026-01-05.csv"
        make_chain(ticker=ticker, expiries=expiries).to_csv(path, index=False)
        files.append(path)
    monkeypatch.setattr(model_tab.market_data, "list_option_snapshots", lambda: files)

    tab = window.model_tab
    tab.refresh_snapshots()
    assert tab.selected_files() == files
    tab.epochs.setValue(3)
    tab.layers.setText("8, 8")
    tab.start_training()
    qtbot.waitUntil(lambda: tab.result is not None, timeout=120_000)

    assert tab.metrics.rowCount() == 3
    assert window.state.pricer is tab.result.pricer
    assert cell(window.pricer_tab.table, "Neural-network price", 0)
    for i in range(tab.chart.count()):
        tab.chart.setCurrentIndex(i)
    assert isinstance(tab.result.predictions, pd.DataFrame)
