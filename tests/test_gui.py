import pandas as pd
import pytest

pytest.importorskip("pytestqt")

from bsnn import pricing  # noqa: E402
from bsnn.gui import model_tab  # noqa: E402
from bsnn.gui.app import MainWindow  # noqa: E402
from bsnn.gui.common import ticker_of  # noqa: E402
from bsnn.gui.levels_tab import load_levels  # noqa: E402
from bsnn.gui.market_tab import MarketData  # noqa: E402

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
    for box, value in ((tab.spot, S), (tab.strike, K), (tab.days, days), (tab.rate, r), (tab.carry, q),
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


def test_pricer_switches_to_black76_for_futures(window):
    tab = window.pricer_tab
    set_contract(tab, S=31_000, K=31_000, days=30, r=4, q=1, vol=20)
    tab.model.setCurrentIndex(tab.model.findData("black76"))
    assert tab.form.labelForField(tab.spot).text() == "Futures price (F)"
    assert not tab.carry.isEnabled()
    expected = pricing.model_greeks("black76", 31_000, 31_000, 30 / 365, 0.04, 0.2, is_call=True)["price"]
    assert cell(tab.table, "Price", 0) == f"{expected:.4f}"


def test_pricer_loads_fx_with_garman_kohlhagen(window):
    tab = window.pricer_tab
    tab.on_loaded({"symbol": "EU", "ticker": "EURUSD=X", "model": "garman_kohlhagen", "spot": 1.12413,
                   "asof": pd.Timestamp("2026-10-05"), "hist_vol": 0.07, "rate": 0.04, "carry": 0.026,
                   "asset_class": "FX", "note": ""})
    assert tab.current_model() == "garman_kohlhagen"
    assert tab.spot.decimals() == 5 and tab.spot.value() == pytest.approx(1.12413)
    assert tab.form.labelForField(tab.carry).text() == "Foreign rate (r_f)"
    assert tab.iv_result.text() == "7.00%"  # the solver starts from this contract's own price


def test_market_tab_summarises_prices(window, history):
    tab = window.market_tab
    tab.on_loaded(MarketData("TEST", "TEST", "1y", history))
    assert tab.stats["Last"].text() == f"{history['Close'].iloc[-1]:,.5g}"
    for i in range(tab.chart.count()):
        tab.chart.setCurrentIndex(i)


def test_market_tab_measures_yields_in_basis_points(window, history):
    yields = history.assign(Close=4.0 + history["Close"] / 1000)
    tab = window.market_tab
    tab.on_loaded(MarketData("US10Y", "^TNX", "1y", yields, is_yield=True, vol_index="^MOVE",
                             vol_index_history=history.assign(Close=110.0)))
    assert tab.stats["1-day change"].text().endswith(" bp")
    assert tab.stats["30d realised vol"].text().endswith(" bp")
    assert tab.stats["Implied vol index"].text() == "110.0 (MOVE)"
    for i in range(tab.chart.count()):
        tab.chart.setCurrentIndex(i)


def test_ticker_box_reads_watchlist_labels(window):
    box = window.market_tab.ticker
    box.setCurrentText("NQ — Nasdaq-100 E-mini")
    assert ticker_of(box) == "NQ"
    box.setCurrentText(" aapl ")
    assert ticker_of(box) == "AAPL"
    assert all("US10Y" not in window.pricer_tab.ticker.itemText(i) for i in range(window.pricer_tab.ticker.count()))


def test_levels_tab_offers_index_and_etf_options(window):
    tab = window.levels_tab
    tab.ticker.setCurrentText("NQ — Nasdaq-100 E-mini")
    assert [tab.source.itemData(i) for i in range(tab.source.count())] == ["^NDX", "QQQ"]
    tab.ticker.setCurrentText("AAPL")
    assert [tab.source.itemData(i) for i in range(tab.source.count())] == ["AAPL"]


def test_levels_tab_shows_moves_and_gamma(window, tmp_path):
    path = tmp_path / "TEST_options_2026-01-05.csv"
    make_chain().to_csv(path, index=False)
    data = load_levels(None, path)
    assert data.futures is None and not data.notes
    tab = window.levels_tab
    tab.on_loaded(data)
    assert tab.table.rowCount() == 4
    assert tab.summary["Spot"][0].text() == "100.00"
    assert tab.summary["1-day implied move"][0].text().startswith("±")
    assert tab.summary["Net gamma (per 1%)"][0].text() != "–"
    for i in range(tab.chart.count()):
        tab.chart.setCurrentIndex(i)
    tab.gamma_window.setCurrentText("Next 7 days")  # no expiries that close: gamma clears instead of failing
    assert tab.summary["Net gamma (per 1%)"][0].text() == "–"


def test_levels_tab_explains_missing_open_interest(window, tmp_path):
    path = tmp_path / "TEST_options_2026-01-05.csv"
    make_chain().assign(openInterest=0.0).to_csv(path, index=False)
    data = load_levels(None, path)
    assert any("open interest" in note for note in data.notes)
    window.levels_tab.on_loaded(data)
    assert window.levels_tab.summary["Call wall"][0].text() == "–"


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
