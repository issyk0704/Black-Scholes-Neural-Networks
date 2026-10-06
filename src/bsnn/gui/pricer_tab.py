"""Option pricer for stocks, futures and FX: price, Greeks and implied vol for one contract, with charts."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QSplitter, QTableWidget, QVBoxLayout, QWidget)

from bsnn import instruments, market_data, pricing
from bsnn.features import single_contract
from bsnn.gui.common import (CALL_COLOUR, NN_COLOUR, PUT_COLOUR, AppState, PlotCanvas, fill_table,
                             run_in_background, spin, status_label, ticker_box, ticker_of)

log = logging.getLogger(__name__)

GREEK_ROWS = [("price", "Price", "{:.4f}"), ("delta", "Delta", "{:.4f}"), ("gamma", "Gamma", "{:.5f}"),
              ("vega", "Vega (per 1 vol pt)", "{:.4f}"), ("theta", "Theta (per day)", "{:.4f}"),
              ("rho", "Rho (per 1% rate)", "{:.4f}")]
CHARTS = ["Price vs spot", "Delta vs spot", "Gamma vs spot", "Vega vs spot", "Theta vs spot", "Price vs volatility"]
# What the spot and carry inputs mean under each model.
SPOT_LABEL = {"bsm": "Spot (S)", "black76": "Futures price (F)", "garman_kohlhagen": "Spot rate (S)"}
CARRY_LABEL = {"bsm": "Dividend yield (q)", "black76": "Carry", "garman_kohlhagen": "Foreign rate (r_f)"}


def load_market_inputs(symbol: str) -> dict:
    """Spot, rate, carry and historical vol for a watchlist market or any Yahoo ticker."""
    inst = instruments.resolve(symbol)
    if inst is not None and inst.model is None:
        raise ValueError(f"{symbol} is a yield, not an option underlying. Use ZT, ZN or ZB for Treasury options.")
    ticker = inst.price_ticker if inst else symbol
    model = inst.model if inst else "bsm"
    history = market_data.get_history(ticker, refresh=True)
    spot = float(history["Close"].iloc[-1])
    rate = market_data.risk_free_rate()
    carry, carry_note = 0.0, ""
    if model == "bsm":
        carry = market_data.dividend_yield(history)
    elif model == "garman_kohlhagen":
        try:
            carry = market_data.implied_foreign_rate(spot, inst.futures_ticker, rate)
            carry_note = f" Foreign rate implied from {inst.futures_ticker} futures."
        except Exception:
            log.warning("Couldn't infer the foreign rate for %s", symbol, exc_info=True)
            carry_note = " Couldn't infer the foreign rate; enter it by hand."
    return {
        "symbol": symbol, "ticker": ticker, "model": model, "spot": spot, "asof": history.index[-1],
        "hist_vol": float(market_data.realized_vol(history["Close"]).iloc[-1]), "rate": rate, "carry": carry,
        "asset_class": inst.asset_class if inst else instruments.SINGLE_STOCK,
        "note": (f" {inst.note}" if inst and inst.note else "") + carry_note,
    }


class PricerTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.hist_vol: float | None = None  # from the last loaded ticker
        self.asset_class = instruments.SINGLE_STOCK
        self._iv = np.nan

        # Underlying
        self.ticker = ticker_box(include_yields=False)
        self.ticker.currentTextChanged.connect(self.forget_loaded_ticker)
        self.load_button = QPushButton("Load market data")
        self.load_button.clicked.connect(self.load_ticker)
        self.status = status_label()
        self.status.setText("Load a market to fill in price, rate, carry and volatility, or type your own.")
        ticker_row = QHBoxLayout()
        ticker_row.addWidget(self.ticker, 1)
        ticker_row.addWidget(self.load_button)
        self.model = QComboBox()
        for key, name in pricing.MODELS.items():
            self.model.addItem(name, key)
        self.model.currentIndexChanged.connect(self.on_model_changed)
        underlying = QGroupBox("Underlying")
        u_layout = QVBoxLayout(underlying)
        u_layout.addLayout(ticker_row)
        u_layout.addWidget(self.model)
        u_layout.addWidget(self.status)

        # Contract inputs
        self.spot = spin(100, 0.0001, 1e7, 2, step=1)
        self.strike = spin(100, 0.0001, 1e7, 2, step=1)
        self.days = spin(30, 0.1, 3650, 1, " days", step=1)
        self.rate = spin(4.0, -5, 50, 3, " %", step=0.1)
        self.carry = spin(0.0, -5, 50, 3, " %", step=0.1)
        self.vol = spin(20.0, 0.1, 500, 2, " %", step=0.5)
        self.vol.setToolTip("Used by Black-Scholes. A neural-network model that needs ATM implied vol uses this too.")
        inputs = QGroupBox("Contract")
        self.form = QFormLayout(inputs)
        for text, box in (("Spot (S)", self.spot), ("Strike (K)", self.strike), ("Time to expiry", self.days),
                          ("Risk-free rate (r)", self.rate), ("Dividend yield (q)", self.carry),
                          ("Volatility (σ)", self.vol)):
            self.form.addRow(text, box)
            box.valueChanged.connect(self.recalculate)

        # Results
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(240)
        self.nn_note = status_label()

        # Implied vol solver
        self.market_price = spin(2.5, 0.0, 1e7, 4, step=0.05)
        self.iv_type = QComboBox()
        self.iv_type.addItems(["Call", "Put"])
        self.iv_result = QLabel("–")
        use_iv = QPushButton("Use as σ")
        use_iv.clicked.connect(self.use_implied_vol)
        self.market_price.valueChanged.connect(self.solve_iv)
        self.iv_type.currentIndexChanged.connect(self.solve_iv)
        iv_box = QGroupBox("Implied volatility from a market price")
        iv_form = QFormLayout(iv_box)
        iv_form.addRow("Market price", self.market_price)
        iv_form.addRow("Option type", self.iv_type)
        iv_row = QHBoxLayout()
        iv_row.addWidget(self.iv_result, 1)
        iv_row.addWidget(use_iv)
        iv_form.addRow("Implied vol", iv_row)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(underlying)
        left_layout.addWidget(inputs)
        left_layout.addWidget(self.table)
        left_layout.addWidget(self.nn_note)
        left_layout.addWidget(iv_box)
        left_layout.addStretch()

        # Chart
        self.chart = QComboBox()
        self.chart.addItems(CHARTS)
        self.chart.currentIndexChanged.connect(self.redraw)
        self.plot = PlotCanvas()
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.chart)
        right_layout.addWidget(self.plot)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 700])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        state.pricer_changed.connect(lambda _: self.recalculate())
        self.on_model_changed()

    # --- Inputs ---------------------------------------------------------------

    def current_model(self) -> str:
        return self.model.currentData()

    def inputs(self) -> dict:
        model, r, carry = self.current_model(), self.rate.value() / 100, self.carry.value() / 100
        return {"model": model, "S": self.spot.value(), "K": self.strike.value(), "T": self.days.value() / 365,
                "r": r, "sigma": self.vol.value() / 100, "carry": carry, "q": pricing.carry_yield(model, r, carry)}

    def on_model_changed(self):
        model = self.current_model()
        self.form.labelForField(self.spot).setText(SPOT_LABEL[model])
        self.form.labelForField(self.carry).setText(CARRY_LABEL[model])
        self.carry.setEnabled(model != "black76")
        self.carry.setToolTip("Futures carry no dividend or foreign rate" if model == "black76" else "")
        self.recalculate()

    def set_precision(self, price: float):
        """FX rates need more decimals than index futures."""
        decimals = 5 if price < 10 else 2
        for box in (self.spot, self.strike):
            box.setDecimals(decimals)
            box.setSingleStep(10 ** -(decimals - 1) if decimals > 2 else 1)

    def load_ticker(self):
        symbol = ticker_of(self.ticker)
        if not symbol:
            return
        self.load_button.setEnabled(False)
        self.status.setText(f"Loading {symbol}…")
        run_in_background(self, load_market_inputs, symbol, on_done=self.on_loaded, on_error=self.on_load_failed)

    def on_loaded(self, data: dict):
        self.load_button.setEnabled(True)
        self.hist_vol = data["hist_vol"]
        self.asset_class = data["asset_class"]
        boxes = [self.model, self.spot, self.strike, self.rate, self.carry, self.vol]
        for box in boxes:
            box.blockSignals(True)
        self.model.setCurrentIndex(self.model.findData(data["model"]))
        self.set_precision(data["spot"])
        self.spot.setValue(data["spot"])
        self.strike.setValue(round(data["spot"], 3 if data["spot"] < 10 else 0))
        self.rate.setValue(data["rate"] * 100)
        self.carry.setValue(data["carry"] * 100)
        self.vol.setValue(data["hist_vol"] * 100)
        for box in boxes:
            box.blockSignals(False)
        source = f" ({data['ticker']})" if data["ticker"] != data["symbol"] else ""
        self.status.setText(
            f"{data['symbol']}{source}: close {data['spot']:,.5g} on {data['asof']:%d %b %Y}. "
            f"σ set to 30-day historical vol ({data['hist_vol']:.1%}); r is the 13-week T-bill yield.{data['note']}")
        # Start the implied-vol solver from this contract's own price, so it reads back σ.
        p = self.inputs()
        self.market_price.blockSignals(True)
        self.market_price.setValue(pricing.price(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"]))
        self.iv_type.setCurrentText("Call")
        self.market_price.blockSignals(False)
        self.on_model_changed()

    def forget_loaded_ticker(self):
        if self.hist_vol is not None:
            self.hist_vol = None
            self.asset_class = instruments.SINGLE_STOCK
            self.recalculate()

    def on_load_failed(self, message: str):
        self.load_button.setEnabled(True)
        self.status.setText(f"Couldn't load {ticker_of(self.ticker)}: {message}")

    # --- Results ---------------------------------------------------------------

    def nn_prices(self, p: dict) -> tuple[float, float] | None:
        pricer = self.state.pricer
        if pricer is None:
            return None
        hist_vol = self.hist_vol if self.hist_vol else p["sigma"]
        rows = pd.concat([single_contract(p["S"], p["K"], p["T"], p["r"], p["q"], hist_vol, is_call, p["sigma"],
                                          self.asset_class) for is_call in (True, False)], ignore_index=True)
        call, put = pricer.predict(rows)
        return float(call), float(put)

    def recalculate(self):
        p = self.inputs()
        call = pricing.model_greeks(p["model"], p["S"], p["K"], p["T"], p["r"], p["sigma"], p["carry"], True)
        put = pricing.model_greeks(p["model"], p["S"], p["K"], p["T"], p["r"], p["sigma"], p["carry"], False)
        rows = [[fmt.format(call[key]), fmt.format(put[key])] for key, _, fmt in GREEK_ROWS]
        labels = [text for _, text, _ in GREEK_ROWS]
        nn = self.nn_prices(p)
        if nn:
            rows.append([f"{nn[0]:.4f}", f"{nn[1]:.4f}"])
            labels.append("Neural-network price")
            vol_used = "σ as the ATM implied vol" if self.state.pricer.uses_atm_iv else (
                "the loaded 30-day historical vol" if self.hist_vol else "σ as the historical vol")
            self.nn_note.setText(f"Neural network: {self.state.pricer.description}; this contract uses {vol_used}"
                                 f" and is treated as {self.asset_class.lower()}.")
        else:
            self.nn_note.setText("Train or load a model in the Neural network tab to see its price here.")
        fill_table(self.table, ["Call", "Put"], rows, labels)
        self.solve_iv()
        self.redraw()

    def solve_iv(self):
        p = self.inputs()
        iv = pricing.implied_volatility(self.market_price.value(), p["S"], p["K"], p["T"], p["r"], p["q"],
                                        self.iv_type.currentText() == "Call")
        self.iv_result.setText("No volatility fits this price" if np.isnan(iv) else f"{iv:.2%}")
        self._iv = iv

    def use_implied_vol(self):
        if not np.isnan(self._iv):
            self.vol.setValue(self._iv * 100)

    def redraw(self):
        p = self.inputs()
        chart = self.chart.currentText()
        ax = self.plot.axes()
        if chart == "Price vs volatility":
            x = np.linspace(0.02, max(1.0, p["sigma"] * 2.5), 200)
            for is_call, name, colour in ((True, "Call", CALL_COLOUR), (False, "Put", PUT_COLOUR)):
                ax.plot(x * 100, pricing.price(p["S"], p["K"], p["T"], p["r"], x, p["q"], is_call),
                        color=colour, label=name)
            ax.axvline(p["sigma"] * 100, color=NN_COLOUR, ls="--", lw=1, label="Current σ")
            ax.set_xlabel("Volatility (%)")
            ax.set_ylabel("Option price")
        else:
            x = np.linspace(p["K"] * 0.6, p["K"] * 1.4, 300)
            key = chart.split()[0].lower()
            for is_call, name, colour in ((True, "Call", CALL_COLOUR), (False, "Put", PUT_COLOUR)):
                values = pricing.model_greeks(p["model"], x, p["K"], p["T"], p["r"], p["sigma"], p["carry"],
                                              is_call)[key]
                ax.plot(x, values, color=colour, label=f"{name} today")
                if key == "price":
                    ax.plot(x, pricing.payoff(x, p["K"], is_call), color=colour, ls=":", lw=1,
                            label=f"{name} at expiry")
            ax.axvline(p["S"], color=NN_COLOUR, ls="--", lw=1, label="Spot")
            ax.set_xlabel(SPOT_LABEL[p["model"]].split(" (")[0])
            ax.set_ylabel(chart.replace(" vs spot", ""))
        ax.set_title(f"{chart}  (K={p['K']:g}, {p['T'] * 365:.0f} days, σ={p['sigma']:.1%})")
        self.plot.legend(ax)
        self.plot.draw()
