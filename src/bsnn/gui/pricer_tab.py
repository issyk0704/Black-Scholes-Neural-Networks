"""Black-Scholes pricer: price, Greeks and implied vol for one contract, with charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QSplitter, QTableWidget, QVBoxLayout, QWidget)

from bsnn import market_data, pricing
from bsnn.features import single_contract
from bsnn.gui.common import (CALL_COLOUR, NN_COLOUR, PUT_COLOUR, AppState, PlotCanvas, fill_table,
                             run_in_background, spin, status_label, ticker_box, ticker_of)

GREEK_ROWS = [("price", "Price", "{:.4f}"), ("delta", "Delta", "{:.4f}"), ("gamma", "Gamma", "{:.5f}"),
              ("vega", "Vega (per 1 vol pt)", "{:.4f}"), ("theta", "Theta (per day)", "{:.4f}"),
              ("rho", "Rho (per 1% rate)", "{:.4f}")]
CHARTS = ["Price vs spot", "Delta vs spot", "Gamma vs spot", "Vega vs spot", "Theta vs spot", "Price vs volatility"]


def load_market_inputs(ticker: str) -> dict:
    history = market_data.get_history(ticker, refresh=True)
    return {
        "ticker": ticker,
        "spot": float(history["Close"].iloc[-1]),
        "asof": history.index[-1],
        "hist_vol": float(market_data.realized_vol(history["Close"]).iloc[-1]),
        "dividend_yield": market_data.dividend_yield(history),
        "rate": market_data.risk_free_rate(),
    }


class PricerTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.hist_vol: float | None = None  # from the last loaded ticker
        self._iv = np.nan

        # Underlying
        self.ticker = ticker_box()
        self.ticker.currentTextChanged.connect(self.forget_loaded_ticker)
        self.load_button = QPushButton("Load market data")
        self.load_button.clicked.connect(self.load_ticker)
        self.status = status_label()
        self.status.setText("Load a ticker to fill in spot, rate, dividend yield and volatility, or type your own.")
        ticker_row = QHBoxLayout()
        ticker_row.addWidget(self.ticker, 1)
        ticker_row.addWidget(self.load_button)
        underlying = QGroupBox("Underlying")
        u_layout = QVBoxLayout(underlying)
        u_layout.addLayout(ticker_row)
        u_layout.addWidget(self.status)

        # Contract inputs
        self.spot = spin(100, 0.01, 1e6, 2, step=1)
        self.strike = spin(100, 0.01, 1e6, 2, step=1)
        self.days = spin(30, 0.1, 3650, 1, " days", step=1)
        self.rate = spin(4.0, -5, 50, 3, " %", step=0.1)
        self.dividend = spin(0.0, 0, 50, 3, " %", step=0.1)
        self.vol = spin(20.0, 0.1, 500, 2, " %", step=0.5)
        self.vol.setToolTip("Used by Black-Scholes. A neural-network model that needs ATM implied vol uses this too.")
        inputs = QGroupBox("Contract")
        form = QFormLayout(inputs)
        for label, box in (("Spot (S)", self.spot), ("Strike (K)", self.strike), ("Time to expiry", self.days),
                           ("Risk-free rate (r)", self.rate), ("Dividend yield (q)", self.dividend),
                           ("Volatility (σ)", self.vol)):
            form.addRow(label, box)
            box.valueChanged.connect(self.recalculate)

        # Results
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(240)
        self.nn_note = status_label()

        # Implied vol solver
        self.market_price = spin(2.5, 0.0, 1e6, 4, step=0.05)
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
        self.recalculate()

    # --- Inputs ---------------------------------------------------------------

    def inputs(self) -> dict:
        return {"S": self.spot.value(), "K": self.strike.value(), "T": self.days.value() / 365,
                "r": self.rate.value() / 100, "sigma": self.vol.value() / 100, "q": self.dividend.value() / 100}

    def load_ticker(self):
        ticker = ticker_of(self.ticker)
        if not ticker:
            return
        self.load_button.setEnabled(False)
        self.status.setText(f"Loading {ticker}…")
        run_in_background(self, load_market_inputs, ticker, on_done=self.on_loaded, on_error=self.on_load_failed)

    def on_loaded(self, data: dict):
        self.load_button.setEnabled(True)
        self.hist_vol = data["hist_vol"]
        boxes = [self.spot, self.strike, self.rate, self.dividend, self.vol]
        for box in boxes:
            box.blockSignals(True)
        self.spot.setValue(data["spot"])
        self.strike.setValue(round(data["spot"]))
        self.rate.setValue(data["rate"] * 100)
        self.dividend.setValue(data["dividend_yield"] * 100)
        self.vol.setValue(data["hist_vol"] * 100)
        for box in boxes:
            box.blockSignals(False)
        self.status.setText(
            f"{data['ticker']}: close {data['spot']:,.2f} on {data['asof']:%d %b %Y}. "
            f"σ set to 30-day historical vol ({data['hist_vol']:.1%}); r is the 13-week T-bill yield.")
        # Start the implied-vol solver from this contract's own price, so it reads back σ.
        p = self.inputs()
        self.market_price.blockSignals(True)
        self.market_price.setValue(round(pricing.price(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"]), 2))
        self.iv_type.setCurrentText("Call")
        self.market_price.blockSignals(False)
        self.recalculate()

    def forget_loaded_ticker(self):
        if self.hist_vol is not None:
            self.hist_vol = None
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
        rows = pd.concat([single_contract(p["S"], p["K"], p["T"], p["r"], p["q"], hist_vol, is_call, p["sigma"])
                          for is_call in (True, False)], ignore_index=True)
        call, put = pricer.predict(rows)
        return float(call), float(put)

    def recalculate(self):
        p = self.inputs()
        call = pricing.greeks(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"], True)
        put = pricing.greeks(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"], False)
        rows = [[fmt.format(call[key]), fmt.format(put[key])] for key, _, fmt in GREEK_ROWS]
        labels = [label for _, label, _ in GREEK_ROWS]
        nn = self.nn_prices(p)
        if nn:
            rows.append([f"{nn[0]:.4f}", f"{nn[1]:.4f}"])
            labels.append("Neural-network price")
            vol_used = "σ as the ATM implied vol" if self.state.pricer.uses_atm_iv else (
                "the loaded 30-day historical vol" if self.hist_vol else "σ as the historical vol")
            self.nn_note.setText(f"Neural network: {self.state.pricer.description}; this contract uses {vol_used}.")
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
                values = pricing.greeks(x, p["K"], p["T"], p["r"], p["sigma"], p["q"], is_call)[key]
                ax.plot(x, values, color=colour, label=f"{name} today")
                if key == "price":
                    ax.plot(x, pricing.payoff(x, p["K"], is_call), color=colour, ls=":", lw=1,
                            label=f"{name} at expiry")
            ax.axvline(p["S"], color=NN_COLOUR, ls="--", lw=1, label="Spot")
            ax.set_xlabel("Spot price")
            ax.set_ylabel(chart.replace(" vs spot", ""))
        ax.set_title(f"{chart}  (K={p['K']:g}, {p['T'] * 365:.0f} days, σ={p['sigma']:.1%})")
        self.plot.legend(ax)
        self.plot.draw()
