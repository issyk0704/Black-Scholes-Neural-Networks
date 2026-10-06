"""Price history, moving averages and realised volatility for a ticker."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from bsnn import market_data
from bsnn.gui.common import (ACCENT_COLOUR, CALL_COLOUR, NN_COLOUR, PUT_COLOUR, PlotCanvas, run_in_background,
                             status_label, ticker_box, ticker_of)

PERIODS = ["6mo", "1y", "2y", "5y"]
CHARTS = ["Price with 50/200-day moving averages", "Realised volatility (30 and 60 day)",
          "Distribution of daily returns"]


class MarketTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cache: dict[tuple[str, str], pd.DataFrame] = {}
        self.current: tuple[str, str] | None = None

        self.ticker = ticker_box()
        self.period = QComboBox()
        self.period.addItems(PERIODS)
        self.period.setCurrentText("2y")
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load)
        self.chart = QComboBox()
        self.chart.addItems(CHARTS)
        self.chart.currentIndexChanged.connect(self.redraw)
        self.status = status_label()

        top = QHBoxLayout()
        top.addWidget(QLabel("Ticker"))
        top.addWidget(self.ticker, 1)
        top.addWidget(QLabel("Period"))
        top.addWidget(self.period)
        top.addWidget(self.load_button)
        top.addSpacing(24)
        top.addWidget(self.chart, 2)

        self.stats = {name: QLabel("–") for name in
                      ("Last close", "1-day change", "Period return", "30d realised vol", "60d realised vol",
                       "Dividend yield (12m)")}
        stats_box = QGroupBox("Summary")
        grid = QGridLayout(stats_box)
        for i, (name, label) in enumerate(self.stats.items()):
            grid.addWidget(QLabel(name), 0, i)
            label.setStyleSheet("font-size: 15px; font-weight: 600;")
            grid.addWidget(label, 1, i)

        self.plot = PlotCanvas()
        self.plot.message("Pick a ticker and press Load")
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.status)
        layout.addWidget(stats_box)
        layout.addWidget(self.plot, 1)

    def load(self):
        key = (ticker_of(self.ticker), self.period.currentText())
        if not key[0]:
            return
        self.load_button.setEnabled(False)
        self.status.setText(f"Downloading {key[0]} ({key[1]})…")
        run_in_background(self, market_data.get_history, key[0], refresh=True, period=key[1],
                          on_done=lambda df: self.on_loaded(key, df), on_error=self.on_failed)

    def on_loaded(self, key, df: pd.DataFrame):
        self.load_button.setEnabled(True)
        self.cache[key] = df
        self.current = key
        self.status.setText(f"{key[0]}: {len(df)} trading days, {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y}.")
        close = df["Close"]
        vol = market_data.realized_vol(close)
        self.stats["Last close"].setText(f"{close.iloc[-1]:,.2f}")
        self.stats["1-day change"].setText(f"{close.iloc[-1] / close.iloc[-2] - 1:+.2%}" if len(close) > 1 else "–")
        self.stats["Period return"].setText(f"{close.iloc[-1] / close.iloc[0] - 1:+.1%}")
        self.stats["30d realised vol"].setText(f"{vol.iloc[-1]:.1%}")
        self.stats["60d realised vol"].setText(f"{market_data.realized_vol(close, 60).iloc[-1]:.1%}")
        self.stats["Dividend yield (12m)"].setText(f"{market_data.dividend_yield(df):.2%}")
        self.redraw()

    def on_failed(self, message: str):
        self.load_button.setEnabled(True)
        self.status.setText(f"Couldn't load {ticker_of(self.ticker)}: {message}")

    def redraw(self):
        if self.current is None:
            return
        ticker, period = self.current
        df = self.cache[self.current]
        close = df["Close"]
        chart = self.chart.currentText()
        ax = self.plot.axes()
        if chart.startswith("Price"):
            ax.plot(close.index, close, color=CALL_COLOUR, lw=1.2, label="Close")
            ax.plot(close.index, close.rolling(50).mean(), color=NN_COLOUR, lw=1, label="50-day MA")
            ax.plot(close.index, close.rolling(200).mean(), color=PUT_COLOUR, lw=1, label="200-day MA")
            ax.set_ylabel("Price")
        elif chart.startswith("Realised"):
            ax.plot(close.index, market_data.realized_vol(close, 30) * 100, color=CALL_COLOUR, label="30-day")
            ax.plot(close.index, market_data.realized_vol(close, 60) * 100, color=PUT_COLOUR, label="60-day")
            ax.set_ylabel("Annualised volatility (%)")
        else:
            returns = np.log(close).diff().dropna() * 100
            ax.hist(returns, bins=60, color=CALL_COLOUR, alpha=0.75, density=True, label="Daily log returns")
            x = np.linspace(returns.min(), returns.max(), 200)
            sd = returns.std()
            normal = np.exp(-0.5 * ((x - returns.mean()) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            ax.plot(x, normal, color=ACCENT_COLOUR, label="Normal with same σ")
            ax.set_xlabel("Daily return (%)")
            ax.set_ylabel("Density")
        ax.set_title(f"{ticker} · {chart} · {period}")
        self.plot.legend(ax)
        self.plot.draw()
