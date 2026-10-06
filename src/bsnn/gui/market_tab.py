"""Price history, moving averages, realised volatility and implied-vol indices for a market."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from bsnn import instruments, market_data
from bsnn.gui.common import (ACCENT_COLOUR, CALL_COLOUR, NN_COLOUR, PUT_COLOUR, PlotCanvas, run_in_background,
                             status_label, ticker_box, ticker_of)

PERIODS = ["6mo", "1y", "2y", "5y"]
CHARTS = ["Price with 50/200-day moving averages", "Realised volatility (30 and 60 day)",
          "Distribution of daily moves"]


@dataclass
class MarketData:
    symbol: str
    ticker: str
    period: str
    history: pd.DataFrame
    is_yield: bool = False
    vol_index: str | None = None
    vol_index_history: pd.DataFrame | None = None


def load_market(symbol: str, period: str) -> MarketData:
    inst = instruments.resolve(symbol)
    ticker = inst.price_ticker if inst else symbol
    data = MarketData(symbol, ticker, period, market_data.get_history(ticker, refresh=True, period=period),
                      is_yield=bool(inst and inst.is_yield), vol_index=inst.vol_index if inst else None)
    if data.vol_index:
        try:
            data.vol_index_history = market_data.get_history(data.vol_index, refresh=True, period=period)
        except Exception:
            data.vol_index = None  # the chart still works without it
    return data


def daily_moves(data: MarketData) -> pd.Series:
    """Daily log returns in percent, or daily changes in basis points for a yield."""
    close = data.history["Close"]
    return close.diff().dropna() * 100 if data.is_yield else np.log(close).diff().dropna() * 100


def realised_vol(data: MarketData, window: int) -> pd.Series:
    """Annualised: percent for prices, basis points a year for yields (normal vol)."""
    close = data.history["Close"]
    if data.is_yield:
        return close.diff().rolling(window).std() * np.sqrt(market_data.TRADING_DAYS) * 100
    return market_data.realized_vol(close, window) * 100


class MarketTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cache: dict[tuple[str, str], MarketData] = {}
        self.current: MarketData | None = None

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
        top.addWidget(QLabel("Market"))
        top.addWidget(self.ticker, 1)
        top.addWidget(QLabel("Period"))
        top.addWidget(self.period)
        top.addWidget(self.load_button)
        top.addSpacing(24)
        top.addWidget(self.chart, 2)

        self.stats = {name: QLabel("–") for name in
                      ("Last", "1-day change", "Period change", "30d realised vol", "60d realised vol", "Implied vol index")}
        stats_box = QGroupBox("Summary")
        grid = QGridLayout(stats_box)
        for i, (name, value) in enumerate(self.stats.items()):
            grid.addWidget(QLabel(name), 0, i)
            value.setStyleSheet("font-size: 15px; font-weight: 600;")
            grid.addWidget(value, 1, i)

        self.plot = PlotCanvas()
        self.plot.message("Pick a market and press Load")
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.status)
        layout.addWidget(stats_box)
        layout.addWidget(self.plot, 1)

    def load(self):
        symbol, period = ticker_of(self.ticker), self.period.currentText()
        if not symbol:
            return
        self.load_button.setEnabled(False)
        self.status.setText(f"Downloading {symbol} ({period})…")
        run_in_background(self, load_market, symbol, period, on_done=self.on_loaded, on_error=self.on_failed)

    def on_loaded(self, data: MarketData):
        self.load_button.setEnabled(True)
        self.cache[(data.symbol, data.period)] = data
        self.current = data
        df = data.history
        source = f" ({data.ticker})" if data.ticker != data.symbol else ""
        self.status.setText(f"{data.symbol}{source}: {len(df)} trading days, "
                            f"{df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y}.")
        close = df["Close"]
        unit = "bp" if data.is_yield else "%"
        self.stats["Last"].setText(f"{close.iloc[-1]:,.3f}%" if data.is_yield else f"{close.iloc[-1]:,.5g}")
        if data.is_yield:
            self.stats["1-day change"].setText(f"{(close.iloc[-1] - close.iloc[-2]) * 100:+.1f} bp")
            self.stats["Period change"].setText(f"{(close.iloc[-1] - close.iloc[0]) * 100:+.0f} bp")
        else:
            self.stats["1-day change"].setText(f"{close.iloc[-1] / close.iloc[-2] - 1:+.2%}")
            self.stats["Period change"].setText(f"{close.iloc[-1] / close.iloc[0] - 1:+.1%}")
        self.stats["30d realised vol"].setText(f"{realised_vol(data, 30).iloc[-1]:.1f} {unit}")
        self.stats["60d realised vol"].setText(f"{realised_vol(data, 60).iloc[-1]:.1f} {unit}")
        if data.vol_index_history is not None:
            self.stats["Implied vol index"].setText(
                f"{data.vol_index_history['Close'].iloc[-1]:.1f} ({data.vol_index.lstrip('^')})")
        else:
            self.stats["Implied vol index"].setText("–")
        self.redraw()

    def on_failed(self, message: str):
        self.load_button.setEnabled(True)
        self.status.setText(f"Couldn't load {ticker_of(self.ticker)}: {message}")

    def redraw(self):
        data = self.current
        if data is None:
            return
        close = data.history["Close"]
        chart = self.chart.currentText()
        ax = self.plot.axes()
        if chart.startswith("Price"):
            ax.plot(close.index, close, color=CALL_COLOUR, lw=1.2, label="Yield" if data.is_yield else "Close")
            ax.plot(close.index, close.rolling(50).mean(), color=NN_COLOUR, lw=1, label="50-day MA")
            ax.plot(close.index, close.rolling(200).mean(), color=PUT_COLOUR, lw=1, label="200-day MA")
            ax.set_ylabel("Yield (%)" if data.is_yield else "Price")
        elif chart.startswith("Realised"):
            ax.plot(close.index, realised_vol(data, 30), color=CALL_COLOUR, label="30-day realised")
            ax.plot(close.index, realised_vol(data, 60), color=PUT_COLOUR, label="60-day realised")
            # MOVE is quoted in bp a year, like yield vol; the other indices are in percent like
            # price vol. An index is only drawn on the chart when its units match.
            if data.vol_index_history is not None and (data.vol_index == "^MOVE") == data.is_yield:
                vix = data.vol_index_history["Close"]
                ax.plot(vix.index, vix, color=ACCENT_COLOUR, lw=1, label=f"{data.vol_index.lstrip('^')} (implied)")
            ax.set_ylabel("Annualised volatility (bp)" if data.is_yield else "Annualised volatility (%)")
        else:
            moves = daily_moves(data)
            ax.hist(moves, bins=60, color=CALL_COLOUR, alpha=0.75, density=True,
                    label="Daily changes" if data.is_yield else "Daily log returns")
            x = np.linspace(moves.min(), moves.max(), 200)
            sd = moves.std()
            normal = np.exp(-0.5 * ((x - moves.mean()) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            ax.plot(x, normal, color=ACCENT_COLOUR, label="Normal with same σ")
            ax.set_xlabel("Daily change (bp)" if data.is_yield else "Daily return (%)")
            ax.set_ylabel("Density")
        ax.set_title(f"{data.symbol} · {chart} · {data.period}")
        self.plot.legend(ax)
        self.plot.draw()
