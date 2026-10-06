"""Implied moves and gamma levels for an index, ETF or stock, converted to futures points."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                             QSplitter, QTableWidget, QVBoxLayout, QWidget)

from bsnn import analytics, instruments, market_data
from bsnn.features import Filters, build_dataset
from bsnn.gui.common import (ACCENT_COLOUR, CALL_COLOUR, NN_COLOUR, PUT_COLOUR, PlotCanvas, fill_table,
                             run_in_background, status_label, ticker_box, ticker_of)

# Every strike with a usable quote; gamma far from spot is tiny, so a wide range costs nothing.
FILTERS = Filters(min_days=0.05, min_moneyness=0.5, max_moneyness=2.0, max_relative_spread=1.0, min_price=0.01)
GAMMA_WINDOWS = {"All expiries": None, "Next 30 days": 30, "Next 7 days": 7}
CHARTS = ["Gamma exposure by strike", "Net gamma vs price (gamma flip)", "Implied move by expiry"]
SUMMARY = ["Spot", "1-day implied move", "Net gamma (per 1%)", "Gamma flip", "Call wall", "Put wall"]


@dataclass
class LevelsData:
    ticker: str
    snapshot_time: str
    dataset: pd.DataFrame
    moves: pd.DataFrame
    open_interest: float
    futures: instruments.Instrument | None = None
    futures_ratio: float = np.nan  # futures price / underlying price on the snapshot day
    notes: list[str] = field(default_factory=list)


def futures_ratio(inst: instruments.Instrument, snapshot_time, spot: float) -> float:
    history = market_data.get_history(inst.price_ticker, refresh=True)
    return market_data.unadjusted_close(history, snapshot_time) / spot


def load_levels(ticker: str, path: Path | None = None, progress=None) -> LevelsData:
    """Fetch a live chain (or open a saved one) and work out its implied moves."""
    if path is None:
        raw = market_data.fetch_option_chain(ticker, progress=progress)
    else:
        raw = market_data.load_option_snapshots([path])
    ticker = raw["ticker"].iloc[0]
    data = LevelsData(ticker, raw["snapshotTime"].iloc[0], build_dataset(raw, FILTERS), pd.DataFrame(),
                      float(raw["openInterest"].fillna(0).sum()) if "openInterest" in raw else 0.0)
    # A saved snapshot passed this check when it was saved (or predates it), so only quote
    # coverage is re-checked; a live fetch must also fall within US trading hours.
    problem = market_data.snapshot_problem(raw) if path is None else (
        "most quotes are blank" if market_data.live_quote_share(raw) < market_data.MIN_LIVE_SHARE else None)
    if problem:
        data.notes.append(f"These levels are unreliable: {problem}. Fetch between 14:45 and 21:00 UK.")
    if data.dataset.empty:
        return data
    data.moves = analytics.implied_moves(data.dataset)
    if data.open_interest == 0:
        data.notes.append("No open interest in this chain (Yahoo blanks it before the US open), "
                          "so gamma levels aren't available.")
    data.futures = instruments.futures_for(ticker)
    if data.futures:
        try:
            data.futures_ratio = futures_ratio(data.futures, data.snapshot_time, float(data.dataset["S"].iloc[0]))
        except Exception:
            data.notes.append(f"Couldn't load {data.futures.price_ticker} to convert levels to futures points.")
    return data


def format_money(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    value = abs(value)
    return f"{sign}${value / 1e9:.2f}bn" if value >= 1e9 else f"{sign}${value / 1e6:.0f}m"


class LevelsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data: LevelsData | None = None
        self.levels: dict | None = None

        self.ticker = ticker_box(include_yields=False)
        self.ticker.setCurrentText("NQ — Nasdaq-100 E-mini")
        self.ticker.currentTextChanged.connect(self.refresh_sources)
        self.source = QComboBox()
        self.source.setMinimumWidth(240)
        self.fetch_button = QPushButton("Fetch live chain")
        self.fetch_button.clicked.connect(self.fetch)
        self.saved = QComboBox()
        self.saved.setMinimumWidth(200)
        self.saved.activated.connect(self.open_saved)
        self.gamma_window = QComboBox()
        self.gamma_window.addItems(GAMMA_WINDOWS)
        self.gamma_window.setToolTip("Which expiries count towards gamma. Short-dated options carry most of it.")
        self.gamma_window.currentIndexChanged.connect(self.recompute_gamma)
        self.status = status_label()
        self.status.setText("Pick a market and fetch its chain during US hours (14:30-21:00 UK), "
                            "or open a saved snapshot.")

        top = QHBoxLayout()
        top.addWidget(QLabel("Market"))
        top.addWidget(self.ticker, 1)
        top.addWidget(self.source, 1)
        top.addWidget(self.fetch_button)
        top.addWidget(QLabel("or open"))
        top.addWidget(self.saved)
        top.addSpacing(16)
        top.addWidget(QLabel("Gamma from"))
        top.addWidget(self.gamma_window)

        self.summary = {name: (QLabel("–"), QLabel("")) for name in SUMMARY}
        summary_box = QGroupBox("Levels")
        grid = QGridLayout(summary_box)
        for i, (name, (value, futures)) in enumerate(self.summary.items()):
            grid.addWidget(QLabel(name), 0, i)
            value.setStyleSheet("font-size: 15px; font-weight: 600;")
            grid.addWidget(value, 1, i)
            grid.addWidget(futures, 2, i)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.chart = QComboBox()
        self.chart.addItems(CHARTS)
        self.chart.currentIndexChanged.connect(self.redraw)
        self.plot = PlotCanvas()
        self.plot.message("Fetch a chain to see implied moves and gamma levels")
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.chart)
        right_layout.addWidget(self.plot)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(right)
        splitter.setSizes([560, 640])

        explainer = status_label()
        explainer.setText(
            "Implied move: the size (not direction) of move the options market is pricing. The straddle is "
            "roughly the expected move; the 1σ range holds about 68% of the time. Gamma assumes dealers are long "
            "customers' calls and short their puts: positive net gamma tends to damp moves, negative gamma "
            "amplifies them. Futures levels are converted at the futures/underlying price ratio.")

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.status)
        layout.addWidget(summary_box)
        layout.addWidget(splitter, 1)
        layout.addWidget(explainer)
        self.refresh_sources()
        self.refresh_saved()

    # --- Choosing what to load ----------------------------------------------------

    def refresh_sources(self):
        symbol = ticker_of(self.ticker)
        inst = instruments.resolve(symbol)
        self.source.clear()
        if inst is None:
            self.source.addItem(f"{symbol} options", symbol)
            return
        if inst.index_options:
            self.source.addItem(f"{inst.index_options.lstrip('^')} index options (European)", inst.index_options)
        thin = " – thin" if inst.proxy_is_thin else ""
        self.source.addItem(f"{inst.options_proxy} ETF options{thin}", inst.options_proxy)

    def refresh_saved(self):
        self.saved.clear()
        self.saved.addItem("Saved snapshot…", None)
        for path in reversed(market_data.list_option_snapshots()):
            self.saved.addItem(market_data.snapshot_label(path), path)

    def showEvent(self, event):
        self.refresh_saved()
        super().showEvent(event)

    def fetch(self):
        ticker = self.source.currentData()
        if not ticker:
            return
        self.fetch_button.setEnabled(False)
        self.status.setText(f"Fetching the {ticker} option chain…")
        run_in_background(self, load_levels, ticker, on_done=self.on_loaded, on_error=self.on_failed,
                          on_progress=self.status.setText)

    def open_saved(self, index: int):
        path = self.saved.itemData(index)
        if path is None:
            return
        self.status.setText(f"Opening {path.name}…")
        run_in_background(self, load_levels, None, path, on_done=self.on_loaded, on_error=self.on_failed)

    def on_failed(self, message: str):
        self.fetch_button.setEnabled(True)
        self.status.setText(f"Couldn't load the chain: {message}")

    def on_loaded(self, data: LevelsData):
        self.fetch_button.setEnabled(True)
        self.refresh_saved()
        self.data = data
        when = market_data.market_date(data.snapshot_time)
        futures = (f" Levels converted to {data.futures.symbol} at a ratio of {data.futures_ratio:.4f}."
                   if np.isfinite(data.futures_ratio) else "")
        self.status.setText(f"{data.ticker} on {when:%a %d %b %Y}: {len(data.dataset):,} quoted contracts, "
                            f"open interest {data.open_interest:,.0f}.{futures} {' '.join(data.notes)}")
        self.show_moves()
        self.recompute_gamma()

    # --- Results ------------------------------------------------------------------

    def in_futures(self, level: float) -> str:
        data = self.data
        if data is None or not np.isfinite(data.futures_ratio) or not np.isfinite(level):
            return ""
        return f"≈ {data.futures.symbol} {level * data.futures_ratio:,.2f}"

    def show_moves(self):
        moves = self.data.moves
        headers = ["Expiry", "Days", "ATM strike", "ATM IV", "Straddle", "Straddle %", "1σ move", "1σ range"]
        if np.isfinite(self.data.futures_ratio):
            headers.append(f"{self.data.futures.symbol} range")
        rows = []
        for m in moves.itertuples():
            row = [m.expiry, f"{m.days:.1f}", f"{m.atm_strike:g}", f"{m.atm_iv:.1%}", f"{m.straddle:,.2f}",
                   f"±{m.straddle_pct:.2%}", f"±{m.move:,.2f}", f"{m.low:,.2f} – {m.high:,.2f}"]
            if np.isfinite(self.data.futures_ratio):
                r = self.data.futures_ratio
                row.append(f"{m.low * r:,.2f} – {m.high * r:,.2f}")
            rows.append(row)
        fill_table(self.table, headers, rows)

        spot = float(self.data.dataset["S"].iloc[0]) if not self.data.dataset.empty else np.nan
        self.set_summary("Spot", f"{spot:,.2f}", self.in_futures(spot))
        day = analytics.one_day_move(moves) if not moves.empty else None
        if day:
            self.set_summary("1-day implied move", f"±{day['move']:,.2f} ({day['move_pct']:.2%})",
                             f"≈ ±{day['move'] * self.data.futures_ratio:,.2f} {self.data.futures.symbol} pts"
                             if np.isfinite(self.data.futures_ratio) else f"from {day['expiry']} ATM IV")
        else:
            self.set_summary("1-day implied move", "–", "")

    def recompute_gamma(self):
        self.levels = None
        if self.data is not None and not self.data.dataset.empty and self.data.open_interest > 0:
            df = analytics.within_days(self.data.dataset, GAMMA_WINDOWS[self.gamma_window.currentText()])
            self.levels = analytics.gamma_levels(df) if not df.empty else None
        if self.levels is None:
            for name in SUMMARY[2:]:
                self.set_summary(name, "–", "")
        else:
            lv = self.levels
            regime = "dealers damp moves" if lv["net"] > 0 else "dealers amplify moves"
            self.set_summary("Net gamma (per 1%)", format_money(lv["net"]), regime)
            for name, key in (("Gamma flip", "flip"), ("Call wall", "call_wall"), ("Put wall", "put_wall")):
                value = lv[key]
                self.set_summary(name, f"{value:,.2f}" if np.isfinite(value) else "none in ±15%",
                                 self.in_futures(value))
        self.redraw()

    def set_summary(self, name: str, value: str, futures: str):
        self.summary[name][0].setText(value)
        self.summary[name][1].setText(futures)

    def redraw(self):
        if self.data is None or self.data.dataset.empty:
            return
        chart = self.chart.currentText()
        ax = self.plot.axes()
        spot = float(self.data.dataset["S"].iloc[0])
        if chart == "Implied move by expiry":
            moves = self.data.moves[self.data.moves["days"] <= 120]
            ax.fill_between(moves["days"], moves["low"], moves["high"], color=CALL_COLOUR, alpha=0.15,
                            label="1σ range")
            ax.plot(moves["days"], spot + moves["straddle"], color=PUT_COLOUR, lw=1, label="± straddle")
            ax.plot(moves["days"], spot - moves["straddle"], color=PUT_COLOUR, lw=1)
            ax.axhline(spot, color="grey", lw=1, label="Spot")
            ax.set_xlabel("Days to expiry")
            ax.set_ylabel("Price")
        elif self.levels is None:
            self.plot.message("Gamma needs open interest: fetch during or after the US session")
            return
        elif chart == "Gamma exposure by strike":
            by_strike = self.levels["by_strike"]
            by_strike = by_strike[(by_strike.index > spot * 0.9) & (by_strike.index < spot * 1.1)]
            width = np.median(np.diff(by_strike.index)) * 0.8 if len(by_strike) > 1 else 1.0
            ax.bar(by_strike.index, by_strike["call"] / 1e6, width=width, color=CALL_COLOUR, alpha=0.6, label="Calls")
            ax.bar(by_strike.index, by_strike["put"] / 1e6, width=width, color=PUT_COLOUR, alpha=0.6, label="Puts")
            ax.plot(by_strike.index, by_strike["net"] / 1e6, color=NN_COLOUR, lw=1.2, label="Net")
            self._mark_levels(ax, spot)
            ax.set_xlabel("Strike")
            ax.set_ylabel("Gamma exposure ($m per 1% move)")
        else:
            spots, curve = self.levels["curve"]
            ax.plot(spots, curve / 1e9, color=NN_COLOUR, label="Net gamma")
            ax.axhline(0, color="grey", lw=0.8)
            ax.fill_between(spots, curve / 1e9, 0, where=curve > 0, color=CALL_COLOUR, alpha=0.12)
            ax.fill_between(spots, curve / 1e9, 0, where=curve < 0, color=PUT_COLOUR, alpha=0.12)
            self._mark_levels(ax, spot)
            ax.set_xlabel("Underlying price")
            ax.set_ylabel("Net gamma exposure ($bn per 1% move)")
        ax.set_title(f"{self.data.ticker} · {chart}")
        self.plot.legend(ax)
        self.plot.draw()

    def _mark_levels(self, ax, spot: float):
        ax.axvline(spot, color="grey", lw=1, label="Spot")
        styles = (("flip", "Gamma flip", ACCENT_COLOUR, "--"), ("call_wall", "Call wall", CALL_COLOUR, ":"),
                  ("put_wall", "Put wall", PUT_COLOUR, ":"))
        for key, name, colour, ls in styles:
            if np.isfinite(self.levels[key]):
                ax.axvline(self.levels[key], color=colour, ls=ls, lw=1.2, label=f"{name} {self.levels[key]:,.0f}")
