"""Live or saved option chains, priced by Black-Scholes and the neural network side by side."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from bsnn import instruments, market_data, pricing
from bsnn.features import ATM_IV_FEATURE, Filters, build_dataset
from bsnn.gui.common import (ACCENT_COLOUR, CALL_COLOUR, NN_COLOUR, PUT_COLOUR, AppState, PlotCanvas, fill_table,
                             run_in_background, status_label, ticker_box, ticker_of)
from bsnn.model import black_scholes_prices

# Looser than the training filters: show everything with a usable two-sided quote.
DISPLAY_FILTERS = Filters(min_days=0.05, min_moneyness=0.5, max_moneyness=2.0, max_relative_spread=1.0, min_price=0.01)
TRAINING_FILTERS = Filters()
INSIDE_SPREAD = QColor(47, 163, 107, 70)


def _num(value: float, fmt: str = "{:.2f}") -> QTableWidgetItem:
    item = QTableWidgetItem("–" if not np.isfinite(value) else fmt.format(value))
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


class ChainTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.chain: pd.DataFrame | None = None
        self.source = ""
        self._source = ""  # description of the fetch in progress
        self._spot_row = 0

        self.ticker = ticker_box()
        self.fetch_button = QPushButton("Fetch live chain")
        self.fetch_button.setToolTip("Downloads every expiry and saves it to data/options for training")
        self.fetch_button.clicked.connect(self.fetch)
        self.saved = QComboBox()
        self.saved.setMinimumWidth(220)
        self.saved.activated.connect(self.open_saved)
        self.expiry = QComboBox()
        self.expiry.setMinimumWidth(160)
        self.expiry.currentIndexChanged.connect(self.show_expiry)
        self.side = QComboBox()
        self.side.addItems(["Calls and puts", "Calls", "Puts"])
        self.side.currentIndexChanged.connect(self.show_expiry)
        self.status = status_label()

        top = QHBoxLayout()
        top.addWidget(QLabel("Ticker"))
        top.addWidget(self.ticker, 1)
        top.addWidget(self.fetch_button)
        top.addWidget(QLabel("or open"))
        top.addWidget(self.saved, 1)
        top.addSpacing(24)
        top.addWidget(QLabel("Expiry"))
        top.addWidget(self.expiry)
        top.addWidget(self.side)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSortingEnabled(False)
        self.plot = PlotCanvas()
        self.plot.message("Fetch a live chain or open a saved snapshot")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.plot)
        splitter.setSizes([640, 560])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.status)
        layout.addWidget(splitter, 1)

        state.pricer_changed.connect(lambda _: self.show_expiry())
        self.refresh_saved()

    # --- Loading ----------------------------------------------------------------

    def refresh_saved(self):
        self.saved.clear()
        self.saved.addItem("Saved snapshot…", None)
        for path in reversed(market_data.list_option_snapshots()):
            self.saved.addItem(market_data.snapshot_label(path), path)

    def showEvent(self, event):
        self.refresh_saved()
        super().showEvent(event)
        self.scroll_to_spot()

    def fetch(self):
        symbol = ticker_of(self.ticker)
        if not symbol:
            return
        inst = instruments.resolve(symbol)
        ticker = inst.options_proxy if inst else symbol
        self._source = "live, saved to data/options"
        if inst:
            self._source = f"options proxy for {symbol}, live, saved to data/options"
            if inst.proxy_is_thin:
                self._source += "; few strikes and expiries, so treat with caution"
        self.fetch_button.setEnabled(False)
        self.status.setText(f"Fetching the {ticker} option chain…")
        run_in_background(self, market_data.fetch_option_chain, ticker, on_done=self.on_fetched,
                          on_error=self.on_failed, on_progress=self.status.setText)

    def on_fetched(self, raw: pd.DataFrame):
        self.fetch_button.setEnabled(True)
        self.refresh_saved()
        if market_data.live_quote_share(raw) < 0.5:
            self._source += ". Most quotes are blank: Yahoo clears them outside US hours (14:30-21:00 UK)"
        self.set_chain(raw, self._source)

    def on_failed(self, message: str):
        self.fetch_button.setEnabled(True)
        self.status.setText(f"Couldn't fetch {ticker_of(self.ticker)}: {message}")

    def open_saved(self, index: int):
        path: Path | None = self.saved.itemData(index)
        if path is None:
            return
        try:
            raw = market_data.load_option_snapshots([path])
        except Exception as exc:
            self.status.setText(f"Couldn't open {path.name}: {exc}")
            return
        self.set_chain(raw, f"saved snapshot {path.name}")

    def set_chain(self, raw: pd.DataFrame, source: str):
        self.chain = build_dataset(raw, DISPLAY_FILTERS)
        self.source = source
        first = raw.iloc[0]
        if self.chain.empty:
            self.table.setRowCount(0)
            self.expiry.clear()
            self.plot.message("No contracts with a usable two-sided quote")
            self.status.setText(f"{first['ticker']} ({source}): no contracts with a usable quote.")
            return
        when = market_data.market_date(first["snapshotTime"])
        self.status.setText(
            f"{first['ticker']} ({source}) on {when:%d %b %Y}: spot {first['underlyingPrice']:,.2f}, "
            f"30d hist. vol {first['histVol']:.1%}, r {first['riskFreeRate']:.2%}, "
            f"q {first['dividendYield']:.2%}. {len(self.chain):,} of {len(raw):,} contracts have a usable quote.")
        self.expiry.blockSignals(True)
        self.expiry.clear()
        days = self.chain.groupby("expiry")["time_to_expiry"].first() * 365
        for expiry, d in days.sort_index().items():
            self.expiry.addItem(f"{expiry}  ({d:.0f}d)", expiry)
        # Default to the first expiry at least two weeks out: more strikes, smoother smile.
        later = [i for i, d in enumerate(days.sort_index()) if d >= 14]
        self.expiry.setCurrentIndex(later[0] if later else 0)
        self.expiry.blockSignals(False)
        self.show_expiry()

    # --- Display ----------------------------------------------------------------

    def selected(self) -> pd.DataFrame | None:
        if self.chain is None or self.expiry.currentData() is None:
            return None
        df = self.chain[self.chain["expiry"] == self.expiry.currentData()]
        side = self.side.currentText()
        if side != "Calls and puts":
            df = df[df["is_call"] == (1.0 if side == "Calls" else 0.0)]
        return df.sort_values(["K", "is_call"], ascending=[True, False]).reset_index(drop=True)

    def show_expiry(self):
        df = self.selected()
        if df is None or df.empty:
            return
        pricer = self.state.pricer
        df = df.assign(bs_hist=black_scholes_prices(df, "hist_vol"), bs_atm=black_scholes_prices(df, ATM_IV_FEATURE))
        headers = ["Strike", "Type", "Bid", "Ask", "Mid", "Implied vol", "BS (hist. vol)", "BS (ATM vol)"]
        if pricer is not None:
            # Only price contracts inside the moneyness range the model learned from.
            in_range = (df["S"] / df["K"]).between(TRAINING_FILTERS.min_moneyness, TRAINING_FILTERS.max_moneyness)
            df["nn"] = np.where(in_range, pricer.predict(df), np.nan)
            df["nn_iv"] = pricing.implied_volatility(df["nn"], df["S"], df["K"], df["time_to_expiry"],
                                                     df["risk_free_rate"], df["dividend_yield"], df["is_call"] > 0)
            headers += ["Neural net", "NN − mid"]

        rows = []
        for r in df.itertuples():
            row = [_num(r.K, "{:g}"), QTableWidgetItem("Call" if r.is_call else "Put"), _num(r.bid), _num(r.ask),
                   _num(r.mid), _num(r.market_iv * 100, "{:.1f}%"), _num(r.bs_hist), _num(r.bs_atm)]
            if pricer is not None:
                nn = _num(r.nn)
                if not np.isfinite(r.nn):
                    nn.setToolTip("Outside the spot/strike range the model was trained on")
                elif r.bid <= r.nn <= r.ask:
                    nn.setBackground(QBrush(INSIDE_SPREAD))
                    nn.setToolTip("Inside the bid-ask spread")
                row += [nn, _num(r.nn - r.mid, "{:+.2f}")]
            rows.append(row)
        fill_table(self.table, headers, rows)
        self._spot_row = int((df["K"] - df["S"]).abs().to_numpy().argmin())
        self.scroll_to_spot()
        self.draw_smile(df)

    def scroll_to_spot(self):
        # Deferred: a table that isn't laid out yet (hidden tab) ignores scroll requests.
        QTimer.singleShot(0, self._scroll_now)

    def _scroll_now(self):
        item = self.table.item(self._spot_row, 0)
        if item is not None:
            self.table.scrollToItem(item, QTableWidget.ScrollHint.PositionAtCenter)

    def draw_smile(self, df: pd.DataFrame):
        ax = self.plot.axes()
        for is_call, name, colour, marker in ((1.0, "Call", CALL_COLOUR, "o"), (0.0, "Put", PUT_COLOUR, "s")):
            side = df[df["is_call"] == is_call]
            if side.empty:
                continue
            ax.scatter(side["K"], side["market_iv"] * 100, s=14, color=colour, marker=marker, alpha=0.75,
                       label=f"{name} market IV")
            if "nn_iv" in side:
                ax.plot(side["K"], side["nn_iv"] * 100, color=colour, lw=1.5, label=f"{name} neural net")
        first = df.iloc[0]
        ax.axhline(first[ATM_IV_FEATURE] * 100, color=NN_COLOUR, ls="--", lw=1, label="ATM implied vol")
        ax.axhline(first["hist_vol"] * 100, color=ACCENT_COLOUR, ls=":", lw=1, label="30d hist. vol")
        ax.axvline(first["S"], color="grey", lw=1, alpha=0.6, label="Spot")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied volatility (%)")
        ax.set_title(f"{first['ticker']} volatility smile · expires {first['expiry']}")
        self.plot.legend(ax)
        self.plot.draw()
