"""Shared GUI pieces: background workers, app-wide state, plots and input widgets."""

from __future__ import annotations

import logging
from collections.abc import Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QLabel, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from bsnn.instruments import INSTRUMENTS, label

log = logging.getLogger(__name__)

# Line colours that read on both light and dark backgrounds.
CALL_COLOUR, PUT_COLOUR, NN_COLOUR, ACCENT_COLOUR = "#2a7fd4", "#e0663a", "#2fa36b", "#9467bd"


# --- Background work ---------------------------------------------------------

class _Signals(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(object)


class Worker(QRunnable):
    """Runs ``fn`` on the global thread pool and reports back through Qt signals.

    With ``progress=True``, ``fn`` is called with a ``progress`` keyword argument
    it can call (from the worker thread) to send updates to the GUI.
    """

    def __init__(self, fn: Callable, *args, progress: bool = False, **kwargs):
        super().__init__()
        self.signals = _Signals()
        self._fn, self._args, self._kwargs = fn, args, kwargs
        if progress:
            self._kwargs["progress"] = self.signals.progress.emit

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:
            log.exception("Background task failed")
            self.signals.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.signals.finished.emit(result)


def run_in_background(owner: QObject, fn: Callable, *args, on_done: Callable, on_error: Callable,
                      on_progress: Callable | None = None, **kwargs) -> Worker:
    worker = Worker(fn, *args, progress=on_progress is not None, **kwargs)
    # Keep the worker (and its signals object) alive until it reports back.
    pending = owner.__dict__.setdefault("_pending_workers", set())
    pending.add(worker)
    worker.signals.finished.connect(on_done)
    worker.signals.failed.connect(on_error)
    if on_progress:
        worker.signals.progress.connect(on_progress)
    for signal in (worker.signals.finished, worker.signals.failed):
        signal.connect(lambda *_: pending.discard(worker))
    QThreadPool.globalInstance().start(worker)
    return worker


# --- App-wide state ----------------------------------------------------------

class AppState(QObject):
    """State shared between tabs: the active neural-network pricer."""

    pricer_changed = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.pricer = None

    def set_pricer(self, pricer) -> None:
        self.pricer = pricer
        self.pricer_changed.emit(pricer)


# --- Plotting ----------------------------------------------------------------

def _is_dark() -> bool:
    return QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128


class PlotCanvas(QWidget):
    """A Matplotlib figure with the standard zoom/pan toolbar, themed to match the app."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def axes(self):
        """Clear the figure and return a fresh, themed axes."""
        self.figure.clear()
        ax = self.figure.add_subplot()
        palette = QApplication.palette()
        bg = palette.color(QPalette.ColorRole.Base).name()
        fg = palette.color(QPalette.ColorRole.Text).name()
        self.figure.set_facecolor(palette.color(QPalette.ColorRole.Window).name())
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color(fg)
            spine.set_alpha(0.3)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        ax.grid(True, alpha=0.15 if _is_dark() else 0.3)
        return ax

    def legend(self, ax):
        palette = QApplication.palette()
        leg = ax.legend(facecolor=palette.color(QPalette.ColorRole.Base).name(), edgecolor="none", framealpha=0.85)
        for text in leg.get_texts():
            text.set_color(palette.color(QPalette.ColorRole.Text).name())

    def message(self, text: str):
        ax = self.axes()
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, alpha=0.6,
                color=QApplication.palette().color(QPalette.ColorRole.Text).name())
        self.canvas.draw_idle()

    def draw(self):
        self.canvas.draw_idle()


# --- Inputs and tables -------------------------------------------------------

def ticker_box(include_yields: bool = True) -> QComboBox:
    """The watchlist markets, plus any symbol Yahoo Finance knows typed in by hand."""
    box = QComboBox()
    box.setEditable(True)
    box.addItems([label(i) for i in INSTRUMENTS if include_yields or not i.is_yield])
    box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    box.setToolTip("Pick a watchlist market, or type any Yahoo Finance symbol (e.g. AAPL)")
    return box


def ticker_of(box: QComboBox) -> str:
    """The symbol in the box: 'NQ' for 'NQ — Nasdaq-100 E-mini', or whatever was typed."""
    return box.currentText().split("—")[0].strip().upper()


def spin(value: float, minimum: float, maximum: float, decimals: int = 2, suffix: str = "",
         step: float = 1.0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


def status_label() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    muted = QApplication.palette().color(QPalette.ColorRole.PlaceholderText).name()
    label.setStyleSheet(f"color: {muted};")
    return label


def fill_table(table: QTableWidget, headers: list[str], rows: list[list], row_labels: list[str] | None = None):
    table.clear()
    table.setColumnCount(len(headers))
    table.setRowCount(len(rows))
    table.setHorizontalHeaderLabels(headers)
    if row_labels is not None:
        table.setVerticalHeaderLabels(row_labels)
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            item = value if isinstance(value, QTableWidgetItem) else QTableWidgetItem(str(value))
            table.setItem(r, c, item)
