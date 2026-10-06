"""Main window and entry point."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtGui import QCursor, QGuiApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget

from bsnn import __version__, paths
from bsnn.gui.chain_tab import ChainTab
from bsnn.gui.common import AppState, run_in_background
from bsnn.gui.levels_tab import LevelsTab
from bsnn.gui.market_tab import MarketTab
from bsnn.gui.model_tab import ModelTab
from bsnn.gui.pricer_tab import PricerTab
from bsnn.model import OptionPricer


class MainWindow(QMainWindow):
    def __init__(self, load_latest_model: bool = True):
        super().__init__()
        self.setWindowTitle(f"Black-Scholes & Neural Networks {__version__}")
        self.resize(1280, 820)
        self.state = AppState()

        self.pricer_tab = PricerTab(self.state)
        self.market_tab = MarketTab()
        self.chain_tab = ChainTab(self.state)
        self.levels_tab = LevelsTab()
        self.model_tab = ModelTab(self.state)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self.pricer_tab, "Pricer")
        tabs.addTab(self.market_tab, "Market")
        tabs.addTab(self.chain_tab, "Option chain")
        tabs.addTab(self.levels_tab, "Moves && gamma")
        tabs.addTab(self.model_tab, "Neural network")
        self.setCentralWidget(tabs)

        self.state.pricer_changed.connect(self.show_model_status)
        self.statusBar().showMessage("No neural-network model loaded. Train one in the Neural network tab.")
        if load_latest_model:
            self.load_latest_model()

    def load_latest_model(self):
        models = sorted(paths.MODELS_DIR.glob("*.keras"))
        if not models:
            return
        self.statusBar().showMessage(f"Loading model {models[-1].name}…")
        run_in_background(self, OptionPricer.load, models[-1], on_done=self.model_tab.on_loaded,
                          on_error=lambda msg: self.statusBar().showMessage(f"Couldn't load the saved model: {msg}"))

    def show_model_status(self, pricer):
        self.statusBar().showMessage(f"Neural network: {pricer.description}" if pricer else "No model loaded.")


def place_on_current_screen(window: QMainWindow) -> None:
    """Centre the window on the screen the mouse is on, shrunk to fit if that screen is small.

    Left to itself, Windows can open the window on another monitor, out of sight.
    """
    screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
    area = screen.availableGeometry()
    window.resize(min(window.width(), area.width() - 40), min(window.height(), area.height() - 40))
    frame = window.frameGeometry()
    frame.moveCenter(area.center())
    window.move(frame.topLeft())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Black-Scholes & Neural Networks")
    window = MainWindow()
    place_on_current_screen(window)
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
