"""Render every tab with real data into docs/screenshots/ (used by the README).

    python scripts/screenshots.py

Trains a fresh default model on all saved snapshots and fetches live prices for
SPY, so it needs a network connection and takes about a minute.
"""

import sys
import threading
from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from bsnn import market_data
from bsnn.gui.app import MainWindow
from bsnn.gui.model_tab import train_from_files
from bsnn.gui.pricer_tab import load_market_inputs
from bsnn.model import TrainConfig

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow(load_latest_model=False)
    window.resize(1400, 860)
    window.show()
    tabs = window.centralWidget()

    files = market_data.list_option_snapshots()
    result = train_from_files(files, TrainConfig(split="date"), progress=lambda _: None, stop=threading.Event())
    window.model_tab.split.setCurrentIndex(window.model_tab.split.findData("date"))
    window.model_tab.on_trained(result)

    pricer = window.pricer_tab
    pricer.ticker.setCurrentText("SPY")
    pricer.on_loaded(load_market_inputs("SPY"))
    pricer.days.setValue(45)

    window.market_tab.ticker.setCurrentText("SPY")
    window.market_tab.on_loaded(("SPY", "2y"), market_data.get_history("SPY"))

    chain = window.chain_tab
    latest_spy = [f for f in files if f.name.startswith("SPY")][-1]
    chain.set_chain(market_data.load_option_snapshots([latest_spy]), f"saved snapshot {latest_spy.name}")

    for index, name in enumerate(["pricer", "market", "chain", "model"]):
        tabs.setCurrentIndex(index)
        app.processEvents()
        window.grab().save(str(OUT / f"{name}.png"))
        print("saved", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
