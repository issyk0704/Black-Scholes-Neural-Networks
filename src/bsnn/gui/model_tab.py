"""Train, evaluate, save and load the neural-network pricer."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLineEdit,
                             QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
                             QSplitter, QTableWidget, QVBoxLayout, QWidget)

from bsnn import market_data, paths
from bsnn.features import build_dataset
from bsnn.gui.common import (ACCENT_COLOUR, CALL_COLOUR, NN_COLOUR, PUT_COLOUR, AppState, PlotCanvas, fill_table,
                             run_in_background, spin, status_label)
from bsnn.instruments import ASSET_CLASSES, asset_class_of
from bsnn.model import (BS_ATM, BS_HIST, MODEL_KINDS, NN, SPLITS, OptionPricer, TrainConfig, TrainingResult,
                        metrics_by_class, train)

CHARTS = ["Predicted vs market price", "Error by asset class", "Error by moneyness", "Error by days to expiry",
          "Training loss"]
MODEL_COLOURS = {NN: NN_COLOUR, BS_HIST: ACCENT_COLOUR, BS_ATM: CALL_COLOUR}


def train_from_files(files: list[Path], config: TrainConfig, progress, stop: threading.Event) -> TrainingResult:
    progress("Building dataset…")
    dataset = build_dataset(market_data.load_option_snapshots(files))
    progress(f"Training on {len(dataset):,} contracts…")
    return train(dataset, config, on_epoch=lambda epoch, total, logs: progress((epoch, total, logs)),
                 should_stop=stop.is_set)


class ModelTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.result: TrainingResult | None = None
        self.stop_flag = threading.Event()

        # Data
        self.snapshots = QListWidget()
        self.snapshots.setMinimumHeight(140)
        self.class_filter = QComboBox()
        self.class_filter.addItem("Tick all asset classes", None)
        for asset_class in ASSET_CLASSES:
            self.class_filter.addItem(f"Tick only {asset_class.lower()}", asset_class)
        self.class_filter.activated.connect(self.tick_asset_class)
        data_box = QGroupBox("Option snapshots to learn from")
        data_layout = QVBoxLayout(data_box)
        data_layout.addWidget(self.class_filter)
        data_layout.addWidget(self.snapshots)
        data_hint = status_label()
        data_hint.setText("Fetch more in the Option chain tab; each fetch is saved here.")
        data_layout.addWidget(data_hint)

        # Settings
        self.kind = QComboBox()
        for key, label in MODEL_KINDS.items():
            self.kind.addItem(label, key)
        self.reference = QComboBox()
        self.reference.addItem("Expiry's ATM implied vol", True)
        self.reference.addItem("30-day historical vol", False)
        self.split = QComboBox()
        for key, label in SPLITS.items():
            self.split.addItem(label, key)
        self.split.setToolTip("Random expiries: test on unseen expiries from the same days.\n"
                              "Newest snapshot date: train on older days, test on the latest one.")
        defaults = TrainConfig()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.epochs.setValue(defaults.epochs)
        self.layers = QLineEdit(", ".join(map(str, defaults.hidden_layers)))
        self.learning_rate = spin(defaults.learning_rate, 1e-5, 1.0, 5, step=1e-4)
        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        self.seed.setValue(defaults.seed)
        settings = QGroupBox("Model")
        form = QFormLayout(settings)
        self.use_asset_class = QCheckBox("Tell the network each contract's asset class")
        self.use_asset_class.setChecked(defaults.use_asset_class)
        self.use_asset_class.setToolTip("Equity, metals and bond smiles have different shapes; with this ticked,\n"
                                        "one model can learn all of them.")
        form.addRow("Model type", self.kind)
        form.addRow("Volatility input", self.reference)
        form.addRow("", self.use_asset_class)
        form.addRow("Test set", self.split)
        form.addRow("Max epochs", self.epochs)
        form.addRow("Hidden layers", self.layers)
        form.addRow("Learning rate", self.learning_rate)
        form.addRow("Random seed", self.seed)

        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_flag.set)
        self.progress = QProgressBar()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        buttons = QHBoxLayout()
        buttons.addWidget(self.train_button)
        buttons.addWidget(self.stop_button)

        # Saved models
        self.saved = QComboBox()
        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_selected)
        self.save_button = QPushButton("Save current model")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_current)
        self.active = status_label()
        self.active.setText("No model loaded.")
        saved_box = QGroupBox("Saved models")
        saved_layout = QVBoxLayout(saved_box)
        row = QHBoxLayout()
        row.addWidget(self.saved, 1)
        row.addWidget(load_button)
        saved_layout.addLayout(row)
        saved_layout.addWidget(self.save_button)
        saved_layout.addWidget(self.active)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(data_box)
        left_layout.addWidget(settings)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.log, 1)
        left_layout.addWidget(saved_box)

        # Results
        self.metrics = QTableWidget()
        self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.metrics.setMaximumHeight(140)
        self.results_note = status_label()
        self.results_note.setText("Results on the held-out test set appear here after training.")
        self.chart = QComboBox()
        self.chart.addItems(CHARTS)
        self.chart.currentIndexChanged.connect(self.redraw)
        self.plot = PlotCanvas()
        self.plot.message("Train a model to compare it with Black-Scholes")
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.metrics)
        right_layout.addWidget(self.results_note)
        right_layout.addWidget(self.chart)
        right_layout.addWidget(self.plot, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([380, 820])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        state.pricer_changed.connect(self.on_pricer_changed)
        self.refresh_snapshots()
        self.refresh_saved()

    def showEvent(self, event):
        self.refresh_snapshots()
        super().showEvent(event)

    # --- Lists ------------------------------------------------------------------

    def refresh_snapshots(self):
        unchecked = {self.snapshots.item(i).data(Qt.ItemDataRole.UserRole)
                     for i in range(self.snapshots.count())
                     if self.snapshots.item(i).checkState() == Qt.CheckState.Unchecked}
        self.snapshots.clear()
        for path in market_data.list_option_snapshots():
            item = QListWidgetItem(market_data.snapshot_label(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked if path in unchecked else Qt.CheckState.Checked)
            self.snapshots.addItem(item)

    def tick_asset_class(self, index: int):
        wanted = self.class_filter.itemData(index)
        for i in range(self.snapshots.count()):
            item = self.snapshots.item(i)
            ticker = item.data(Qt.ItemDataRole.UserRole).name.split("_options_")[0]
            keep = wanted is None or asset_class_of(ticker) == wanted
            item.setCheckState(Qt.CheckState.Checked if keep else Qt.CheckState.Unchecked)

    def selected_files(self) -> list[Path]:
        return [self.snapshots.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.snapshots.count())
                if self.snapshots.item(i).checkState() == Qt.CheckState.Checked]

    def refresh_saved(self):
        self.saved.clear()
        for path in sorted(paths.MODELS_DIR.glob("*.keras"), reverse=True):
            self.saved.addItem(path.stem, path)

    # --- Training ---------------------------------------------------------------

    def config(self) -> TrainConfig:
        layers = tuple(int(x) for x in self.layers.text().replace(" ", "").split(",") if x)
        if not layers or min(layers) < 1:
            raise ValueError("Hidden layers should be a list of sizes, e.g. 64, 64, 64")
        return TrainConfig(kind=self.kind.currentData(), use_atm_iv=self.reference.currentData(),
                           hidden_layers=layers, epochs=self.epochs.value(),
                           learning_rate=self.learning_rate.value(), split=self.split.currentData(),
                           seed=self.seed.value(), use_asset_class=self.use_asset_class.isChecked())

    def start_training(self):
        try:
            config = self.config()
        except ValueError as exc:
            self.log.appendPlainText(str(exc))
            return
        files = self.selected_files()
        if not files:
            self.log.appendPlainText("Tick at least one snapshot.")
            return
        self.stop_flag.clear()
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, config.epochs)
        self.progress.setValue(0)
        self.log.clear()
        run_in_background(self, train_from_files, files, config, stop=self.stop_flag,
                          on_done=self.on_trained, on_error=self.on_failed, on_progress=self.on_progress)

    def on_progress(self, update):
        if isinstance(update, str):
            self.log.appendPlainText(update)
            return
        epoch, total, logs = update
        self.progress.setValue(epoch)
        if epoch == 1 or epoch % 10 == 0:
            self.log.appendPlainText(f"epoch {epoch}/{total}  loss {logs.get('loss', np.nan):.5f}  "
                                     f"val_loss {logs.get('val_loss', np.nan):.5f}")

    def on_failed(self, message: str):
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log.appendPlainText(f"Training failed: {message}")

    def on_trained(self, result: TrainingResult):
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setValue(self.progress.maximum())
        self.result = result
        meta = result.pricer.metadata
        self.log.appendPlainText(f"Done after {meta['epochs_run']} epochs.")
        self.save_button.setEnabled(True)
        self.state.set_pricer(result.pricer)
        self.show_metrics(result.metrics, meta)
        self.redraw()

    # --- Saving and loading -----------------------------------------------------

    def save_current(self):
        if self.result is None:
            return
        path = self.result.pricer.save()
        self.save_button.setEnabled(False)
        self.log.appendPlainText(f"Saved to {path}")
        self.refresh_saved()

    def load_selected(self):
        path = self.saved.currentData()
        if path is None:
            return
        self.active.setText(f"Loading {path.name}…")
        run_in_background(self, OptionPricer.load, path, on_done=self.on_loaded,
                          on_error=lambda msg: self.active.setText(f"Couldn't load the model: {msg}"))

    def on_loaded(self, pricer: OptionPricer):
        self.result = None
        self.save_button.setEnabled(False)
        self.state.set_pricer(pricer)
        stored = pricer.metadata.get("test_metrics")
        if stored:
            self.show_metrics(pd.DataFrame(stored).T, pricer.metadata)
        self.plot.message("Charts are available for models trained in this session")

    def on_pricer_changed(self, pricer: OptionPricer | None):
        if pricer is None:
            self.active.setText("No model loaded.")
            return
        meta = pricer.metadata
        trained = meta.get("created", "this session").replace("T", " ")
        self.active.setText(f"Active: {pricer.description}. Trained {trained} on "
                            f"{', '.join(meta.get('tickers', []))} ({', '.join(meta.get('snapshot_dates', []))}).")

    # --- Results ----------------------------------------------------------------

    def show_metrics(self, metrics: pd.DataFrame, meta: dict):
        fill_table(self.metrics, list(metrics.columns),
                   [[f"{v:,.2f}" for v in row] for row in metrics.to_numpy()], list(metrics.index))
        rows = meta.get("rows", {})
        split = SPLITS.get(meta.get("config", {}).get("split"), "")
        self.results_note.setText(
            f"Test set: {rows.get('test', 0):,} contracts the model never saw ({split.lower()}); "
            f"trained on {rows.get('train', 0):,}. Lower error is better; 'Inside bid-ask' is the share of "
            f"prices that land between the bid and the ask.")

    def redraw(self):
        if self.result is None:
            return
        chart = self.chart.currentText()
        df = self.result.predictions
        ax = self.plot.axes()
        if chart == "Training loss":
            hist = self.result.history
            ax.plot(hist["loss"], color=CALL_COLOUR, label="Training")
            ax.plot(hist["val_loss"], color=PUT_COLOUR, label="Validation")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (MSE)")
        elif chart == "Predicted vs market price":
            for name in (BS_HIST, BS_ATM, NN):
                ax.scatter(df["mid"], df[name], s=5, alpha=0.35, color=MODEL_COLOURS[name], label=name)
            top = float(df["mid"].quantile(0.99))
            ax.plot([0, top], [0, top], color="grey", lw=1, label="Perfect")
            ax.set_xlim(0, top)
            ax.set_ylim(0, top * 1.2)
            ax.set_xlabel("Market mid price")
            ax.set_ylabel("Model price")
        elif chart == "Error by asset class":
            # Percentage error, because dollar errors on a $5 ETF and a $700 one aren't comparable.
            by_class = metrics_by_class(df)["Median abs error (%)"].unstack("Model")
            x = np.arange(len(by_class))
            for offset, name in zip((-0.27, 0.0, 0.27), (BS_HIST, BS_ATM, NN)):
                ax.bar(x + offset, by_class[name], width=0.27, color=MODEL_COLOURS[name], label=name)
            counts = df.groupby("asset_class").size()
            ax.set_xticks(x, [f"{c}\n({counts[c]:,} contracts)" for c in by_class.index])
            ax.set_ylabel("Median absolute error (%)")
        else:
            if chart == "Error by moneyness":
                bucket = pd.cut(df["S"] / df["K"], np.arange(0.7, 1.31, 0.05))
                ax.set_xlabel("Spot / strike")
            else:
                bucket = pd.cut(df["time_to_expiry"] * 365, [0, 7, 30, 60, 90, 180, 365, 730])
                ax.set_xlabel("Days to expiry")
            for name in (BS_HIST, BS_ATM, NN):
                err = (df[name] - df["mid"]).abs().groupby(bucket, observed=True).mean()
                ax.plot([str(i) for i in err.index], err.to_numpy(), marker="o", color=MODEL_COLOURS[name], label=name)
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylabel("Mean absolute error ($)")
        ax.set_title(f"{chart} (test set)" if chart != "Training loss" else chart)
        self.plot.legend(ax)
        self.plot.draw()
