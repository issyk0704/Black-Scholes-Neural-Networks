import os
from PyQt5.QtCore import QThread, pyqtSignal
import yfinance as yf

class DataFetcher(QThread):
    finished = pyqtSignal(str)

    def __init__(self, tickers, folder_path):
        super().__init__()

        self.folder_path = os.path.expanduser("~/Desktop/BS-GUI/Stock-Data")
        self.tickers = tickers
        self.ensure_folder_exists(self.folder_path)

    def run(self):
        try:
            self.remove_old_files()
            for ticker in self.tickers:
                data = yf.Ticker(ticker).history(period="2y")
                file_path = os.path.join(self.folder_path, f"{ticker}_data.csv")
                data.to_csv(file_path)
            self.finished.emit("Data fetching complete.")
        except Exception as e:
            self.finished.emit(f"An error occurred: {e}")

    def ensure_folder_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def remove_old_files(self):
        for ticker in self.tickers:
            file_path = os.path.join(self.folder_path, f"{ticker}_data.csv")
            if os.path.exists(file_path):
                os.remove(file_path)



