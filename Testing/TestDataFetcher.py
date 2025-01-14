import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
from data_fetcher import DataFetcher  #

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        # Mocking the os.path.exists and os.makedirs to prevent actual directory operations
        self.patcher_exists = patch('os.path.exists', return_value=True)
        self.mock_exists = self.patcher_exists.start()
        self.patcher_makedirs = patch('os.makedirs')
        self.mock_makedirs = self.patcher_makedirs.start()
        self.patcher_remove = patch('os.remove')
        self.mock_remove = self.patcher_remove.start()

        # Mock yfinance Ticker and its history method
        self.patcher_yf = patch('yfinance.Ticker')
        self.mock_yf = self.patcher_yf.start()
        self.mock_history = MagicMock()
        self.mock_yf.return_value.history = MagicMock(return_value=pd.DataFrame({'Close': [100, 101, 102], 'Open': [99, 100, 101]}))

        # Mock the to_csv function to prevent any actual file writing
        self.patcher_csv = patch('pandas.DataFrame.to_csv')
        self.mock_csv = self.patcher_csv.start()

        # Initialise the DataFetcher with mock data
        self.fetcher = DataFetcher(["AAPL", "MSFT"], "~/Desktop/BS-GUI/Stock-Data")
        self.fetcher.finished = MagicMock()  # Mock the signal

    def tearDown(self):
        self.patcher_exists.stop()
        self.patcher_makedirs.stop()
        self.patcher_remove.stop()
        self.patcher_yf.stop()
        self.patcher_csv.stop()

    def test_initialization(self):
        self.assertEqual(self.fetcher.tickers, ["AAPL", "MSFT"])
        self.assertEqual(self.fetcher.folder_path, os.path.expanduser("~/Desktop/BS-GUI/Stock-Data"))

    def test_folder_creation(self):
        # Simulate the folder not existing before calling ensure_folder_exists
        self.mock_exists.return_value = False
        self.fetcher.ensure_folder_exists(self.fetcher.folder_path)
        self.mock_makedirs.assert_called_once_with(os.path.expanduser("~/Desktop/BS-GUI/Stock-Data"))

    def test_file_removal(self):
        self.fetcher.remove_old_files()
        self.assertEqual(self.mock_remove.call_count, len(self.fetcher.tickers))

    def test_data_fetching(self):
        self.fetcher.run()
        self.fetcher.finished.emit.assert_called_once_with("Data fetching complete.")

    def test_error_handling(self):
        # Simulate an error during the fetching process
        self.mock_yf.return_value.history.side_effect = Exception("Error fetching data")
        self.fetcher.run()
        self.fetcher.finished.emit.assert_called_once_with("An error occurred: Error fetching data")

if __name__ == '__main__':
    unittest.main()
