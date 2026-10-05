import sys
import unittest
from unittest.mock import Mock, patch
from PyQt5.QtWidgets import QApplication
import pandas as pd


from neural_network_widget import NeuralNetworkWidget

class TestNeuralNetworkWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the QApplication is initialised only once for all tests
        if QApplication.instance() is None:
            cls.app = QApplication(sys.argv)

        # Setting up all necessary patchers
        cls.mock_data = pd.DataFrame({
            'Close': [100, 105, 110], 'Open': [99, 104, 109], 'High': [106, 111, 116], 'Low': [98, 103, 108], 'Volume': [1000, 1500, 2000]
        })

        cls.patcher_yf = patch('yfinance.Ticker', return_value=Mock(history=Mock(return_value=cls.mock_data)))
        cls.mock_ticker = cls.patcher_yf.start()

        cls.patcher_os = patch('os.path.exists', return_value=True)
        cls.mock_os = cls.patcher_os.start()

        cls.patcher_pd = patch('pandas.read_csv', return_value=cls.mock_data)
        cls.mock_pd = cls.patcher_pd.start()

    def setUp(self):
        # Initialising the widget with a mock for the switch callback
        self.mock_callback = Mock()
        self.widget = NeuralNetworkWidget(switch_to_selection_screen_callback=self.mock_callback)

    def test_fetchData(self):
        """Test fetchData to ensure it processes and saves data correctly."""
        self.widget.fetchData()
        self.assertTrue(self.mock_ticker.called)
        self.assertIn("Downloaded latest data for SPY", self.widget.outputArea.toPlainText())

    def test_predictOptionPrices(self):
        """Test predictOptionPrices to ensure it calculates and displays correct prices."""
        # Setting the valid input values
        self.widget.strikePriceInput.setText("100")
        self.widget.timeToExpirationInput.setText("1")
        self.widget.riskFreeRateInput.setText("0.05")
        self.widget.impliedVolatilityInput.setText("0.2")

        # Select the ticker
        self.widget.tickerSelector.setCurrentIndex(0)  # Assume that 'SPY' is at index 0

        # Run the predict method
        self.widget.predictOptionPrices()

        # Check the output for expected results
        output = self.widget.outputArea.toPlainText()
        self.assertIn("Calculated Call Price:", output)
        self.assertIn("Calculated Put Price:", output)

    @classmethod
    def tearDownClass(cls):
        #  stop all patchers
        cls.patcher_yf.stop()
        cls.patcher_os.stop()
        cls.patcher_pd.stop()
        # Quit the QApplication instance to clean up resources
        cls.app.quit()

if __name__ == '__main__':
    unittest.main()
