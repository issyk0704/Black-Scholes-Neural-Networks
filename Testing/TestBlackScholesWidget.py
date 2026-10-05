import sys
import unittest
from PyQt5.QtWidgets import QApplication
from unittest.mock import Mock, patch
import pandas as pd


from black_scholes_Widget import BlackScholesWidget

class TestBlackScholesWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure QApplication is initialised only once for all tests
        if QApplication.instance() is None:
            cls.app = QApplication(sys.argv)
        # Prepare the DataFrame which is simulates the yfinance fetch data
        cls.mock_data = pd.DataFrame({
            'Close': [100, 102, 103, 105, 107],
            'Open': [99, 101, 102, 104, 106]
        })
        # Start patching yfinance Ticker to return a Mock with our DataFrame
        cls.patcher = patch('yfinance.Ticker', return_value=Mock(history=Mock(return_value=cls.mock_data)))
        cls.mock_ticker = cls.patcher.start()

    def setUp(self):
        # Mock the switch_to_selection_screen_callback to avoid GUI interaction during tests
        self.mock_callback = Mock()
        # Create an instance of BlackScholesWidget with the mocked callback
        self.widget = BlackScholesWidget(switch_to_selection_screen_callback=self.mock_callback)

    def test_initial_conditions(self):
        """Test that initial conditions in the widget are as expected."""
        self.assertIsNone(self.widget.currentTicker, "currentTicker should initially be None.")
        self.assertIsNone(self.widget.last_closing_price, "last_closing_price should initially be None.")

    def test_fetch_data(self):
        """Test fetchData method to check if data fetching updates state correctly."""
        self.widget.fetchData("AAPL")
        # Assert that the ticker was called with the correct argument
        self.mock_ticker.assert_called_with("AAPL")
        # Assert that the last closing price is correctly updated
        self.assertEqual(self.widget.last_closing_price, 107, "Last closing price should be the last item in 'Close' column.")

    def test_calculate_option_price(self):
        """Test the calculation of option prices."""
        # Setup test conditions
        self.widget.last_closing_price = 100  # Set a known last closing price
        self.widget.strikePrice.setText("95")
        self.widget.riskFreeRate.setText("0.05")
        self.widget.timeToMaturity.setText("1")
        self.widget.volatility.setText("0.2")
        # Run the method to test
        self.widget.calculateOptionPrice()
        # Check the results
        results = self.widget.optionsOutput.toPlainText()
        self.assertIn("Call Option Price:", results)
        self.assertIn("Put Option Price:", results)

    @classmethod
    def tearDownClass(cls):
        # Stop the patcher after all tests in this class
        cls.patcher.stop()
        # Quit the QApplication instance to clean up resources
        cls.app.quit()

if __name__ == '__main__':
    unittest.main()
