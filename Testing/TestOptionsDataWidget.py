import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication
from OptionsDataWidget import OptionsDataWidget

class TestOptionsDataWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])  # Ensuring the QApplication instance is available for QWidget

    def setUp(self):
        self.widget = OptionsDataWidget(lambda: None)  # Lambda replaces the switch_to_selection_screen_callback function
        self.widget.scaler.fit(np.array([[100, 10, 0.5, 0.5], [150, 15, 0.6, 0.6]]))  # Example data to fit the scaler

        # Preparing a mock model that will be used in the prediction test
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([[1.0, 2.0]])
        self.widget.models['SPY_Call'] = self.mock_model

    @patch('pandas.read_csv')
    def test_predictOptionPrice(self, mock_read_csv):
        # Setting up the DataFrame that the mocked read_csv will return
        mock_read_csv.return_value = pd.DataFrame({
            'Close': [150.0]
        })

        # Set the  input values for the test
        self.widget.strikePriceInput.setText('100')
        self.widget.expirationTimeInput.setText('30')

        # Running  the prediction
        self.widget.predictOptionPrice('Call')
        output = self.widget.bidAskOutput.toPlainText()

        # Assertions to check if the predicted values are displayed correctly
        self.assertIn('Predicted Bid Price: 1.0', output)
        self.assertIn('Predicted Ask Price: 2.0', output)

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()  # Ensure the QApplication is properly closed

if __name__ == '__main__':
    unittest.main()
