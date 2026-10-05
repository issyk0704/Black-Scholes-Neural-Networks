from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox
import os
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class NeuralNetworkWidget(QWidget):
    def __init__(self, switch_to_selection_screen_callback):
        super().__init__()
        self.switch_to_selection_screen_callback = switch_to_selection_screen_callback
        self.models = {}
        self.scaler = MinMaxScaler()
        self.initUI()
        self.loadDataAndTrainModels()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.fetchDataButton = QPushButton("Fetch Latest Data")
        self.fetchDataButton.clicked.connect(self.fetchData)
        self.layout.addWidget(self.fetchDataButton)

        self.fetchDataButton = QPushButton("Display Latest Data")
        self.fetchDataButton.clicked.connect(self.displayLatestData)
        self.layout.addWidget(self.fetchDataButton)

        self.tickerSelector = QComboBox(self)
        self.tickerSelector.addItems(["SPY", "QQQ", "AAPL", "NVDA"])
        self.layout.addWidget(QLabel("Select Ticker:"))
        self.layout.addWidget(self.tickerSelector)

        self.strikePriceInput = QLineEdit(self)
        self.riskFreeRateInput = QLineEdit(self)
        self.timeToExpirationInput = QLineEdit(self)
        self.impliedVolatilityInput = QLineEdit(self)
        self.setupInputFields()

        self.predictButton = QPushButton("Predict Option Prices", self)
        self.predictButton.clicked.connect(self.predictOptionPrices)
        self.layout.addWidget(self.predictButton)



        self.outputArea = QTextEdit()
        self.outputArea.setReadOnly(True)
        self.layout.addWidget(self.outputArea)

        self.returnButton = QPushButton("Return to Selection Screen")
        self.returnButton.clicked.connect(self.switch_to_selection_screen_callback)
        self.layout.addWidget(self.returnButton)

        self.setLayout(self.layout)

    def fetchData(self):
        self.outputArea.clear()
        tickers = ["SPY", "QQQ", "AAPL", "NVDA"]
        base_path = os.path.expanduser("~/Desktop/BS-GUI/Stock-Data")
        for ticker in tickers:
            data = yf.Ticker(ticker)
            df = data.history(period="2y")  # adjust period when need to - set to 2 year for now
            file_path = f"{base_path}/{ticker}_data.csv"
            df.to_csv(file_path)
            self.outputArea.append(f"Downloaded latest data for {ticker}")


    def setupInputFields(self):
        self.layout.addWidget(QLabel("Strike Price (K):"))
        self.layout.addWidget(self.strikePriceInput)
        self.layout.addWidget(QLabel("Risk-Free Rate (r, in decimal):"))
        self.layout.addWidget(self.riskFreeRateInput)
        self.layout.addWidget(QLabel("Time to Expiration (T in years, in decimal):"))
        self.layout.addWidget(self.timeToExpirationInput)
        self.layout.addWidget(QLabel("Implied Volatility (in decimal):"))
        self.layout.addWidget(self.impliedVolatilityInput)

    def displayLatestData(self):
        self.outputArea.clear()
        tickers = ["SPY", "QQQ", "AAPL", "NVDA"]
        base_path = os.path.expanduser("~/Desktop/BS-GUI/Stock-Data")
        for ticker in tickers:
            file_path = f"{base_path}/{ticker}_data.csv"
            df = pd.read_csv(file_path)
            latest_data = df.iloc[-1]
            latest_close_price = latest_data['Close']
            latest_date = latest_data.get('Date', "Unknown Date")
            self.outputArea.append(f"{ticker} - Latest Close Price on {latest_date}: ${latest_close_price:.2f}")

    def create_nn_model(self):
        model = Sequential([
            Dense(100, input_dim=5, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(2, activation='linear')  # Predicting
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def loadDataAndTrainModels(self):
        tickers = ["SPY", "QQQ", "AAPL", "NVDA"]
        base_path = os.path.expanduser("~/Desktop/BS-GUI/Stock-Data")
        for ticker in tickers:
            file_path = f"{base_path}/{ticker}_data.csv"
            df = pd.read_csv(file_path)
            X = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            y = df['Close'].values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            model = self.create_nn_model()
            model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)
            self.models[ticker] = model

    def black_scholes(self, S, K, T, r, sigma):
        """Calculate Call and Put prices using the Black-Scholes formula."""
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return call_price, put_price

    def predictOptionPrices(self):
        ticker = self.tickerSelector.currentText()
        K = float(self.strikePriceInput.text())
        T = float(self.timeToExpirationInput.text())
        r = float(self.riskFreeRateInput.text())
        sigma = float(self.impliedVolatilityInput.text())

        if ticker in self.models:
            base_path = os.path.expanduser("~/Desktop/BS-GUI/Stock-Data")
            file_path = f"{base_path}/{ticker}_data.csv"
            df = pd.read_csv(file_path)

            # Use the latest actual close price as S in the Black-Scholes formula
            latest_close_price = df.iloc[-1]['Close']

            # Calculate call and put prices using the Black-Scholes formula
            call_price, put_price = self.black_scholes(latest_close_price, K, T, r, sigma)

            # Display the information in the GUI
            self.outputArea.append(f"Ticker: {ticker}")
            self.outputArea.append(f"Latest Close Price: {latest_close_price:.2f}")
            self.outputArea.append(f"Calculated Call Price: {call_price:.2f}")
            self.outputArea.append(f"Calculated Put Price: {put_price:.2f}")
        else:
            self.outputArea.append(f"No model available for {ticker}.")