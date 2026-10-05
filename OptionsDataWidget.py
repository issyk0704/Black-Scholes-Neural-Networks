import os
import numpy as np
import pandas as pd
import yfinance as yf
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit, QLineEdit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class OptionsDataWidget(QWidget):
    def __init__(self, switch_to_selection_screen_callback):
        super().__init__()
        self.switch_to_selection_screen_callback = switch_to_selection_screen_callback
        self.scaler = MinMaxScaler()
        self.models = {}  # Store models for each ticker
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.tickerSelector = QComboBox()
        self.tickerSelector.addItems(["SPY", "QQQ", "AAPL", "NVDA"])
        self.layout.addWidget(QLabel("Select Ticker:"))
        self.layout.addWidget(self.tickerSelector)

        self.fetchOptionsButton = QPushButton("Fetch and Save Options Data")
        self.fetchOptionsButton.clicked.connect(self.fetchAndSaveOptionsData)
        self.layout.addWidget(self.fetchOptionsButton)

        self.trainCallOptionsButton = QPushButton("Train Call Option Contracts")
        self.trainCallOptionsButton.clicked.connect(
            lambda: self.prepareAndTrainModel(self.tickerSelector.currentText(), 'Call'))
        self.layout.addWidget(self.trainCallOptionsButton)

        self.trainPutOptionsButton = QPushButton("Train Put Option Contracts")
        self.trainPutOptionsButton.clicked.connect(
            lambda: self.prepareAndTrainModel(self.tickerSelector.currentText(), 'Put'))
        self.layout.addWidget(self.trainPutOptionsButton)

        # Input for Strike Price
        self.strikePriceInput = QLineEdit(self)
        self.layout.addWidget(QLabel("Strike Price:"))
        self.layout.addWidget(self.strikePriceInput)

        # Input for Time to Expiration in Days
        self.expirationTimeInput = QLineEdit(self)
        self.layout.addWidget(QLabel("Time to Expiration (Days):"))
        self.layout.addWidget(self.expirationTimeInput)

        # Button for predicting Call option price
        self.predictCallButton = QPushButton("Predict Call Option Price")
        self.predictCallButton.clicked.connect(lambda: self.predictOptionPrice('Call'))
        self.layout.addWidget(self.predictCallButton)

        # Button for predicting Put option price
        self.predictPutButton = QPushButton("Predict Put Option Price")
        self.predictPutButton.clicked.connect(lambda: self.predictOptionPrice('Put'))
        self.layout.addWidget(self.predictPutButton)

        # Output for displaying Bid and Ask prices
        self.bidAskOutput = QTextEdit()
        self.bidAskOutput.setReadOnly(True)
        self.layout.addWidget(self.bidAskOutput)

        self.optionsDataOutput = QTextEdit()
        self.optionsDataOutput.setReadOnly(True)
        self.layout.addWidget(self.optionsDataOutput)

        self.returnButton = QPushButton("Return to Selection Screen")
        self.returnButton.clicked.connect(self.switch_to_selection_screen_callback)
        self.layout.addWidget(self.returnButton)

        self.setLayout(self.layout)

    def fetchAndSaveOptionsData(self):
        ticker = self.tickerSelector.currentText()
        data = yf.Ticker(ticker)
        options_dates = data.options

        all_options_data = []

        for date in options_dates:
            options_chain = data.option_chain(date)
            calls = options_chain.calls.assign(OptionType='Call', ExpiryDate=date)
            puts = options_chain.puts.assign(OptionType='Put', ExpiryDate=date)
            all_options_data.extend([calls, puts])

        if all_options_data:
            options_data_df = pd.concat(all_options_data)
            base_path = os.path.expanduser("~/Desktop/BS-GUI/Options-Data")
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            file_path = f"{base_path}/{ticker}_options_data.csv"
            options_data_df.to_csv(file_path, index=False)
            self.optionsDataOutput.setText(f"Options data for {ticker} saved to {file_path}")
        else:
            self.optionsDataOutput.setText("No options data available.")

    def prepareData(self, df):
        features = df[['strike', 'volume', 'openInterest', 'impliedVolatility']].values
        target = df['lastPrice'].values
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled, target

    def createModel(self):
        model = Sequential([
            Dense(100, input_dim=4, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(2, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def prepareAndTrainModel(self, ticker, option_type):
        base_path = os.path.expanduser("~/Desktop/BS-GUI/Options-Data")
        file_path = f"{base_path}/{ticker}_options_data.csv"

        if not os.path.exists(file_path):
            self.optionsDataOutput.setText(f"No options data found for {ticker}.")
            return

        df = pd.read_csv(file_path)
        df = df[df['OptionType'] == option_type]

        if df.empty:
            self.optionsDataOutput.setText(f"No {option_type} options data available for {ticker}.")
            return

        df.dropna(inplace=True)
        X, y = self.prepareData(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.createModel()
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)
        self.models[f"{ticker}_{option_type}"] = model
        self.optionsDataOutput.setText(f"Model trained on {option_type} options data for {ticker}.")

    def predictOptionPrice(self, option_type):
        try:
            # Validate and convert user inputs
            strike_price = float(self.strikePriceInput.text())
            time_to_expiration_days = float(self.expirationTimeInput.text())

            # Including dummy values for the missing features to match the model's expectations
            dummy_feature_1 = 0.5  # Example placeholder value
            dummy_feature_2 = 0.5  # Example placeholder value

            # Construct the feature array with the correct number of features
            features = np.array([[strike_price, time_to_expiration_days, dummy_feature_1, dummy_feature_2]])
            features_scaled = self.scaler.transform(features)

            # Select the model based on option type and ticker
            ticker = self.tickerSelector.currentText()
            model_key = f"{ticker}_{option_type}"
            if model_key not in self.models:
                self.bidAskOutput.setText("Model not found for the selected option type and ticker.")
                return

            model = self.models[model_key]
            predicted_prices = model.predict(features_scaled)

            # Assuming the model outputs bid and ask prices directly

            predicted_bid = predicted_prices[0][0]
            predicted_ask = predicted_prices[0][1]

            self.bidAskOutput.setText(f"Predicted Bid Price: {predicted_bid}\nPredicted Ask Price: {predicted_ask}")
        except ValueError as e:
            self.bidAskOutput.setText(f"Error in input values: {str(e)}")
        except Exception as e:
            self.bidAskOutput.setText(f"Error in prediction: {str(e)}")

