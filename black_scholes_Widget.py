# Final Year Project  - Ismail Khan, 21009142

# Black Scholes User Interface

# 1. Import libraries for manipulating data and for the GUI
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QTextEdit, QFormLayout, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import yfinance as yf  # Importing Yahoo finance for user input instrument data
import numpy as np
from scipy.stats import norm


# 2. Matplotlib class - for the matplotlib canvas to enable plotting within application
class MplCanvas(FigureCanvas):
    #initialisation of matplotlib class
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)  # Creating a figure for plotting
        self.axes = fig.add_subplot(111)  # Adding a subplot to the figure
        super().__init__(fig)  # Initialising the parent class with the figure


# 3. Main application class for the GUI
class BlackScholesWidget(QWidget):
    # 3.1 Initialisation of application class
    def __init__(self, switch_to_selection_screen_callback):
        super().__init__()
        self.switch_to_selection_screen_callback = switch_to_selection_screen_callback
        self.setWindowTitle('Black Scholes Simulator')  # Set the window title
        self.setGeometry(150, 150, 900, 700)  # Set the window size and position
        self.initUI()  # Initialising the UI
        self.last_closing_price = None # stores the last closing price
        self.currentTicker = None


        # 3.2 Method to set up the user interface
    def initUI(self):
        layout = QVBoxLayout()  # Main layout for the widget

        # User input for the stock ticker symbol
        self.tickerButtons = {
            "SPY": QPushButton('Fetch SPY', self),
            "QQQ": QPushButton('Fetch QQQ', self),
            "AAPL": QPushButton('Fetch AAPL', self),
            "NVDA": QPushButton('Fetch NVDA', self),
        }

        # Connect each button to fetchData method with the ticker as a parameter
        for ticker, button in self.tickerButtons.items():
            button.clicked.connect(lambda _, t=ticker: self.fetchData(t))
            layout.addWidget(button)

        # User input for Black Scholes parameters
        formLayout = QFormLayout()
        self.strikePrice = QLineEdit(self)
        formLayout.addRow("Strike Price (K):", self.strikePrice)
        self.riskFreeRate = QLineEdit(self)
        formLayout.addRow("Risk-Free Interest Rate (r, in decimal):", self.riskFreeRate)
        self.timeToMaturity = QLineEdit(self)
        formLayout.addRow("Time to Maturity (T, in years, in decimal):", self.timeToMaturity)
        self.volatility = QLineEdit(self)
        formLayout.addRow("Volatility (σ, in decimal):", self.volatility)
        layout.addLayout(formLayout)  # Add form layout to main layout

        # Calculate Option Price button
        self.calculateButton = QPushButton('Calculate Option Price', self)
        self.calculateButton.clicked.connect(self.calculateOptionPrice)  # Connect button to calculateOptionPrice method
        layout.addWidget(self.calculateButton)

        self.returnButton = QPushButton("Return to Selection Screen", self)
        self.returnButton.clicked.connect(self.switch_to_selection_screen_callback)
        layout.addWidget(self.returnButton)

        # Dropdown box for selecting graph types
        self.graphSelector = QComboBox(self)
        self.graphSelector.addItems(["Closing Price Trend", "Opening Price Trend", "Moving Average", "Volatility"])
        layout.addWidget(self.graphSelector)
        self.graphSelector.currentIndexChanged.connect(
            self.updateGraph)  # Connect selection change to updateGraph method

        # Output for displaying fetched data and the calculations
        self.dataOutput = QTextEdit(self)
        self.dataOutput.setReadOnly(True)
        layout.addWidget(self.dataOutput)

        self.optionsOutput = QTextEdit(self)
        self.optionsOutput.setReadOnly(True)
        layout.addWidget(self.optionsOutput)

        # Matplotlib canvas for plotting stock data
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        self.setLayout(layout)  # Setting the main layout for the widget

    # 3.3 Method to fetch stock data and update the UI
    def fetchData(self, ticker_symbol):
        self.currentTicker = ticker_symbol
        data = yf.Ticker(ticker_symbol).history(period="2y")  # Fetching 2 year of data on inputted stock ticker

        if not data.empty:  # Check if data is fetched successfully
            self.last_closing_price = data['Close'].iloc[-1]  # Stores the latest closing price
            summary_text = f"Last Closing Price: {self.last_closing_price:.2f}\n"
            self.dataOutput.setText(summary_text)  # Display summary in the dataOutput widget
            self.calculateButton.setEnabled(True)

            # Plotting the closing price trend on chart
            self.canvas.axes.clear()
            self.canvas.axes.plot(data.index, data['Close'], label='Closing Price')
            self.canvas.axes.set_title(f'Closing Price Trend for {ticker_symbol}')
            self.canvas.axes.legend()
            self.canvas.draw()
        else:
            self.dataOutput.setText("Failed to fetch data or data is empty.")
            self.last_closing_price = None
            self.calculateButton.setEnabled(False)

    # 3.4 Method to calculate and display the Black Scholes options prices

    def calculateOptionPrice(self):
        print(f"Attempting to calculate option price with last closing price: {self.last_closing_price}")
        if self.last_closing_price is not None:
            try:
                S = self.last_closing_price  # Using the latest stored closing price
                K = float(self.strikePrice.text())  # Strike price (K)
                T = float(self.timeToMaturity.text())  # Time to maturity (T)
                r = float(self.riskFreeRate.text())  # Risk-free interest rate (r)
                sigma = float(self.volatility.text())  # Volatility (sigma)

                # Calculate call and put option prices
                option_price_call = self.blackScholes(S, K, T, r, sigma, option_type="call")
                option_price_put = self.blackScholes(S, K, T, r, sigma, option_type="put")

                # Calculate Greeks for call options (example)
                delta_call = self.delta_call(S, K, T, r, sigma)
                vega_call = self.vega_call(S, K, T, r, sigma)

                # Calculate Greeks for put options
                delta_put = self.delta_put(S, K, T, r, sigma)
                vega_put = self.vega_put(S, K, T, r, sigma)

                # Prepare the results to display
                options_text = (f"Call Option Price: {option_price_call:.2f}\n"
                                f"Put Option Price: {option_price_put:.2f}\n\n"
                                f"Call Delta: {delta_call:.2f}, Call Vega: {vega_call:.2f}\n"
                                f"Put Delta: {delta_put:.2f}, Put Vega: {vega_put:.2f}")

                # Display in the dedicated QTextEdit for options and Greeks
                self.optionsOutput.setText(options_text)

            except ValueError as e:
                self.optionsOutput.setText(f"Error calculating option price: {e}")
        else:
                self.optionsOutput.setText("Last closing price not available. \n Fetch data first.")

    # 3.5 Black-Scholes formula
    def blackScholes(self, S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":  # call
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    #3.6  delta calls
    def delta_call(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)

    # 3.7 delta puts
    def delta_put(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return -norm.cdf(-d1)   # Delta put formula

    # 3.8  vega calls
    def vega_call(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    # 3.9 Correctly calling vega_call from vega_put
    def vega_put(self, S, K, T, r, sigma):
        return self.vega_call(S, K, T, r, sigma)

    # 3.10 Method to update graph based on user input on dropdown
    def updateGraph(self):
        selection = self.graphSelector.currentText()
        ticker_symbol = self.currentTicker
        # Calling the appropriate method based on the selected graph
        if selection == "Closing Price Trend":
            self.plotClosingPriceTrend(ticker_symbol)
        elif selection == "Opening Price Trend":
            self.plotOpeningPriceTrend(ticker_symbol)
        elif selection == "Moving Average":
            self.plotMovingAverage(ticker_symbol)
        elif selection == "Volatility":
            self.plotVolatility(ticker_symbol)

    # 3.11 plotting closing price trend graph
    def plotClosingPriceTrend(self, ticker_symbol):
        data = yf.Ticker(ticker_symbol).history(period="2y")
        if not data.empty:
            self.canvas.axes.clear()
            self.canvas.axes.plot(data.index, data['Close'], label='Closing Price', color='blue')
            self.canvas.axes.set_title(f'Closing Price Trend for {ticker_symbol}')
            self.canvas.axes.set_xlabel('Date')
            self.canvas.axes.set_ylabel('Price')
            self.canvas.axes.legend()
            self.canvas.draw()

    # 3.12 plotting opening price trend graph
    def plotOpeningPriceTrend(self, ticker_symbol):
        data = yf.Ticker(ticker_symbol).history(period="2y")
        if not data.empty:
            self.canvas.axes.clear()
            self.canvas.axes.plot(data.index, data['Open'], label='Opening Price', color='green')
            self.canvas.axes.set_title(f'Opening Price Trend for {ticker_symbol}')
            self.canvas.axes.set_xlabel('Date')
            self.canvas.axes.set_ylabel('Price')
            self.canvas.axes.legend()
            self.canvas.draw()

    # 3.13 plotting moving average lines on closing price
    def plotMovingAverage(self, ticker_symbol):
        data = yf.Ticker(ticker_symbol).history(period="2y")
        if not data.empty:
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()

            self.canvas.axes.clear()
            self.canvas.axes.plot(data.index, data['Close'], label='Closing Price', color='blue')
            self.canvas.axes.plot(data.index, data['MA50'], label='50-Day MA', color='orange')
            self.canvas.axes.plot(data.index, data['MA200'], label='200-Day MA', color='red')
            self.canvas.axes.set_title(f'Moving Averages for {ticker_symbol}')
            self.canvas.axes.set_xlabel('Date')
            self.canvas.axes.set_ylabel('Price')
            self.canvas.axes.legend()
            self.canvas.draw()

    # 3.14 plotting volatility graph
    def plotVolatility(self, ticker_symbol):
        data = yf.Ticker(ticker_symbol).history(period="2y")
        if not data.empty:
            data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Log Returns'].rolling(window=60).std() * np.sqrt(
                252)  # Annualizing, 252 trading days

            self.canvas.axes.clear()
            self.canvas.axes.plot(data.index, data['Volatility'], label='Historical Volatility', color='purple')
            self.canvas.axes.set_title(f'Historical Volatility for {ticker_symbol}')
            self.canvas.axes.set_xlabel('Date')
            self.canvas.axes.set_ylabel('Volatility')
            self.canvas.axes.legend()
            self.canvas.draw()


# 4.  Main function to run the UI  pyqt5 application
def main():
    app = QApplication(sys.argv)  # create an instance of QApplication
    window = BlackScholesWidget()  # creates instance of the Financial app
    window.show()
    sys.exit(app.exec_())  # start application event loop

# 4.1 conditional statement to run the main function
if __name__ == '__main__':
    main()  # calling main function to run application
