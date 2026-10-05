Black-Scholes and Neural Networks for Financial Derivatives Pricing

This project focuses on pricing financial derivatives using both traditional methods, such as the Black-Scholes formula, and modern machine learning techniques, specifically neural networks. The application provides a user-friendly GUI that bridges the gap between theoretical finance and practical application, enabling users to perform financial analysis efficiently.
Table of Contents

Introduction
Features
Project Structure
Installation
Usage
Technologies Used
Future Enhancements
License
Introduction

The application is designed to make advanced financial analysis accessible to financial professionals, students, and traders. By integrating numerical solutions for the Black-Scholes equation and neural networks for market predictions, the tool provides robust pricing and analytical capabilities.
Features

Black-Scholes Widget: Computes option prices using the Black-Scholes formula. Includes visualization tools for market trends like volatility and moving averages.
Neural Network Widget: Trains neural networks on historical stock data to predict option prices and stock behaviors.
Options Data Widget: Fetches real-time options data, trains models on Call and Put contracts, and predicts bid and ask prices.
Data Fetcher Class: Dynamically fetches real-time stock and options data from Yahoo Finance APIs.
GUI Navigation: Simplified navigation screen for accessing different functionalities.
Project Structure

├── BlackScholesWidget.py         # Handles Black-Scholes option pricing
├── NeuralNetworkWidget.py        # Implements neural network functionality
├── OptionsDataWidget.py          # Manages options data and predictions
├── DataFetcher.py                # Fetches real-time financial data
├── main.py                       # Entry point for the GUI application
├── Testing/                      # Unit tests for all modules
├── README.md                     # Project documentation
Installation

Clone the repository:
git clone https://github.com/issyk0704/Black-Scholes-Neural-Networks.git
Navigate to the project directory:
cd Black-Scholes-Neural-Networks
Create and activate a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt

Usage

Run the application:
python main.py

Navigate through the GUI:
Select widgets for Black-Scholes calculations, neural network training, or options data analysis.
Input required parameters and visualize results in real-time.

Testing:
Run unit tests using:
python -m unittest discover -s Testing
Technologies Used


Programming Language: Python

Libraries:

PyQt5: For GUI development
TensorFlow/Keras: For neural network implementation
yfinance: For fetching real-time stock and options data
Pandas & NumPy: For data manipulation and processing

Future Enhancements

Real-Time Data Integration: Extend capabilities for live market data analysis.
Expanded Financial Instruments: Add support for bonds, commodities, and cryptocurrencies.
Enhanced Risk Management Tools: Include advanced analytics for portfolio risk assessment.
User Experience Improvements: Optimize the GUI for better accessibility and customization.
License

This project is licensed under the MIT License. See the LICENSE file for details.
