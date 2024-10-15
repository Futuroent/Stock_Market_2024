import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def download_stock_data(ticker, start_date, end_date):
    # Download stock data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def plot_stock_trend(ticker, stock_data):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'], label=f'{ticker} Closing Price')
    plt.title(f'{ticker} Stock Price Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_sector_average(stocks, stock_data):
    closing_prices = pd.DataFrame()
    for stock in stocks:
        closing_prices[stock] = stock_data[stock]['Close']
    return closing_prices.mean(axis=1)  # Calculate the average closing price across the sector
