
# Apple Stock Performance & Sentiment Analysis
![Apple Stock and Sentiment Analysis](https://github.com/Futuroent/Stock_Market_2024/blob/main/captrader_markttechnik_trade_3.jpeg)


## Project Overview

This project analyzes the performance of **Apple's stock (AAPL)** over a specified period and compares it with sentiment analysis from news articles. The aim is to gain visual insights into Apple's stock performance and investigate how news sentiment impacts stock movements. The results are visualized using **Tableau**, creating interactive dashboards that can be useful for investors and market analysts.

## Motivation

The idea behind this project is to understand the relationship between **news sentiment** and **stock movements**. For large tech companies like Apple, media coverage can have a significant influence on stock performance. By linking news articles with sentiment analysis and corresponding stock price movements, trends and potential influencing factors can be identified.

## Features

- **Data Sources**:
  - Historical Apple stock prices from [Yahoo Finance](https://finance.yahoo.com).
  - News articles about Apple from various sources via the **News API**.
  - Sentiment analysis of the articles using a financial sentiment model.

- **Tableau Dashboards**:
  - **Stock Price Over Time**: A dashboard showing stock price trends over various time periods (months, quarters, years).
  - **Sentiment Analysis Impact**: An overview of news and their sentiment scores, compared with corresponding stock prices.
  - **Volume vs. Price**: A correlation between trading volume and stock prices to analyze potential volatility.
  - **Real-Time Updates**: The system can collect real-time data on Apple stock and sentiment updates hourly and visualize it in a separate dashboard.

## Technologies Used

- **Programming Language**: Python
- **Data Sources**: 
  - **Yahoo Finance API** for historical stock prices.
  - **News API** for gathering news articles.
  - **Hugging Face Transformers** for sentiment analysis.
- **Database**: SQLite for storing sentiment and stock data.
- **Visualization Tool**: Tableau

## Setup & Installation

### 1. Requirements

To run this project, you need the following dependencies:
- **Python 3.x**
- **Tableau Public** (free desktop tool for visualization)
- **Pandas**: For data manipulation
- **SQLite**: For data storage
- **Yahoo Finance API**: To collect stock prices
- **News API**: To fetch news articles
- **Hugging Face Transformers**: For sentiment analysis

You can install the required Python libraries using the following command:

```bash
pip install pandas yfinance sqlite3 requests transformers
```

### 2. Project Installation

1. Clone this repository to your local machine:

   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:

   ```bash
   cd apple-stock-sentiment-analysis
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up the SQLite database:

   ```bash
   python setup_database.py
   ```

5. Run the Python scripts to collect stock data and perform sentiment analysis:

   ```bash
   python collect_stock_data.py
   python sentiment_analysis.py
   ```

6. Open the Tableau files (`*.twbx`) to visualize the dashboards and explore the data interactively.

## Data Sources

- **Apple Stock Data**: Stock prices are retrieved using the Yahoo Finance API. Key fields include "Open", "Close", "High", "Low", and "Volume".
- **Sentiment Data**: News articles about Apple are gathered using the News API, and a sentiment model classifies the sentiment as positive, negative, or neutral.

## Tableau Dashboards

### 1. **Stock Price Over Time**
   - **Data**: AAPL stock data (Adj Close)
   - **X-Axis**: Time axis (Years, Quarters, Months)
   - **Y-Axis**: Adjusted closing price (Adj Close)
   - **Description**: This dashboard visualizes Apple’s stock price performance over time, highlighting trends and important points.

### 2. **Sentiment Analysis Impact**
   - **Data**: Sentiment data and AAPL stock data
   - **X-Axis**: Date
   - **Y-Axis**: Stock price (Adj Close)
   - **Description**: This dashboard compares sentiment from news articles with stock price movements, showing how positive, negative, or neutral news impacted the stock price.

### 3. **Volume vs. Price**
   - **Data**: AAPL stock data
   - **X-Axis**: Trading volume
   - **Y-Axis**: Stock price (Adj Close)
   - **Description**: This dashboard visualizes the correlation between trading volume and stock price. It helps identify periods of high volume and analyze their impact on price movements.

### 4. **Real-Time Updates**
   - **Data**: Real-time data from Yahoo Finance and sentiment analysis
   - **Description**: This dashboard updates hourly and displays the latest stock movements and sentiment analysis. It’s particularly useful for spotting short-term trends.

## Future Developments

In future iterations, the project could be expanded to include:
- **API Expansion**: Adding more news sources and stock data to enable broader market analysis.
- **Prediction Models**: Implementing machine learning models to predict stock movements based on sentiment.
- **Enhanced Dashboards**: Improving interactivity and visualizations.

## Conclusion

This project provides a comprehensive analysis of Apple’s stock performance in relation to news sentiment. It highlights both long-term trends and real-time data, allowing users to gain deeper insights into the factors driving stock prices.
