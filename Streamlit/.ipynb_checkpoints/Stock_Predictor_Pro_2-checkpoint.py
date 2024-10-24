import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import date, timedelta
import requests
import numpy as np

# Set project name and layout
st.set_page_config(page_title="Futuro Stock App", layout="wide")

# Load stock data (with caching)
@st.cache_data
def load_data(symbol):
    stock_data = yf.download(symbol, start='2020-01-01', end='2024-01-01')
    stock_data.reset_index(inplace=True)
    return stock_data

# Sidebar navigation
selected_menu = st.sidebar.selectbox("Select a Page", ["Home", "Visualization", "Prediction", "Trade News", "Prediction App"], index=0)

# Stock selection
stock_symbol = st.sidebar.selectbox("Choose Stock Symbol", options=["AAPL", "GOOGL", "AMZN", "MSFT"])

# Load stock data globally
stock_data = load_data(stock_symbol)

# 1. Home Page
if selected_menu == "Home":
    st.title("Welcome to Futuro Stock App!")
    st.write("This app provides stock analysis and predictions for various companies.")

# 2. Stock Data Visualization
if selected_menu == "Visualization":
    st.title(f"{stock_symbol} Stock Data Visualization")
    
    # Adjusted Close Price Over Time
    st.subheader("Adjusted Close Price Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Adj Close'], mode='lines', name='Adj Close'))
    fig.update_layout(title=f"{stock_symbol} Adjusted Close Price Over Time",
                      xaxis_title="Date", yaxis_title="Adjusted Close Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # Volume Over Time
    st.subheader("Trading Volume Over Time")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stock_data['Date'], y=stock_data['Volume'], name='Volume'))
    fig.update_layout(title=f"{stock_symbol} Trading Volume Over Time",
                      xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig, use_container_width=True)

# 3. Stock Price Prediction
if selected_menu == "Prediction":
    st.title(f"{stock_symbol} Stock Price Prediction")

    # Prepare the data
    X = stock_data[['Open', 'High', 'Low', 'Volume']]
    y = stock_data['Adj Close']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Interactive Plotly Visualization of predictions
    st.subheader("Prediction vs Real Data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'].iloc[-len(y_test):], y=y_test, mode='lines', name='Real Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=stock_data['Date'].iloc[-len(y_test):], y=y_pred, mode='lines', name='Predicted Data', line=dict(color='red')))
    fig.update_layout(title=f"{stock_symbol} Stock Price: Real vs Predicted",
                      xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

# 4. Trade News Section (Interactive)
if selected_menu == "Trade News":
    st.title(f"Trade News for {stock_symbol} Inc.")
    
    st.write("## Latest Trade-Related News")

    # Placeholder for live trade-related news articles
    news_articles = [
        {"title": "Apple Stock Hits All-Time High", "url": "https://www.apple.com", "date": "2023-10-17"},
        {"title": "Apple Announces Stock Buyback", "url": "https://www.apple.com", "date": "2023-10-16"},
        {"title": "Google Surpasses Expectations in Earnings", "url": "https://www.google.com", "date": "2023-10-15"},
        {"title": "Amazon Acquires New AI Stock", "url": "https://www.amazon.com", "date": "2023-10-14"},
        {"title": "Microsoft Releases New Stock Options", "url": "https://www.microsoft.com", "date": "2023-10-13"},
        {"title": "Tesla Sees Huge Jump in Stock After AI Investment", "url": "https://www.tesla.com", "date": "2023-10-12"}
    ]

    # Displaying the trade-related news with clickable links
    for article in news_articles:
        st.markdown(f"**[{article['title']}]({article['url']})** \nPublished on: {article['date']}")

# 5. Stock Prediction App (Improved Date Handling)
if selected_menu == "Prediction App":
    st.title(f"Futuro {stock_symbol} Stock Prediction App")
    
    # Input for future date
    future_date = st.date_input("Select a future date for prediction:", value=date.today() + timedelta(days=30), min_value=date.today())

    # Calculate days into the future (fixed date type error)
    future_days = (future_date - stock_data['Date'].max().date()).days
    
    # Ensure the model is trained
    if 'model' not in st.session_state:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.session_state['model'] = model
    else:
        model = st.session_state['model']
    
    # Predict future price (adjust the prediction logic)
    if future_days > 0:
        trend = np.mean(np.diff(stock_data['Adj Close'][-5:]))  # Get recent trend
        future_price = stock_data['Adj Close'].iloc[-1] + (trend * future_days)
        st.write(f"Predicted Price for {stock_symbol} on {future_date}: **${future_price:.2f}**")
    else:
        st.write("Please select a valid future date for prediction.")
    
    # Display chart showing historical and predicted data
    st.subheader(f"Price Trend with Predicted Data for {future_date}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Adj Close'], mode='lines', name='Real Data'))
    fig.add_trace(go.Scatter(x=[future_date], y=[future_price], mode='markers', marker=dict(size=10, color='red'), name='Predicted Price'))
    fig.update_layout(title=f"{stock_symbol} Real Data and Predicted Price for {future_date}",
                      xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)
