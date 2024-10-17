import streamlit as st
import joblib
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Load the trained Ridge model and the correct scaler
ridge_model_path = '../models/mlp_stock_model.pkl'  # Update the path if necessary
scaler_path = '../models/standard_scaler.pkl'  # Ensure this path points to the correct scaler file

# Load the Ridge model and stock scaler
ridge_model = joblib.load(ridge_model_path)

# If your stock_scaler was trained for 4 features, load it (adjust accordingly)
stock_scaler = joblib.load(scaler_path)

# Function to make stock price predictions using Ridge
def predict_stock_ridge(features):
    # Scale the input features (only use 4 features, the ones expected by the model)
    features_to_use = features[:4]  # Assuming the first 4 features are used by Ridge
    features_scaled = stock_scaler.transform([features_to_use])

    # Predict using the Ridge model
    prediction = ridge_model.predict(features_scaled)

    # Return the predicted value
    return prediction[0]

# Streamlit app layout
st.title('📈 Stock Price Predictor Pro (Ridge Model)')

st.write("""
### Predict the stock closing price using the trained Ridge regression model.
Enter the relevant stock market data below to get predictions:
""")

# Date selection for stock prediction
date = st.date_input("Select the date for prediction", datetime.today())
st.write(f"Selected Date: {date.strftime('%Y-%m-%d')}")

# Input fields for stock features
daily_return = st.slider('Daily Return', min_value=-1.0, max_value=1.0, step=0.01, value=0.0)
moving_avg_20 = st.slider('20-Day Moving Average', min_value=0.0, max_value=1000.0, step=1.0, value=500.0)
moving_avg_50 = st.slider('50-Day Moving Average', min_value=0.0, max_value=1000.0, step=1.0, value=500.0)
volume = st.number_input('Volume', min_value=0, value=1000000, step=10000)
ema_50 = st.slider('EMA 50', min_value=0.0, max_value=1000.0, step=1.0, value=500.0)
rsi = st.slider('RSI (Relative Strength Index)', min_value=0.0, max_value=100.0, step=1.0, value=50.0)
bollinger_high = st.slider('Bollinger High', min_value=0.0, max_value=1000.0, step=1.0, value=500.0)
bollinger_low = st.slider('Bollinger Low', min_value=0.0, max_value=1000.0, step=1.0, value=500.0)

# Collect all input features
input_features = [daily_return, moving_avg_20, moving_avg_50, volume, ema_50, rsi, bollinger_high, bollinger_low]

# Add some space
st.write("---")

# Predict button with interactivity
if st.button('📊 Predict Stock Price'):
    prediction = predict_stock_ridge(input_features)
    st.success(f"Predicted Stock Closing Price on {date.strftime('%Y-%m-%d')}: ${prediction:.2f}")
    
    # Display input features
    st.write(f"### Summary of input features:")
    st.write({
        'Daily Return': daily_return,
        '20-Day MA': moving_avg_20,
        '50-Day MA': moving_avg_50,
        'Volume': volume,
        'EMA 50': ema_50,
        'RSI': rsi,
        'Bollinger High': bollinger_high,
        'Bollinger Low': bollinger_low
    })
else:
    st.info("Please fill in the stock features and click 'Predict Stock Price'")

# Adding a footer
st.write("""
#### Disclaimer:
This is a demo app for educational purposes only and should not be used for actual financial decision-making.
""")
