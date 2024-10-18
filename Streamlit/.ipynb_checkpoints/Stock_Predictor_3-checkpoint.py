import streamlit as st
import joblib
import numpy as np
import torch
from torch import nn
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


# Define the LSTM-based model architecture
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        # LSTM layer with input size of 1 and hidden size of 50
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        # Fully connected layer with input size of 50 and output size of 1
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        # Use the output from the last LSTM time step
        lstm_out = lstm_out[:, -1, :]  # Taking the last time step output
        # Pass through the fully connected layer
        out = self.fc(lstm_out)
        return out

# Load model and scaler
ridge_model_path = '../models/lstm_stock_model2.pth'  
scaler_path = '../models/scaler.pkl'  

# Load the trained model's state_dict and the scaler
ridge_model = LSTMModel()
ridge_model.load_state_dict(torch.load(ridge_model_path))  
ridge_model.eval() 

# Load the scaler for input data
stock_scaler = joblib.load(scaler_path)

# Function to make stock price predictions using the model
def predict_stock_ridge(features):
    # Only use the first feature (as the model is designed for input_size=1)
    features_to_use = features
    features_scaled = stock_scaler.transform([features_to_use])

    # Convert features to a tensor for PyTorch, adding batch and sequence dimensions
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(-1)

    # Make prediction using the model
    with torch.no_grad():
        prediction = ridge_model(features_tensor)

    # Return the predicted value
    return prediction.item()  # Convert the tensor to a scalar

# Streamlit app layout

# Define the stock prediction function
def predict_stock_ridge(input_features):
    # Dummy function for prediction, replace this with your actual model
    return sum(input_features) * 1.5  # Just a placeholder calculation

def main():
    st.title('📈 Stock Price Predictor Pro (LSTM Model)')

    st.write("""
    ### Predict the stock closing price using the trained LSTM model.
    Enter the relevant stock market data below to get a prediction:
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

    # Add a horizontal line for separation
    st.write("---")

    # Button for making predictions
    if st.button('📊 Predict Stock Price'):
        # Make prediction using the function
        prediction = predict_stock_ridge(input_features)
        st.success(f"Predicted Stock Closing Price on {date.strftime('%Y-%m-%d')}: ${prediction:.2f}")

        # Display summary of the input features
        st.write("### Summary of Input Features:")
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

    # Footer Disclaimer
    st.write("""
    #### Disclaimer:
    This is a demo app for educational purposes only and should not be used for actual financial decision-making.
    """)

# Run the main function
if __name__ == '__main__':
    main()
