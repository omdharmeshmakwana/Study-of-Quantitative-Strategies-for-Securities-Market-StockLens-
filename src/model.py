import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from data_loader import load_data  # Make sure `data_loader.py` is in the correct directory

# Function to create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Load and preprocess data
df, scaler = load_data("../data/stock_data.csv")  # Ensure the correct path
data = df['Close'].values.reshape(-1, 1)

# Create sequences
X, y = create_sequences(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("Starting model training...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
print("Model training complete.")

# Ensure the model directory exists
model_dir = "../models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model
model_path = os.path.join(model_dir, "trained_model.h5")
print("Saving model...")
model.save(model_path)
print(f"Model saved successfully at {model_path}")
