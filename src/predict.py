import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_data

def predict_next_day(model_path, file_path):
    model = tf.keras.models.load_model(model_path)
    df, scaler = load_data(file_path)
    
    data = df['Close'].values.reshape(-1, 1)
    last_60_days = data[-60:].reshape(1, 60, 1)
    
    predicted_price = model.predict(last_60_days)
    return scaler.inverse_transform(predicted_price)[0][0], df, scaler

def plot_predictions(df, predicted_price, scaler):
    # Add the predicted price to the original dataframe
    df["Predicted_Close"] = np.nan
    df.loc[df.index[-1], "Predicted_Close"] = predicted_price

    # Plot the actual closing prices and the predicted closing price
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, scaler.inverse_transform(df[['Close']]), label='Actual Close Prices')
    plt.plot(df.index, df["Predicted_Close"], label='Predicted Close Price', marker='o')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    predicted_price, df, scaler = predict_next_day('../models/trained_model.h5', '../data/stock_data.csv')
    print(f"Predicted Price: {predicted_price}")
    plot_predictions(df, predicted_price, scaler)
