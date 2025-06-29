import yfinance as yf
import pandas as pd
import os

# Ensure the "data" directory exists
os.makedirs("../data", exist_ok=True)

# Define stock symbol and time period
stock_symbol = "AAPL"  # Change this to any stock symbol
start_date = "2020-01-01"
end_date = "2024-12-01"

# Download historical stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Save to CSV file
csv_file = "../data/stock_data.csv"
stock_data.to_csv(csv_file)