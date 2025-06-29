import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(file_path):
    # ✅ Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found!")
        return None, None
    else:
        print(f"✅ File '{file_path}' found. Attempting to load...")

    # 📂 Read CSV file
    try:
        print(f"📂 Loading CSV file: {file_path}")
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        print("✅ CSV successfully loaded! Here's a preview:")
        print(df.head())
    except Exception as e:
        print(f"❌ Error reading the CSV file: {e}")
        return None, None

    # 🔍 Check required columns
    expected_cols = ["Close", "High", "Low", "Open", "Volume"]
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        print(f"❌ Error: Missing columns in CSV: {missing_cols}")
        return None, None

    # 🔄 Convert columns to numeric (✅ FIXED: using list instead of set)
    try:
        df[expected_cols] = df[expected_cols].apply(pd.to_numeric, errors="coerce")
        df.dropna(how='all', inplace=True)
        if df.empty:
            print("❌ Error: DataFrame is empty after dropping NaNs!")
            return None, None
    except Exception as e:
        print(f"❌ Error during numeric conversion: {e}")
        return None, None

    # 📉 Normalize "Close" column
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df["Close"] = scaler.fit_transform(df[["Close"]])
        print("\n✅ Normalized 'Close' column preview:")
        print(df.head())
    except Exception as e:
        print(f"❌ Error during normalization: {e}")
        return None, None

    return df, scaler

if __name__ == "__main__":
    file_path = "data/stock_data.csv"
    df, scaler = load_data(file_path)
    if df is not None:
        print("\n✅ Final processed data preview:")
        print(df.head())
    else:
        print("❌ Failed to process data.")
