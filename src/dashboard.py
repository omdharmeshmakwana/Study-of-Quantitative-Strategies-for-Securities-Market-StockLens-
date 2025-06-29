from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import os
from .data_loader import load_data


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

@app.route("/")
def home():
    df, _ = load_data("data/stock_data.csv")
    print(df)  # Debugging: Check if df is None or empty
    print(type(df))  # Debugging: Check the type of df

    if df is None or df.empty:
        return "Error: Data could not be loaded!", 500  # Return an error response

    fig = px.line(df, x=df.index, y='Close', title="Stock Price Over Time")
    graph_html = fig.to_html(full_html=False)
    return render_template("index.html", graph_html=graph_html)


if __name__ == "__main__":
    app.run(debug=True)