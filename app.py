import streamlit as st
import yfinance as yf
import pandas as pd
import time
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Live Trading Dashboard", layout="wide")

# Custom CSS for black background and cyan text
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: #00FFFF;
        }
        .stMarkdown, .stText, .stTextInput, .stSelectbox, .stNumberInput, .stRadio, .stButton>button {
            color: #00FFFF !important;
        }
        .stPlotlyChart {
            background-color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Live Trading Dashboard")
st.markdown("Real-time prices and trading signals")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter ticker (e.g., AAPL, MSFT, TSLA)", value="AAPL")
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m"])
refresh_interval = st.sidebar.number_input("Refresh every (seconds)", min_value=5, max_value=300, value=30)

# Container for live updates
placeholder = st.empty()

def get_close_prices(data: pd.DataFrame, ticker: str = None) -> list[float]:
    """Extract close prices safely from DataFrame or MultiIndex"""
    if data is None or data.empty:
        return []

    close_col = data["Close"]
    if isinstance(close_col, pd.DataFrame):  # MultiIndex case
        close_col = close_col.iloc[:, 0]

    return close_col.astype(float).tolist()

# Live loop (runs while app is open)
while True:
    try:
        data = yf.download(ticker, period="1d", interval=interval, progress=False, auto_adjust=True)

        if data.empty:
            st.error(f"No data found for {ticker}")
            time.sleep(refresh_interval)
            continue

        prices = get_close_prices(data, ticker)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=prices,
            mode="lines",
            name="Price",
            line=dict(color="cyan")
        ))
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="cyan"),
            height=500,
            title=f"Live {ticker} Prices ({interval})"
        )

        with placeholder.container():
            st.subheader(f"Latest price: {prices[-1]:.2f}")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")

    time.sleep(refresh_interval)
