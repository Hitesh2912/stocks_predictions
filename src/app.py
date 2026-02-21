import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import plotly.graph_objects as go

# --- 1. PRO TERMINAL STYLING ---
st.set_page_config(page_title="QUANT-X | INSTITUTIONAL", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e0e0e0; }
    .terminal-header { border-bottom: 2px solid #1e222d; padding: 10px 0; margin-bottom: 20px; }
    .ticker-title { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.6rem; }
    .price-display { font-family: 'JetBrains Mono'; font-size: 2.5rem; font-weight: 700; }
    .target-box {
        background-color: #0d1117; border: 1px solid #30363d;
        padding: 25px; border-left: 5px solid #58a6ff; border-radius: 4px;
    }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1e222d; }
    .stMetric { background-color: #0d1117; padding: 10px; border-radius: 5px; border: 1px solid #1e222d; }
    </style>
""", unsafe_allow_html=True)

# --- 2. THE REAL-WORLD WATCHLIST (2026 ACTIVE) ---
WATCHLIST = {
    "NSE BLUECHIPS": ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "BHARTIARTL.NS", "SBIN.NS"],
    "US TECH GIANTS": ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AVGO"],
    "NSE GROWTH/PSU": ["HAL.NS", "BEL.NS", "RVNL.NS", "ZOMATO.NS", "JIOFIN.NS", "IREDA.NS", "ADANIENT.NS"],
    "GLOBAL COMMODITIES": ["GC=F", "SI=F", "CL=F", "NG=F"]
}

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=300)
def fetch_pro_data(ticker):
    # Fetching 5 years of daily data for deep learning
    data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    
    # Feature 1: RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Feature 2: EMA 20 (Short-term trend)
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    return data.dropna()

# --- 4. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### TERMINAL CONFIG")
    sector = st.selectbox("MARKET SECTOR", list(WATCHLIST.keys()))
    asset = st.selectbox("SELECT TICKER", WATCHLIST[sector])
    
    st.markdown("---")
    lookback = st.slider("Lookback Window", 30, 90, 60)
    epochs = st.select_slider("AI Training Depth", options=[5, 10, 20], value=10)
    st.caption("Higher depth = Better accuracy (Higher RAM)")

# --- 5. DASHBOARD INTERFACE ---
df = fetch_pro_data(asset)

if not df.empty:
    last_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = last_price - prev_close
    pct_change = (change / prev_close) * 100
    
    # Header Section
    st.markdown(f"""
        <div class="terminal-header">
            <span class="ticker-title">{asset} | GLOBAL EXCHANGE</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Price and Indicators Row
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        st.markdown(f'<div class="price-display" style="color:{"#3fb950" if change >= 0 else "#f85149"}">'
                    f'{last_price:,.2f} <span style="font-size:1rem;">{change:+.2f} ({pct_change:+.2f}%)</span></div>', unsafe_allow_html=True)
    c2.metric("RSI (14D)", f"{df['RSI'].iloc[-1]:.2f}")
    c3.metric("EMA (20D)", f"{df['EMA20'].iloc[-1]:,.2f}")
    c4.metric("VOL (24H)", f"{df['Volume'].iloc[-1]:,.0f}")

    tab1, tab2 = st.tabs(["📊 TECHNICAL CHARTS", "🧠 AI NEURAL TARGET"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#58a6ff', width=1.5), name="Trend Line"))
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, yaxis_side="right")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### MULTI-FEATURE NEURAL PROJECTION")
        if st.button('GENERATE INSTITUTIONAL FORECAST'):
            with st.spinner('Accessing Neural Core...'):
                # 1. Feature Engineering
                features = ['Close', 'RSI', 'Volume']
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[features].tail(1200).values)
                
                # 2. Sequence Creation
                X, y = [], []
                for i in range(lookback, len(scaled_data)):
                    X.append(scaled_data[i-lookback:i])
                    y.append(scaled_data[i, 0])
                X, y = np.array(X), np.array(y)

                # 3. Model Build (The Brain)
                model = Sequential([
                    Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, 3)),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False),
                    Dense(16, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='huber')
                model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

                # 4. Prediction Logic
                last_window = scaled_data[-lookback:].reshape(1, lookback, 3)
                prediction_scaled = model.predict(last_window, verbose=0)
                
                # Correct Inverse Scaling Fix
                dummy = np.zeros((1, 3))
                dummy[0, 0] = float(prediction_scaled.flatten()[0])
                dummy[0, 1] = scaled_data[-1, 1] 
                dummy[0, 2] = scaled_data[-1, 2]
                target = float(scaler.inverse_transform(dummy)[0, 0])
                
                # 5. Signal Logic
                delta = target - last_price
                sig_color = "#3fb950" if delta >= 0 else "#f85149"
                rsi_val = df['RSI'].iloc[-1]
                
                if delta > 0 and rsi_val < 40: sentiment = "STRONG ACCUMULATION"
                elif delta > 0: sentiment = "BULLISH BIAS"
                elif delta < 0 and rsi_val > 65: sentiment = "DISTRIBUTION ALERT"
                else: sentiment = "NEUTRAL / SIDEWAYS"

                st.markdown(f"""
                    <div class="target-box" style="border-left-color: {sig_color}">
                        <div style="font-size: 0.9rem; color: #8b949e;">PROJECTED SESSION TARGET</div>
                        <div class="price-display" style="color: {sig_color};">{target:,.2f}</div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 1.2rem; color: {sig_color};">
                            {delta:+.2f} ({(delta/last_price)*100:+.2f}%)
                        </div>
                        <hr style="border: 0.5px solid #30363d;">
                        <div style="font-weight: 700; color: #58a6ff;">AI SENTIMENT: {sentiment}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                

                st.info("Note: AI projections are statistical probabilities based on 4+ years of data.")
else:
    st.error("Market Data Stream Error. Check Ticker symbol.")