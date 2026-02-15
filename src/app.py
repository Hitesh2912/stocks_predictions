import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. OFFICIAL TERMINAL UI ---
st.set_page_config(page_title="NIFTY AI | PRO TERMINAL", layout="wide")

st.markdown("""
    <style>
    /* Professional Dark Palette */
    .stApp { background-color: #05070a; color: #e0e0e0; }
    
    /* Terminal Header */
    .terminal-header {
        border-bottom: 2px solid #1e222d;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Official Data Card */
    .data-card {
        background-color: #0d1117;
        border: 1px solid #1e222d;
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    
    /* Typography */
    .ticker-title { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.5rem; letter-spacing: -0.5px; }
    .price-display { font-family: 'JetBrains Mono', 'Courier New', monospace; font-size: 2.2rem; }
    
    /* Target Display - Official Look */
    .target-box {
        background-color: #0d1117;
        border: 1px solid #30363d;
        padding: 25px;
        border-left: 4px solid #58a6ff; 
        margin-top: 15px;
    }

    /* Minimalist Sidebar */
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1e222d; }
    
    /* Professional Button */
    .stButton>button {
        width: 100%;
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton>button:hover { background-color: #30363d; border-color: #8b949e; }
    </style>
""", unsafe_allow_html=True)

# --- 2. ROBUST DATA ENGINE ---
@st.cache_data(ttl=300)
def fetch_market_data(ticker):
    commodity_map = {
        "GOLD (MCX)": "GC=F",
        "SILVER (MCX)": "SI=F",
        "CRUDE OIL": "CL=F",
        "NATURAL GAS": "NG=F",
        "COPPER": "HG=F"
    }
    search_ticker = commodity_map.get(ticker, ticker)
    
    # Using period="max" to ensure 10Y and ALL buttons have data to display
    data = yf.download(search_ticker, period="max", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex): 
        data.columns = data.columns.get_level_values(0)
    return data.dropna()

# --- 3. EXPANDED PROFESSIONAL WATCHLIST ---
WATCHLIST = {
    "MCX COMMODITIES": ["GOLD (MCX)", "SILVER (MCX)", "CRUDE OIL", "NATURAL GAS", "COPPER", "MCX.NS"],
    "NIFTY HEAVYWEIGHTS": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LICI.NS", "HINDUNILVR.NS"],
    "DEFENCE & PSU": ["HAL.NS", "BEL.NS", "RVNL.NS", "IRFC.NS", "IREDA.NS", "MAZAGONDOC.NS", "PFC.NS", "RECLTD.NS", "BHEL.NS", "GICRE.NS"],
    "TECHNOLOGY & GROWTH": ["ZOMATO.NS", "PAYTM.NS", "NYKAA.NS", "TATAELXSI.NS", "KPITTECH.NS", "LTIM.NS", "COFORGE.NS", "PERSISTENT.NS"],
    "AUTOMOTIVE": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS", "ASHOKLEY.NS"],
    "ENERGY & COMMODITIES": ["ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "HINDALCO.NS"],
    "PHARMA & CONSUMER": ["TITAN.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "APOLLOHOSP.NS", "MAXHEALTH.NS", "TRENT.NS", "DIVISLAB.NS", "CIPLA.NS"]
}

with st.sidebar:
    st.markdown("### MARKET SELECTOR")
    sector = st.selectbox("SECTOR", list(WATCHLIST.keys()))
    asset = st.selectbox("INSTRUMENT", WATCHLIST[sector])
    st.markdown("---")
    st.caption("AI Terminal v3.0 | Real-time Data")

# --- 4. TERMINAL INTERFACE ---
df = fetch_market_data(asset)

if not df.empty:
    last_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = last_price - prev_close
    pct_change = (change / prev_close) * 100
    
    st.markdown(f"""
        <div class="terminal-header">
            <span class="ticker-title">{asset}</span>
            <span style="margin-left:15px; font-family:monospace; color:{'#3fb950' if change >= 0 else '#f85149'}">
                {last_price:,.2f} | {change:+.2f} ({pct_change:+.2f}%)
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["[ MARKET DATA ]", "[ NEURAL PROJECTION ]"])

    with tab1:
        # Professional Grid
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**OPEN** \n`{df['Open'].iloc[-1]:,.2f}`")
        with c2:
            st.markdown(f"**HIGH** \n`{df['High'].iloc[-1]:,.2f}`")
        with c3:
            st.markdown(f"**LOW** \n`{df['Low'].iloc[-1]:,.2f}`")

        # Pro Chart with Timeframe Selector Buttons
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], 
            low=df['Low'], close=df['Close'],
            increasing_line_color='#3fb950', decreasing_line_color='#f85149'
        )])
        
        fig.update_layout(
            template="plotly_dark", height=480,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.04),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(count=10, label="10Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]),
                    bgcolor="#1e222d",
                    activecolor="#58a6ff",
                    font=dict(size=11),
                    y=1.05
                ),
                showgrid=False
            ),
            yaxis=dict(side="right", showgrid=True, gridcolor='#1e222d'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with tab2:
        st.markdown("#### QUANTITATIVE FORECASTING")
        if st.button('EXECUTE NEURAL PROJECTION'):
            with st.spinner('SYSTEM: Processing LSTM Sequence...'):
                scaler = MinMaxScaler()
                # Training on a recent subset for performance efficiency
                train_data = df.tail(1000)
                scaled = scaler.fit_transform(train_data[['Close']].values)
                X = np.array([scaled[i-60:i, 0] for i in range(60, len(scaled))]).reshape(-1, 60, 1)
                
                model = Sequential([
                    Bidirectional(LSTM(32, return_sequences=True), input_shape=(60, 1)),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='huber')
                model.fit(X, scaled[60:], epochs=8, batch_size=64, verbose=0)
                
                p = model.predict(scaled[-60:].reshape(1, 60, 1))
                target = float(scaler.inverse_transform(p)[0,0])
                delta = target - last_price
                delta_pct = (delta / last_price) * 100
                mkt_color = "#3fb950" if delta >= 0 else "#f85149"

                st.markdown(f"""
                    <div class="target-box" style="border-left-color: {mkt_color}">
                        <div style="font-size: 0.85rem; color: #8b949e; letter-spacing: 1px;">EXPECTED SESSION CLOSE</div>
                        <div class="price-display" style="color: {mkt_color};">₹{target:,.2f}</div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 1.1rem; color: {mkt_color};">
                            {delta:+.2f} ({delta_pct:+.2f}%)
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                
                
                st.caption("STATISTICAL CONFIDENCE: 60-Day Lookback Recursive Training")
else:
    st.error("SYSTEM: Data stream unavailable.")