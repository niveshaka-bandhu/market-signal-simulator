import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. Page Configuration
st.set_page_config(page_title="Algorithmic Stock Analyzer", layout="wide")
st.title("📈 Algorithmic Stock Analyzer")
st.markdown("Analyze technical trends and fundamental valuation for any stock.")

# --- SIDEBAR: User Inputs ---
st.sidebar.header("Parameters")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA, INFY.NS):", "AAPL").upper()
timeframe = st.sidebar.selectbox("Select Timeframe:", ["1y", "2y", "5y"])

# 2. Data Fetching and Math Engine (Cached for speed)
@st.cache_data(ttl=3600) # Caches data for 1 hour to prevent constant API calls
def load_data(ticker, period):
    data = yf.Ticker(ticker)
    df = data.history(period=period)
    info = data.info
    return df, info

def calculate_indicators(df):
    # Simple Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# 3. Main Dashboard Logic
if ticker_input:
    with st.spinner(f"Fetching data for {ticker_input}..."):
        try:
            raw_df, stock_info = load_data(ticker_input, timeframe)
            
            if raw_df.empty:
                st.error("No data found. Please check the ticker symbol.")
            else:
                df = calculate_indicators(raw_df)
                
                # Get latest values for the dashboard
                latest_close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2]
                price_change = latest_close - prev_close
                pct_change = (price_change / prev_close) * 100
                
                latest_rsi = df['RSI'].iloc[-1]
                latest_sma50 = df['SMA_50'].iloc[-1]
                latest_sma200 = df['SMA_200'].iloc[-1]
                peg_ratio = stock_info.get('pegRatio', 'N/A')
                
                # --- ALGORITHM: Buy/Sell Logic ---
                score = 0
                reasons = []
                
                if latest_sma50 > latest_sma200:
                    score += 1
                    reasons.append("🟢 Bullish Trend (50 SMA > 200 SMA)")
                else:
                    score -= 1
                    reasons.append("🔴 Bearish Trend (50 SMA < 200 SMA)")
                    
                if latest_rsi < 35:
                    score += 1
                    reasons.append("🟢 Oversold Momentum (RSI < 35)")
                elif latest_rsi > 70:
                    score -= 1
                    reasons.append("🔴 Overbought Momentum (RSI > 70)")
                    
                if isinstance(peg_ratio, (int, float)) and peg_ratio < 1.0:
                    score += 1
                    reasons.append(f"🟢 Undervalued (PEG Ratio {peg_ratio} < 1)")
                elif isinstance(peg_ratio, (int, float)) and peg_ratio > 2.5:
                    score -= 1
                    reasons.append(f"🔴 Overvalued (PEG Ratio {peg_ratio} > 2.5)")
                
                # Determine Final Signal
                if score >= 1:
                    recommendation = "BUY"
                    rec_color = "green"
                elif score <= -1:
                    recommendation = "SELL"
                    rec_color = "red"
                else:
                    recommendation = "HOLD / NEUTRAL"
                    rec_color = "orange"

                # --- UI LAYOUT: Top Metrics Row ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${latest_close:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
                col2.metric("RSI (14-Day)", f"{latest_rsi:.1f}")
                col3.metric("50-Day SMA", f"${latest_sma50:.2f}")
                col4.metric("Recommendation", recommendation)

                st.markdown("---")

                # --- UI LAYOUT: Chart and Details ---
                chart_col, text_col = st.columns([3, 1]) # Chart takes 75% of width, text takes 25%

                with chart_col:
                    st.subheader("Interactive Price & Trend Chart")
                    # Build Plotly Candlestick Chart
                    fig = go.Figure()
                    
                    # Candlestick trace
                    fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['Open'], high=df['High'],
                                    low=df['Low'], close=df['Close'],
                                    name='Price'))
                    
                    # SMA traces
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                                             line=dict(color='blue', width=1.5), name='50-Day SMA'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], 
                                             line=dict(color='orange', width=1.5), name='200-Day SMA'))
                    
                    fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                with text_col:
                    st.subheader(f"Signal Rationale")
                    st.markdown(f"**Final Verdict:** <span style='color:{rec_color}; font-size:20px; font-weight:bold;'>{recommendation}</span>", unsafe_allow_html=True)
                    st.write("### Factors:")
                    for reason in reasons:
                        st.write(reason)
                        
                    st.write("### Company Info:")
                    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                    st.write(f"**Market Cap:** ${stock_info.get('marketCap', 0):,}")
                    st.write(f"**Forward P/E:** {stock_info.get('forwardPE', 'N/A')}")

        except Exception as e:
            st.error(f"An error occurred: {e}")