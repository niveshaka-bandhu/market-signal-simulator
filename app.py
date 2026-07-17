import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# 1. Page Configuration
st.set_page_config(page_title="Algorithmic Trading Dashboard", layout="wide")
st.title("📈 Algorithmic Trading & Backtesting Dashboard")

# --- SIDEBAR: Global User Inputs ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT, INFY.NS):", "AAPL").upper().strip()

# --- HELPER FUNCTIONS ---
def get_robust_session():
    """
    Creates a requests session. It attempts to use curl_cffi to impersonate 
    a real Chrome browser's TLS signature to bypass cloud IP blocks. 
    Falls back to normal requests with browser headers if curl_cffi is missing.
    """
    try:
        from curl_cffi import requests as curl_requests
        # Impersonate a real Chrome browser connection
        session = curl_requests.Session(impersonate="chrome")
    except ImportError:
        # Fallback to standard requests with realistic browser headers
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive"
        })
    return session

def fetch_from_stooq(ticker):
    """
    Keyless fallback database using direct CSV streams from Stooq.
    Supports US, Indian (.NS to .IN translation), and other global markets.
    """
    ticker_clean = ticker.upper().strip()
    
    # Translate standard Indian Yahoo tickers to Stooq format (.NS -> .IN)
    if ticker_clean.endswith('.NS'):
        ticker_clean = ticker_clean.replace('.NS', '.IN')
    # Default standard raw tickers (like AAPL) to US exchange format (.US)
    elif '.' not in ticker_clean:
        ticker_clean = f"{ticker_clean}.US"
        
    url = f"https://stooq.com/q/d/l/?s={ticker_clean}&i=d"
    
    df = pd.read_csv(url)
    if df.empty or 'Date' not in df.columns:
        raise ValueError(f"Could not locate '{ticker_clean}' in Stooq's backup database.")
        
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    # Filter for the last 5 years to match period scope
    five_years_ago = pd.Timestamp.now() - pd.DateOffset(years=5)
    df = df[df.index >= five_years_ago]
    
    # Standardize columns to match the yfinance schema
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols]
    df.index.name = 'Date'
    return df

# 2. Data Fetching and Math Engine (Cached)
@st.cache_data(ttl=3600)
def load_data(ticker, period="5y"):
    df = None
    info = {}
    
    # --- STEP 1: FETCH PRICE DATA ---
    try:
        session = get_robust_session()
        data = yf.Ticker(ticker, session=session)
        df = data.history(period=period)
        if df.empty:
            raise ValueError("Empty response from Yahoo Finance.")
    except Exception as yf_error:
        # Direct automatic fallback if Yahoo fails or blocks
        st.sidebar.warning("Yahoo API is rate-limited on the cloud server. Fetching from Stooq fallback...")
        try:
            df = fetch_from_stooq(ticker)
        except Exception as stooq_error:
            raise RuntimeError(
                f"Yahoo blocked request: {yf_error}. "
                f"Backup database also failed: {stooq_error}. "
                "Verify your ticker is correct or run the app locally."
            )
            
    # --- STEP 2: FETCH COMPANION METADATA ---
    try:
        session = get_robust_session()
        data = yf.Ticker(ticker, session=session)
        info = data.info
        if not isinstance(info, dict):
            info = {}
    except Exception:
        # Silent fallback to keep the backtesting engine active even if metadata fails
        info = {}
        
    return df, info

def calculate_indicators(df):
    df = df.copy()
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

# Run calculations
if ticker_input:
    with st.spinner("Fetching historical market data..."):
        try:
            raw_df, stock_info = load_data(ticker_input, "5y")
            
            if raw_df is None or raw_df.empty:
                st.error("No data found. Please check the ticker symbol.")
            else:
                df = calculate_indicators(raw_df)
                
                # --- TAB SETUP ---
                tab1, tab2 = st.tabs(["🔍 Live Signal Analysis", "📊 5-Year Strategy Backtest"])
                
                # ==========================================
                # TAB 1: LIVE SIGNAL ANALYSIS
                # ==========================================
                with tab1:
                    latest_close = df['Close'].iloc[-1]
                    prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
                    price_change = latest_close - prev_close
                    pct_change = (price_change / prev_close) * 100 if prev_close != 0 else 0
                    
                    latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50.0
                    latest_sma50 = df['SMA_50'].iloc[-1]
                    latest_sma200 = df['SMA_200'].iloc[-1]
                    
                    # Algorithm Score Engine
                    score = 0
                    reasons = []
                    
                    # Trend Rules
                    if not pd.isna(latest_sma50) and not pd.isna(latest_sma200):
                        if latest_sma50 > latest_sma200:
                            score += 1
                            reasons.append("🟢 Bullish Trend (50 SMA > 200 SMA)")
                        else:
                            score -= 1
                            reasons.append("🔴 Bearish Trend (50 SMA < 200 SMA)")
                    else:
                        reasons.append("⚪ SMAs still calculating (need more historical data)")
                        
                    # Momentum Rules
                    if latest_rsi < 35:
                        score += 1
                        reasons.append(f"🟢 Oversold Momentum (RSI {latest_rsi:.1f} < 35)")
                    elif latest_rsi > 70:
                        score -= 1
                        reasons.append(f"🔴 Overbought Momentum (RSI {latest_rsi:.1f} > 70)")
                    else:
                        reasons.append(f"🔵 Neutral Momentum (RSI is stable at {latest_rsi:.1f})")
                    
                    if score >= 1:
                        recommendation, rec_color = "BUY", "green"
                    elif score <= -1:
                        recommendation, rec_color = "SELL", "red"
                    else:
                        recommendation, rec_color = "HOLD / NEUTRAL", "orange"

                    # Metrics Row
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${latest_close:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
                    col2.metric("RSI (14-Day)", f"{latest_rsi:.1f}")
                    col3.metric("50-Day SMA", f"${latest_sma50:.2f}" if not pd.isna(latest_sma50) else "N/A")
                    col4.metric("Recommendation", recommendation)

                    st.markdown("---")
                    
                    chart_col, text_col = st.columns([3, 1])
                    with chart_col:
                        st.subheader("Interactive Price Chart")
                        fig = go.Figure()
                        # Use past 1 year (approx 252 trading days) for visual display
                        chart_df = df[-252:] if len(df) > 252 else df
                        fig.add_trace(go.Candlestick(x=chart_df.index,
                                        open=chart_df['Open'], high=chart_df['High'],
                                        low=chart_df['Low'], close=chart_df['Close'], name='Price'))
                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_50'], line=dict(color='blue', width=1.5), name='50 SMA'))
                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_200'], line=dict(color='orange', width=1.5), name='200 SMA'))
                        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=0, r=0, t=20, b=0))
                        st.plotly_chart(fig, use_container_width=True)

                    with text_col:
                        st.subheader("Signal Rationale")
                        st.markdown(f"**Verdict:** <span style='color:{rec_color}; font-size:20px; font-weight:bold;'>{recommendation}</span>", unsafe_allow_html=True)
                        for r in reasons:
                            st.write(r)
                        
                        st.markdown("---")
                        st.write("### Company Info:")
                        if isinstance(stock_info, dict) and stock_info:
                            st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                            market_cap = stock_info.get('marketCap', 0)
                            if market_cap:
                                st.write(f"**Market Cap:** ${market_cap:,}")
                            else:
                                st.write("**Market Cap:** N/A")
                            st.write(f"**Forward P/E:** {stock_info.get('forwardPE', 'N/A')}")
                        else:
                            st.info("Company profile metadata unavailable due to cloud environment rate limits. Technical analytics remain fully active.")
                
                # ==========================================
                # TAB 2: HISTORICAL BACKTEST
                # ==========================================
                with tab2:
                    st.subheader("Historical Simulation Engine (5-Year Lookback)")
                    st.write("This simulator calculates exactly how much money you would have made executing SMA Crossover trades compared to buying and holding the stock.")
                    
                    # Backtest Inputs
                    initial_capital = st.number_input("Starting Balance ($)", min_value=100, max_value=1000000, value=10000, step=500)
                    
                    # Clean up data for testing (remove days missing SMA values)
                    bt_df = df.dropna(subset=['SMA_200']).copy()
                    
                    if bt_df.empty:
                        st.error("Insufficient historical data to run backtest (requires at least 200 trading days).")
                    else:
                        # Simulation variables
                        position = 0  # 0 means cash, 1 means holding stock
                        cash = initial_capital
                        shares = 0
                        portfolio_history = []
                        trade_count = 0
                        
                        # Loop through chronological history
                        for date, row in bt_df.iterrows():
                            price = row['Close']
                            sma50 = row['SMA_50']
                            sma200 = row['SMA_200']
                            
                            # 1. Buy Logic (Golden Cross)
                            if position == 0 and sma50 > sma200:
                                shares = cash / price
                                cash = 0
                                position = 1
                                trade_count += 1
                            
                            # 2. Sell Logic (Death Cross)
                            elif position == 1 and sma50 < sma200:
                                cash = shares * price
                                shares = 0
                                position = 0
                                trade_count += 1
                                
                            # Track total portfolio value at the end of each day
                            current_value = cash + (shares * price)
                            portfolio_history.append(current_value)
                            
                        bt_df['Strategy_Value'] = portfolio_history
                        
                        # Benchmark: Buy & Hold Strategy
                        bh_shares = initial_capital / bt_df['Close'].iloc[0]
                        bt_df['Buy_Hold_Value'] = bh_shares * bt_df['Close']
                        
                        # Final Metrics
                        final_strategy_val = bt_df['Strategy_Value'].iloc[-1]
                        final_bh_val = bt_df['Buy_Hold_Value'].iloc[-1]
                        
                        strat_return = ((final_strategy_val - initial_capital) / initial_capital) * 100
                        bh_return = ((final_bh_val - initial_capital) / initial_capital) * 100
                        
                        # Show Performance Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Strategy Final Value", f"${final_strategy_val:,.2f}")
                        m2.metric("Strategy Total Return", f"{strat_return:.2f}%")
                        m3.metric("Buy & Hold Return", f"{bh_return:.2f}%")
                        m4.metric("Trades Executed", f"{trade_count}")
                        
                        # Performance Verdict
                        st.markdown("---")
                        if final_strategy_val > final_bh_val:
                            st.success(f"🏆 **Victory!** Your SMA Crossover strategy beat the market index/stock buy-and-hold strategy by **{strat_return - bh_return:.2f}%**.")
                        else:
                            st.warning(f"⚠️ **Market Beats Strategy.** Buying and holding would have made you **{bh_return - strat_return:.2f}%** more than using this system.")
                        
                        # Plotly Equity Curve Comparison Chart
                        st.subheader("Equity Growth Curve: Strategy vs. Buy & Hold")
                        equity_fig = go.Figure()
                        equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Strategy_Value'], 
                                                        line=dict(color='green', width=2), name='SMA Strategy Balance'))
                        equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Buy_Hold_Value'], 
                                                        line=dict(color='grey', width=1.5, dash='dash'), name='Buy & Hold Benchmark'))
                        equity_fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Portfolio Value ($)")
                        st.plotly_chart(equity_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load data for backtesting: {e}")
