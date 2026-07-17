import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# 1. Page Configuration
st.set_page_config(page_title="Indian Algo Trading Dashboard", layout="wide")
st.title("📈 Indian Market Trading & Backtesting Dashboard")

# --- SIDEBAR: Global User Inputs ---
st.sidebar.header("Configuration")
raw_ticker_input = st.sidebar.text_input(
    "Enter Indian Stock Ticker (e.g., RELIANCE, TCS, INFY, ^NSEI):", 
    "RELIANCE"
).upper().strip()

# --- SMART AUTO-FORMATTING FOR INDIAN MARKET ---
if not raw_ticker_input.startswith('^') and '.' not in raw_ticker_input:
    yfinance_ticker = f"{raw_ticker_input}.NS"
else:
    yfinance_ticker = raw_ticker_input

# --- HELPER FUNCTIONS ---
def get_robust_session():
    try:
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome")
    except ImportError:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive"
        })
    return session

def fetch_from_stooq(ticker):
    ticker_clean = ticker.upper().strip()
    if ticker_clean.startswith('^'):
        index_mapping = {'^NSEI': '^NIFTY', '^BSESN': '^SENSEX'}
        ticker_clean = index_mapping.get(ticker_clean, ticker_clean)
    elif ticker_clean.endswith('.NS'):
        ticker_clean = ticker_clean.replace('.NS', '.IN')
    elif '.' not in ticker_clean:
        ticker_clean = f"{ticker_clean}.IN"
        
    url = f"https://stooq.com/q/d/l/?s={ticker_clean}&i=d"
    df = pd.read_csv(url)
    if df.empty or 'Date' not in df.columns:
        raise ValueError(f"Could not locate '{ticker_clean}' in backup database.")
        
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    five_years_ago = pd.Timestamp.now() - pd.DateOffset(years=5)
    df = df[df.index >= five_years_ago]
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols]
    df.index.name = 'Date'
    return df

@st.cache_data(ttl=3600)
def load_data(ticker, period="5y"):
    df = None
    info = {}
    try:
        session = get_robust_session()
        data = yf.Ticker(ticker, session=session)
        df = data.history(period=period)
        if df.empty:
            raise ValueError("Empty response from Yahoo Finance.")
    except Exception as yf_error:
        st.sidebar.warning("Yahoo API is rate-limited on the cloud server. Fetching from Stooq fallback...")
        try:
            df = fetch_from_stooq(ticker)
        except Exception as stooq_error:
            raise RuntimeError(f"Yahoo blocked: {yf_error}. Backup failed: {stooq_error}.")
            
    try:
        session = get_robust_session()
        data = yf.Ticker(ticker, session=session)
        info = data.info
        if not isinstance(info, dict):
            info = {}
    except Exception:
        info = {}
        
    return df, info

def calculate_indicators(df):
    df = df.copy()
    # 1. SMAs
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    return df

# Run calculations
if yfinance_ticker:
    with st.spinner("Analyzing market algorithms..."):
        try:
            raw_df, stock_info = load_data(yfinance_ticker, "5y")
            
            if raw_df is None or raw_df.empty:
                st.error("No data found. Please check the ticker symbol.")
            else:
                df = calculate_indicators(raw_df)
                
                # --- TAB SETUP ---
                tab1, tab2 = st.tabs(["🔍 Live Signal Analysis", "📊 Historical Strategy Backtest"])
                
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
                    latest_macd = df['MACD'].iloc[-1]
                    latest_signal = df['Signal_Line'].iloc[-1]
                    
                    # --- DUAL-ALGO MASTER VERDICT ENGINE ---
                    score = 0
                    reasons = []
                    
                    # Algo 1: SMA Crossover (Trend)
                    if not pd.isna(latest_sma50) and not pd.isna(latest_sma200):
                        if latest_sma50 > latest_sma200:
                            score += 1
                            reasons.append("🟢 **SMA Crossover:** Bullish (Long-term upward trend intact)")
                        else:
                            score -= 1
                            reasons.append("🔴 **SMA Crossover:** Bearish (Long-term downward trend dominates)")
                    
                    # Algo 2: MACD Crossover (Momentum/Short-Term Trend)
                    if not pd.isna(latest_macd) and not pd.isna(latest_signal):
                        if latest_macd > latest_signal:
                            score += 1
                            reasons.append("🟢 **MACD Crossover:** Bullish (Short-term upward momentum is accelerating)")
                        else:
                            score -= 1
                            reasons.append("🔴 **MACD Crossover:** Bearish (Short-term downward pressure is building)")
                            
                    # Algo 3: RSI Overbought/Oversold Filter
                    if latest_rsi < 35:
                        score += 1
                        reasons.append(f"🟢 **RSI Index:** Highly Oversold ({latest_rsi:.1f}) - prime buying territory")
                    elif latest_rsi > 70:
                        score -= 1
                        reasons.append(f"🔴 **RSI Index:** Overbought ({latest_rsi:.1f}) - due for a cooling-off period")
                    else:
                        reasons.append(f"🔵 **RSI Index:** Neutral ({latest_rsi:.1f}) - trading in a healthy channel")
                    
                    # Translate Score to Actionable Investment Verdict
                    if score >= 2:
                        verdict = "🟢 YES - HIGH CONVICTION BUY"
                        verdict_msg = "All major algorithmic signals have aligned bullishly. Excellent window to invest."
                        bg_color = "#d4edda"
                        text_color = "#155724"
                    elif score == 1:
                        verdict = "🟡 YES, BUT CAUTIOUS BUY"
                        verdict_msg = "Overall momentum is positive, but some technical indicators suggest a sub-optimal entry price."
                        bg_color = "#fff3cd"
                        text_color = "#856404"
                    elif score == 0:
                        verdict = "⚪ HOLD / DO NOT INVEST YET"
                        verdict_msg = "The algorithms are in conflict or moving sideways. Best to wait on the sidelines for a clear direction."
                        bg_color = "#e2e3e5"
                        text_color = "#383d41"
                    else:
                        verdict = "🔴 NO - STAY OUT / HIGH RISK"
                        verdict_msg = "Downward trends and selling pressure dominate. Investing right now carries elevated risk."
                        bg_color = "#f8d7da"
                        text_color = "#721c24"

                    # Custom Investment Recommendation Box
                    st.markdown(f"""
                    <div style="background-color:{bg_color}; padding:20px; border-radius:10px; border-left:8px solid {text_color}; margin-bottom: 25px;">
                        <h3 style="margin:0; color:{text_color}; font-weight:bold;">INVESTMENT VERDICT: {verdict}</h3>
                        <p style="margin:5px 0 0 0; color:{text_color}; font-size:16px;">{verdict_msg}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Quick Stats Cards
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"₹{latest_close:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
                    col2.metric("RSI (14-Day)", f"{latest_rsi:.1f}")
                    col3.metric("MACD vs Signal", f"{latest_macd:.2f} / {latest_signal:.2f}")
                    col4.metric("Score", f"{score:+.0f}")

                    st.markdown("---")
                    
                    chart_col, text_col = st.columns([3, 1])
                    with chart_col:
                        st.subheader("Interactive Technical Analysis Charts")
                        chart_df = df[-252:] if len(df) > 252 else df
                        
                        # Subplot 1: Candlesticks & SMAs
                        fig1 = go.Figure()
                        fig1.add_trace(go.Candlestick(x=chart_df.index,
                                        open=chart_df['Open'], high=chart_df['High'],
                                        low=chart_df['Low'], close=chart_df['Close'], name='Price'))
                        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_50'], line=dict(color='blue', width=1.5), name='50 SMA'))
                        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_200'], line=dict(color='orange', width=1.5), name='200 SMA'))
                        fig1.update_layout(xaxis_rangeslider_visible=False, height=350, margin=dict(l=0, r=0, t=10, b=10))
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Subplot 2: Dedicated MACD Oscillator
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MACD'], line=dict(color='purple', width=1.5), name='MACD Line'))
                        fig2.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Signal_Line'], line=dict(color='red', width=1.2, dash='dot'), name='Signal Line'))
                        
                        # Custom colors for MACD histogram bars (Green for positive, Red for negative)
                        colors = ['green' if val >= 0 else 'red' for val in chart_df['MACD_Hist']]
                        fig2.add_trace(go.Bar(x=chart_df.index, y=chart_df['MACD_Hist'], marker_color=colors, name='Histogram', opacity=0.5))
                        fig2.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=10))
                        st.plotly_chart(fig2, use_container_width=True)

                    with text_col:
                        st.subheader("Why this Verdict?")
                        for r in reasons:
                            st.write(r)
                        
                        st.markdown("---")
                        st.write("### Profile Metadata:")
                        if isinstance(stock_info, dict) and stock_info:
                            st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                            market_cap = stock_info.get('marketCap', 0)
                            if market_cap:
                                st.write(f"**Market Cap:** ₹{market_cap:,}")
                            st.write(f"**Forward P/E:** {stock_info.get('forwardPE', 'N/A')}")
                        else:
                            st.info("Metadata profile loaded from backup. Technical analytics remain fully active.")
                
                # ==========================================
                # TAB 2: HISTORICAL BACKTEST
                # ==========================================
                with tab2:
                    st.subheader("Historical Simulation Engine (5-Year Lookback)")
                    
                    # Backtest controls
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        strategy_choice = st.selectbox("Select Backtesting Strategy:", ["SMA Crossover (50 vs 200)", "MACD Line Crossover"])
                    with b_col2:
                        initial_capital = st.number_input("Starting Balance (₹)", min_value=100, max_value=10000000, value=100000, step=1000)
                    
                    # Backtest Execution logic
                    bt_df = df.dropna(subset=['SMA_200']).copy() if strategy_choice == "SMA Crossover (50 vs 200)" else df.dropna(subset=['Signal_Line']).copy()
                    
                    if bt_df.empty:
                        st.error("Insufficient historical data to run backtest.")
                    else:
                        position = 0
                        cash = initial_capital
                        shares = 0
                        portfolio_history = []
                        trade_count = 0
                        
                        for date, row in bt_df.iterrows():
                            price = row['Close']
                            
                            if strategy_choice == "SMA Crossover (50 vs 200)":
                                buy_cond = row['SMA_50'] > row['SMA_200']
                                sell_cond = row['SMA_50'] < row['SMA_200']
                            else: # MACD Choice
                                buy_cond = row['MACD'] > row['Signal_Line']
                                sell_cond = row['MACD'] < row['Signal_Line']
                                
                            # Buy Execution
                            if position == 0 and buy_cond:
                                shares = cash / price
                                cash = 0
                                position = 1
                                trade_count += 1
                            # Sell Execution
                            elif position == 1 and sell_cond:
                                cash = shares * price
                                shares = 0
                                position = 0
                                trade_count += 1
                                
                            portfolio_history.append(cash + (shares * price))
                            
                        bt_df['Strategy_Value'] = portfolio_history
                        bh_shares = initial_capital / bt_df['Close'].iloc[0]
                        bt_df['Buy_Hold_Value'] = bh_shares * bt_df['Close']
                        
                        final_strategy_val = bt_df['Strategy_Value'].iloc[-1]
                        final_bh_val = bt_df['Buy_Hold_Value'].iloc[-1]
                        strat_return = ((final_strategy_val - initial_capital) / initial_capital) * 100
                        bh_return = ((final_bh_val - initial_capital) / initial_capital) * 100
                        
                        # Show Performance Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Strategy Final Value", f"₹{final_strategy_val:,.2f}")
                        m2.metric("Strategy Total Return", f"{strat_return:.2f}%")
                        m3.metric("Buy & Hold Return", f"{bh_return:.2f}%")
                        m4.metric("Trades Executed", f"{trade_count}")
                        
                        st.markdown("---")
                        if final_strategy_val > final_bh_val:
                            st.success(f"🏆 **Victory!** {strategy_choice} beat the market buy-and-hold strategy by **{strat_return - bh_return:.2f}%**.")
                        else:
                            st.warning(f"⚠️ **Market Beats Strategy.** Buying and holding would have made you **{bh_return - strat_return:.2f}%** more than using this system.")
                        
                        # Plotly Chart
                        equity_fig = go.Figure()
                        equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Strategy_Value'], line=dict(color='green', width=2), name=f'{strategy_choice}'))
                        equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Buy_Hold_Value'], line=dict(color='grey', width=1.5, dash='dash'), name='Buy & Hold Benchmark'))
                        equity_fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Portfolio Value (₹)")
                        st.plotly_chart(equity_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not complete calculation run: {e}")
