import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Algorithmic Trading Dashboard", layout="wide")
st.title("📈 Algorithmic Trading & Backtesting Dashboard")

# --- SIDEBAR: Global User Inputs ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT, INFY.NS):", "AAPL").upper()

# 2. Data Fetching and Math Engine (Cached)
@st.cache_data(ttl=3600)
def load_data(ticker, period="5y"):  # Always grab 5 years so backtesting has deep data
    data = yf.Ticker(ticker)
    df = data.history(period=period)
    info = data.info
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
            
            if raw_df.empty:
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
                    prev_close = df['Close'].iloc[-2]
                    price_change = latest_close - prev_close
                    pct_change = (price_change / prev_close) * 100
                    
                    latest_rsi = df['RSI'].iloc[-1]
                    latest_sma50 = df['SMA_50'].iloc[-1]
                    latest_sma200 = df['SMA_200'].iloc[-1]
                    peg_ratio = stock_info.get('pegRatio', 'N/A')
                    
                    # Algorithm Score Engine
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
                    col3.metric("50-Day SMA", f"${latest_sma50:.2f}")
                    col4.metric("Recommendation", recommendation)

                    st.markdown("---")
                    
                    chart_col, text_col = st.columns([3, 1])
                    with chart_col:
                        st.subheader("Interactive Price Chart")
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=df.index[-252:], # Show past 1 year on chart
                                        open=df['Open'][-252:], high=df['High'][-252:],
                                        low=df['Low'][-252:], close=df['Close'][-252:], name='Price'))
                        fig.add_trace(go.Scatter(x=df.index[-252:], y=df['SMA_50'][-252:], line=dict(color='blue', width=1.5), name='50 SMA'))
                        fig.add_trace(go.Scatter(x=df.index[-252:], y=df['SMA_200'][-252:], line=dict(color='orange', width=1.5), name='200 SMA'))
                        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=0, r=0, t=20, b=0))
                        st.plotly_chart(fig, use_container_width=True)

                    with text_col:
                        st.subheader("Signal Rationale")
                        st.markdown(f"**Verdict:** <span style='color:{rec_color}; font-size:20px; font-weight:bold;'>{recommendation}</span>", unsafe_allow_html=True)
                        for r in reasons:
                            st.write(r)
                
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
