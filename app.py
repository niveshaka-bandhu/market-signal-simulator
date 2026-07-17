import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

# ==========================================
# 1. PAGE CONFIGURATION & THEME SETUP
# ==========================================
st.set_page_config(page_title="Indian Quant Deep-Dive Dashboard", layout="wide")
st.title("📊 Indian Quant Trading & Deep-Dive Dashboard")

# ==========================================
# 2. GLOBAL SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("Configuration")
raw_ticker_input = st.sidebar.text_input(
    "Enter Indian Ticker (e.g., RELIANCE, TCS, INFY, ^NSEI):", 
    "RELIANCE"
).upper().strip()

# Exchange Suffixing Logic
if not raw_ticker_input.startswith('^') and '.' not in raw_ticker_input:
    yfinance_ticker = f"{raw_ticker_input}.NS"
else:
    yfinance_ticker = raw_ticker_input

st.sidebar.markdown("---")
st.sidebar.subheader("Technical Overlays")
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_fib = st.sidebar.checkbox("Show Fibonacci Levels", value=False)

# ==========================================
# 3. ROBUST DATA FETCHING PIPELINE
# ==========================================
def get_robust_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }

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
        session = requests.Session()
        session.headers.update(get_robust_headers())
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
        session = requests.Session()
        session.headers.update(get_robust_headers())
        data = yf.Ticker(ticker, session=session)
        info = data.info
        if not isinstance(info, dict):
            info = {}
    except Exception:
        info = {}
        
    return df, info

# --- LIGHTWEIGHT BEAUTIFULSOUP SCREENER SCRAPER ---
@st.cache_data(ttl=86400)
def load_screener_data_light(ticker):
    """
    Scrapes Screener.in's key ratios, pros, and cons without a headless browser.
    """
    clean_symbol = ticker.split(".")[0].strip().upper()
    if clean_symbol.startswith('^'):
        return {"success": False, "error": "Indices not supported on Screener.in"}
        
    url = f"https://www.screener.in/company/{clean_symbol}/"
    try:
        response = requests.get(url, headers=get_robust_headers(), timeout=10)
        if response.status_code != 200:
            return {"success": False, "error": f"Failed HTTP response ({response.status_code})"}
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Parse Key Ratios
        ratios = {}
        ratios_container = soup.find('ul', id='top-ratios') or soup.find('div', class_='company-ratios')
        if ratios_container:
            for item in ratios_container.find_all('li'):
                name_span = item.find('span', class_='name')
                value_span = item.find('span', class_='number') or item.find_all('span')[-1]
                if name_span and value_span:
                    name = name_span.text.strip().replace(":", "")
                    value = value_span.text.strip().replace('\n', '').replace('  ', ' ')
                    ratios[name] = value

        # Parse Pros and Cons
        pros = []
        cons = []
        pros_section = soup.find('div', class_='pros')
        cons_section = soup.find('div', class_='cons')
        
        if pros_section:
            pros = [li.text.strip() for li in pros_section.find_all('li')]
        if cons_section:
            cons = [li.text.strip() for li in cons_section.find_all('li')]
            
        return {
            "success": True,
            "company_name": clean_symbol,
            "ratios": ratios,
            "pros": pros,
            "cons": cons
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_ratio_value(ratios_dict, key_patterns):
    for key, value in ratios_dict.items():
        key_lower = key.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        for pattern in key_patterns:
            if pattern in key_lower:
                try:
                    clean_val = str(value).replace("%", "").replace(",", "").strip()
                    return float(clean_val)
                except ValueError:
                    return value
    return None

def calculate_indicators(df):
    df = df.copy()
    # SMAs
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    return df

# ==========================================
# 4. RUN CALCULATIONS & GENERATE INTERFACE
# ==========================================
if yfinance_ticker:
    with st.spinner("Fetching market data and running calculations..."):
        try:
            raw_df, stock_info = load_data(yfinance_ticker, "5y")
            screener_data = load_screener_data_light(yfinance_ticker)
            
            if raw_df is None or raw_df.empty:
                st.error("No data returned. Please verify ticker symbols.")
            else:
                df = calculate_indicators(raw_df)
                
                # Tabbed Interface Setup
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🔍 Live Deep Analysis", 
                    "📊 Historical Backtest", 
                    "📈 Volatility & Risk Metrics",
                    "💎 Intrinsic Value Calculators"
                ])
                
                # Baseline metrics
                latest_close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
                price_change = latest_close - prev_close
                pct_change = (price_change / prev_close) * 100 if prev_close != 0 else 0
                
                # ==========================================
                # TAB 1: LIVE DEEP ANALYSIS
                # ==========================================
                with tab1:
                    latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50.0
                    latest_sma50 = df['SMA_50'].iloc[-1]
                    latest_sma200 = df['SMA_200'].iloc[-1]
                    latest_macd = df['MACD'].iloc[-1]
                    latest_signal = df['Signal_Line'].iloc[-1]
                    
                    # Scoring Logic
                    score = 0
                    reasons = []
                    
                    # 1. Long-Term Trend
                    if not pd.isna(latest_sma50) and not pd.isna(latest_sma200):
                        if latest_sma50 > latest_sma200:
                            score += 1
                            reasons.append("🟢 **Long-Term Trend (SMA):** Bullish (50 SMA is above 200 SMA)")
                        else:
                            score -= 1
                            reasons.append("🔴 **Long-Term Trend (SMA):** Bearish (50 SMA is below 200 SMA)")
                    
                    # 2. Short-Term Momentum (MACD)
                    if not pd.isna(latest_macd) and not pd.isna(latest_signal):
                        if latest_macd > latest_signal:
                            score += 1
                            reasons.append("🟢 **Short-Term Momentum (MACD):** Bullish crossover detected")
                        else:
                            score -= 1
                            reasons.append("🔴 **Short-Term Momentum (MACD):** Bearish crossover detected")
                            
                    # 3. Exhaustion Index (RSI)
                    if latest_rsi < 30:
                        score += 1.5
                        reasons.append(f"🟢 **Exhaustion (RSI):** Oversold ({latest_rsi:.1f}) - Sellers are exhausted")
                    elif latest_rsi > 70:
                        score -= 1.5
                        reasons.append(f"🔴 **Exhaustion (RSI):** Overbought ({latest_rsi:.1f}) - Buyers are exhausted")
                    else:
                        reasons.append(f"🔵 **Exhaustion (RSI):** Neutral ({latest_rsi:.1f}) - Trend is balanced")

                    # 4. Bollinger Band Position
                    latest_bbu = df['BB_Upper'].iloc[-1]
                    latest_bbl = df['BB_Lower'].iloc[-1]
                    if not pd.isna(latest_bbu) and not pd.isna(latest_bbl):
                        if latest_close >= latest_bbu:
                            score -= 1
                            reasons.append("🔴 **Volatility (BB):** Overextended (Price above Upper Bollinger Band)")
                        elif latest_close <= latest_bbl:
                            score += 1
                            reasons.append("🟢 **Volatility (BB):** Underextended (Price below Lower Bollinger Band)")
                    
                    # Formulate Investment Verdict
                    if score >= 2:
                        verdict, verdict_msg, bg_color, text_color = "🟢 YES - HIGH CONVICTION BUY", "Technical parameters have aligned cleanly. The risk-to-reward ratio is in your favor.", "#d4edda", "#155724"
                    elif 0.5 <= score < 2:
                        verdict, verdict_msg, bg_color, text_color = "🟡 CAUTIOUS / STAGGERED BUY", "Indicators are leaning positive, but moderate overhead resistance remains. Accumulate slowly.", "#fff3cd", "#856404"
                    elif -1 <= score < 0.5:
                        verdict, verdict_msg, bg_color, text_color = "⚪ HOLD / WATCH", "Signals are neutral or conflicting. No distinct mathematical edge exists right now.", "#e2e3e5", "#383d41"
                    else:
                        verdict, verdict_msg, bg_color, text_color = "🔴 NO - STAY OUT / SELL", "Highly bearish structural momentum. Elevated risk of structural drawdown.", "#f8d7da", "#721c24"

                    st.markdown(f"""
                    <div style="background-color:{bg_color}; padding:20px; border-radius:10px; border-left:8px solid {text_color}; margin-bottom:25px;">
                        <h3 style="margin:0; color:{text_color}; font-weight:bold;">CAN I INVEST NOW?: {verdict}</h3>
                        <p style="margin:5px 0 0 0; color:{text_color}; font-size:16px;">{verdict_msg} <i>(Engine Total Score: {score:+.1f})</i></p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"₹{latest_close:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
                    col2.metric("RSI (14-Day)", f"{latest_rsi:.1f}")
                    col3.metric("MACD Hist", f"{df['MACD_Hist'].iloc[-1]:.2f}")
                    col4.metric("Engine Score", f"{score:+.1f}")

                    st.markdown("---")
                    
                    # Custom Chart Interface
                    chart_col, text_col = st.columns([3, 1])
                    with chart_col:
                        st.subheader("Deep Technical Chart")
                        chart_df = df[-252:] if len(df) > 252 else df
                        
                        fig = go.Figure()
                        
                        # Base Candlestick Trace
                        fig.add_trace(go.Candlestick(x=chart_df.index,
                                        open=chart_df['Open'], high=chart_df['High'],
                                        low=chart_df['Low'], close=chart_df['Close'], name='Price'))
                        
                        # Bollinger Bands Visuals
                        if show_bollinger:
                            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Upper'], line=dict(color='rgba(173,216,230,0.4)', width=1), name='BB Upper'))
                            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Lower'], line=dict(color='rgba(173,216,230,0.4)', width=1), name='BB Lower', fill='tonexty', fillcolor='rgba(173,216,230,0.08)'))
                            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Mid'], line=dict(color='grey', width=1, dash='dash'), name='BB Middle'))
                        
                        # Fibonacci Retracement Levels Trace
                        if show_fib:
                            highest_high = chart_df['High'].max()
                            lowest_low = chart_df['Low'].min()
                            diff = highest_high - lowest_low
                            
                            levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
                            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
                            
                            for level, color in zip(levels, colors):
                                value = highest_high - (level * diff)
                                fig.add_trace(go.Scatter(
                                    x=[chart_df.index[0], chart_df.index[-1]],
                                    y=[value, value],
                                    mode="lines",
                                    line=dict(color=color, width=1, dash="dashdot"),
                                    name=f"Fib {level*100:.1f}%"
                                ))

                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_50'], line=dict(color='blue', width=1.5), name='50 SMA'))
                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_200'], line=dict(color='orange', width=1.5), name='200 SMA'))
                        
                        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=0, r=0, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                    with text_col:
                        st.subheader("Deep Technical Analysis")
                        for r in reasons:
                            st.write(r)
                            
                    # --- SCREENER.IN PROS & CONS VIEW ---
                    st.markdown("---")
                    st.subheader("📋 Screener.in Qualitative Sentiment Analysis")
                    if screener_data.get("success"):
                        pro_col, con_col = st.columns(2)
                        with pro_col:
                            st.markdown("##### 🟢 Strengths / Pros")
                            if screener_data.get("pros"):
                                for pro in screener_data["pros"]:
                                    st.write(f"✅ {pro}")
                            else:
                                st.write("Screener reports no immediate structural strengths.")
                        with con_col:
                            st.markdown("##### 🔴 Limitations / Cons")
                            if screener_data.get("cons"):
                                for con in screener_data["cons"]:
                                    st.write(f"⚠️ {con}")
                            else:
                                st.write("Screener reports no immediate structural concerns.")
                    else:
                        st.info(f"Screener.in profile details unavailable: {screener_data.get('error', 'Unknown Error')}")

                # ==========================================
                # TAB 2: HISTORICAL BACKTEST
                # ==========================================
                with tab2:
                    st.subheader("Historical Simulation Engine (5-Year Lookback)")
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        strategy_choice = st.selectbox("Select Strategy to Backtest:", ["SMA Crossover (50 vs 200)", "MACD Line Crossover"])
                    with b_col2:
                        initial_capital = st.number_input("Starting Capital (₹)", min_value=100, max_value=10000000, value=100000, step=1000)
                    
                    bt_df = df.dropna(subset=['SMA_200']).copy() if strategy_choice == "SMA Crossover (50 vs 200)" else df.dropna(subset=['Signal_Line']).copy()
                    
                    if bt_df.empty:
                        st.error("Insufficient historical data to execute backtest.")
                    else:
                        position = 0
                        cash = initial_capital
                        shares = 0
                        portfolio_history = []
                        trade_count = 0
                        
                        for date, row in bt_df.iterrows():
                            price = row['Close']
                            buy_cond = row['SMA_50'] > row['SMA_200'] if strategy_choice == "SMA Crossover (50 vs 200)" else row['MACD'] > row['Signal_Line']
                            sell_cond = row['SMA_50'] < row['SMA_200'] if strategy_choice == "SMA Crossover (50 vs 200)" else row['MACD'] < row['Signal_Line']
                                
                            if position == 0 and buy_cond:
                                shares = cash / price
                                cash = 0
                                position = 1
                                trade_count += 1
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
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Strategy Final Value", f"₹{final_strategy_val:,.2f}")
                        m2.metric("Strategy Total Return", f"{strat_return:.2f}%")
                        m3.metric("Buy & Hold Return", f"{bh_return:.2f}%")
                        m4.metric("Trades Executed", f"{trade_count}")
                        
                        st.markdown("---")
                        if final_strategy_val > final_bh_val:
                            st.success(f"🏆 **Victory!** {strategy_choice} beat buy-and-hold by **{strat_return - bh_return:.2f}%**.")
                        else:
                            st.warning(f"⚠️ **Market Beats Strategy.** Buying and holding would have yielded **{bh_return - strat_return:.2f}%** more.")
                        
                        equity_fig = go.Figure()
                        equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Strategy_Value'], line=dict(color='green', width=2), name='Strategy'))
                        equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Buy_Hold_Value'], line=dict(color='grey', width=1.5, dash='dash'), name='Buy & Hold'))
                        equity_fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Portfolio Value (₹)")
                        st.plotly_chart(equity_fig, use_container_width=True)

                # ==========================================
                # TAB 3: VOLATILITY & RISK METRICS
                # ==========================================
                with tab3:
                    st.subheader("Deep Risk Analysis Profile")
                    st.write("Assess historic standard deviation and drawback thresholds before deploying retail capital.")
                    
                    df['Daily_Return'] = df['Close'].pct_change()
                    daily_vol = df['Daily_Return'].std()
                    ann_vol = daily_vol * np.sqrt(252) * 100
                    
                    rolling_max = df['Close'].cummax()
                    drawdowns = (df['Close'] - rolling_max) / rolling_max
                    max_drawdown = drawdowns.min() * 100
                    
                    rc1, rc2, rc3 = st.columns(3)
                    vol_class = "Low Risk" if ann_vol < 15 else "Moderate Risk" if ann_vol < 30 else "High Risk"
                    rc1.metric("Annualized Volatility", f"{ann_vol:.2f}%", vol_class, delta_color="off")
                    rc2.metric("Maximum Historical Drawdown", f"{max_drawdown:.2f}%", "Worst-case drop from absolute peak", delta_color="inverse")
                    rc3.metric("Standard Deviation of Price", f"₹{df['Close'].std():.2f}")

                # ==========================================
                # TAB 4: INTRINSIC & FAIR VALUE CALCULATORS
                # ==========================================
                with tab4:
                    st.subheader("💎 Valuation Models & Intrinsic Calculations")
                    st.info("💡 **Bonus/Split Adjustments:** Automated databases can lag by several weeks updating EPS and Book Value post-corporate actions. Use the configuration box below to align metrics manually if they appear unadjusted.")

                    screener_ratios = screener_data.get("ratios", {}) if screener_data.get("success") else {}
                    
                    # 1. Extract automated variables
                    auto_pe = extract_ratio_value(screener_ratios, ["stock_p_e", "pe", "p_e"])
                    auto_bv = extract_ratio_value(screener_ratios, ["book_value", "bvps"])
                    auto_mcap = extract_ratio_value(screener_ratios, ["market_cap", "m_cap"])
                    auto_pb = extract_ratio_value(screener_ratios, ["price_to_book", "pb", "p_b"])
                    
                    # Fallback default values (Calibrated for RELIANCE's post-bonus metrics)
                    fallback_pe = 23.1 if "RELIANCE" in yfinance_ticker else 20.0
                    fallback_bv = 668.0 if "RELIANCE" in yfinance_ticker else 150.0
                    
                    # Resolve auto values
                    if not auto_pe and stock_info.get('trailingPE'):
                        auto_pe = float(stock_info.get('trailingPE'))
                    if not auto_bv and stock_info.get('bookValue'):
                        auto_bv = float(stock_info.get('bookValue'))

                    # --- MANUAL OVERRIDE INTERFACE PANEL ---
                    st.markdown("### 🛠️ Calibration Panel")
                    use_override = st.checkbox(
                        "**Enable Manual Calibration Override** (Overrules APIs to resolve stale metrics)", 
                        value=True if not screener_data.get("success") else False
                    )
                    
                    col_ov1, col_ov2, col_ov3 = st.columns(3)
                    
                    if use_override:
                        pe_input_val = col_ov1.number_input("P/E Ratio", value=float(auto_pe if auto_pe else fallback_pe), step=0.1, format="%.2f")
                        bvps_input_val = col_ov2.number_input("Book Value Per Share (BVPS)", value=float(auto_bv if auto_bv else fallback_bv), step=1.0, format="%.2f")
                        cmp_input_val = col_ov3.number_input("Current Market Price (CMP)", value=float(latest_close), step=1.0, format="%.2f")
                        
                        # Dynamic calculations based on manual input overrides
                        cmp = cmp_input_val
                        derived_bvps = bvps_input_val
                        calculated_eps = cmp / pe_input_val if pe_input_val > 0 else 1.0
                    else:
                        # Fallback to pure automated discovery
                        cmp = float(latest_close)
                        if auto_pe:
                            calculated_eps = cmp / auto_pe
                        else:
                            calculated_eps = float(stock_info.get('trailingEps', 25.0)) if stock_info.get('trailingEps') else 25.0
                            
                        if auto_bv:
                            derived_bvps = auto_bv
                        else:
                            derived_pb = auto_pb if auto_pb else (float(stock_info.get('priceToBook')) if stock_info.get('priceToBook') else None)
                            derived_bvps = cmp / derived_pb if derived_pb else (float(stock_info.get('bookValue', 150.0)) if stock_info.get('bookValue') else 150.0)
                        
                        col_ov1.metric("Current P/E Ratio (Auto)", f"{cmp/calculated_eps:.2f}" if calculated_eps > 0 else "N/A")
                        col_ov2.metric("BVPS (Auto)", f"₹{derived_bvps:,.2f}")
                        col_ov3.metric("CMP (Auto)", f"₹{cmp:,.2f}")
                        
                        st.caption(f"Status: Using Automated Pipeline. (Screener.in Connection: {'✅ Online' if screener_data.get('success') else '❌ Offline/Blocked'})")

                    # Recalculate share quantities
                    derived_mcap = auto_mcap * 10000000 if auto_mcap else float(stock_info.get('marketCap', 100000000000.0)) if stock_info.get('marketCap') else 100000000000.0
                    derived_shares = derived_mcap / cmp

                    st.markdown("---")
                    calc_col1, calc_col2 = st.columns(2)

                    # --- MODEL A: BENJAMIN GRAHAM VALUE ---
                    with calc_col1:
                        st.markdown("### 🏛️ Graham's Fair Value Model")
                        st.write("Assesses asset-to-earnings consistency for mature companies.")
                        
                        eps_display = st.number_input("Earnings Per Share (EPS)", value=float(calculated_eps), format="%.2f")
                        bvps_display = st.number_input("Book Value Per Share (BVPS)", value=float(derived_bvps), format="%.2f")
                        
                        if eps_display > 0 and bvps_display > 0:
                            graham_fair_value = np.sqrt(22.5 * eps_display * bvps_display)
                            graham_mos = ((graham_fair_value - cmp) / graham_fair_value) * 100
                            
                            st.markdown(f"#### Calculated Graham Value: **₹{graham_fair_value:,.2f}**")
                            if graham_mos > 20:
                                st.success(f"✅ **Undervalued:** Stock trades at a **{graham_mos:.1f}%** discount to Graham Value.")
                            elif 0 <= graham_mos <= 20:
                                st.info(f"💛 **Fairly Valued:** Margin of Safety is tight (**{graham_mos:.1f}%**).")
                            else:
                                st.warning(f"⚠️ **Overvalued:** Trades at a **{abs(graham_mos):.1f}%** premium.")
                        else:
                            st.warning("Graham model requires positive Earnings and positive Book Value metrics.")

                    # --- MODEL B: DISCOUNTED CASH FLOW (DCF) ---
                    with calc_col2:
                        st.markdown("### 🌀 Discounted Cash Flow (DCF) Model")
                        st.write("Project and discount company cash flows.")
                        
                        fcf_input = st.number_input("Free Cash Flow (FCF in ₹)", value=float(derived_mcap * 0.05))
                        shares_input = st.number_input("Shares Outstanding", value=float(derived_shares))
                        
                        g_rate = st.slider("Expected Growth Rate (g) (%)", min_value=-5.0, max_value=40.0, value=12.0, step=0.5)
                        d_rate = st.slider("Discount Rate (r) (%)", min_value=5.0, max_value=25.0, value=11.0, step=0.5)
                        t_rate = st.slider("Terminal Growth Rate (gn) (%)", min_value=1.0, max_value=8.0, value=4.5, step=0.1)

                        if d_rate <= t_rate:
                            st.error("Error: Discount Rate (r) must be strictly higher than the Terminal Growth Rate (gn).")
                        else:
                            projected_fcf = []
                            discount_factors = []
                            present_values = []
                            
                            temp_fcf = fcf_input
                            for year in range(1, 6):
                                temp_fcf = temp_fcf * (1 + (g_rate / 100))
                                projected_fcf.append(temp_fcf)
                                
                                discount_factor = 1 / ((1 + (d_rate / 100)) ** year)
                                discount_factors.append(discount_factor)
                                present_values.append(temp_fcf * discount_factor)
                                
                            sum_pv_fcf = sum(present_values)
                            terminal_value = (projected_fcf[-1] * (1 + (t_rate / 100))) / ((d_rate - t_rate) / 100)
                            pv_terminal_value = terminal_value / ((1 + (d_rate / 100)) ** 5)
                            
                            total_intrinsic_val = sum_pv_fcf + pv_terminal_value
                            dcf_fair_value = total_intrinsic_val / shares_input
                            dcf_mos = ((dcf_fair_value - cmp) / dcf_fair_value) * 100
                            
                            st.markdown(f"#### Calculated DCF Fair Value: **₹{dcf_fair_value:,.2f}**")
                            if dcf_mos > 20:
                                st.success(f"✅ **Undervalued:** Trading at a **{dcf_mos:.1f}%** Margin of Safety.")
                            elif 0 <= dcf_mos <= 20:
                                st.info(f"💛 **Fairly Valued:** Margin of Safety is minor (**{dcf_mos:.1f}%**).")
                            else:
                                st.warning(f"⚠️ **Overvalued:** Price is **{abs(dcf_mos):.1f}%** above estimated DCF value.")

        except Exception as e:
            st.error(f"Could not load data for analysis: {e}")
