import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & THEME SETUP
# ==========================================
st.set_page_config(page_title="Indian Quant Deep-Dive Dashboard", layout="wide")
st.title("📊 Indian Quant Trading & Deep-Dive Dashboard")

# ==========================================
# 2. MOBILE-FIRST TOP-LEVEL NAVIGATION & SEARCH
# ==========================================
st.markdown("---")
col_search, col_blank = st.columns([2, 2])
with col_search:
    raw_ticker_input = st.text_input(
        "🔍 Search Indian Ticker (e.g., RELIANCE, TCS, INFY, HDFCBANK, ^NSEI):", 
        "RELIANCE"
    ).upper().strip()

# Exchange Suffixing Logic
if not raw_ticker_input.startswith('^') and '.' not in raw_ticker_input:
    yfinance_ticker = f"{raw_ticker_input}.NS"
else:
    yfinance_ticker = raw_ticker_input

# Move layout configurations to sidebar to keep screen clean
st.sidebar.header("Chart Settings")
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

@st.cache_data(ttl=1800)
def load_data(ticker, period="5y"):
    df = None
    info = {}
    news = []
    try:
        session = requests.Session()
        session.headers.update(get_robust_headers())
        ticker_obj = yf.Ticker(ticker, session=session)
        df = ticker_obj.history(period=period)
        if df.empty:
            raise ValueError("Empty response from Yahoo Finance.")
    except Exception as yf_error:
        st.sidebar.warning("Yahoo API limit reached. Fetching prices from Stooq fallback...")
        try:
            df = fetch_from_stooq(ticker)
        except Exception as stooq_error:
            raise RuntimeError(f"Data stream disconnected: {yf_error}. Backup failed: {stooq_error}.")
            
    try:
        session = requests.Session()
        session.headers.update(get_robust_headers())
        ticker_obj = yf.Ticker(ticker, session=session)
        info = ticker_obj.info
        news = ticker_obj.news
        if not isinstance(info, dict):
            info = {}
        if not isinstance(news, list):
            news = []
    except Exception:
        info = {}
        news = []
        
    return df, info, news

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
    
    # Average True Range (ATR) for Advanced Volatility Tool
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift())
    low_cp = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    return df

# ==========================================
# 4. RUN SYSTEM CALCULATIONS & TABS
# ==========================================
if yfinance_ticker:
    with st.spinner("Compiling cross-asset indicators and loading news feeds..."):
        try:
            raw_df, stock_info, stock_news = load_data(yfinance_ticker, "5y")
            
            if raw_df is None or raw_df.empty:
                st.error("Invalid ticker string or matching asset not found.")
            else:
                df = calculate_indicators(raw_df)
                
                # Tabbed Interface Layout
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🔍 Technical Radar", 
                    "🧬 Advanced Quant Factors",
                    "📐 Volatility & Pivot Target Levels",
                    "💎 Valuation Models",
                    "🕒 Engine Backtester"
                ])
                
                # Baseline metrics
                latest_close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
                price_change = latest_close - prev_close
                pct_change = (price_change / prev_close) * 100 if prev_close != 0 else 0
                
                # ==========================================
                # TAB 1: TECHNICAL RADAR & NEWS FEEDS
                # ==========================================
                with tab1:
                    latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50.0
                    latest_sma50 = df['SMA_50'].iloc[-1]
                    latest_sma200 = df['SMA_200'].iloc[-1]
                    latest_macd = df['MACD'].iloc[-1]
                    latest_signal = df['Signal_Line'].iloc[-1]
                    
                    score = 0
                    reasons = []
                    
                    if not pd.isna(latest_sma50) and not pd.isna(latest_sma200):
                        if latest_sma50 > latest_sma200:
                            score += 1
                            reasons.append("🟢 **Long-Term Trend (SMA):** Bullish Structure (50 SMA > 200 SMA)")
                        else:
                            score -= 1
                            reasons.append("🔴 **Long-Term Trend (SMA):** Bearish Structure (50 SMA < 200 SMA)")
                    
                    if not pd.isna(latest_macd) and not pd.isna(latest_signal):
                        if latest_macd > latest_signal:
                            score += 1
                            reasons.append("🟢 **Short-Term Momentum (MACD):** Bullish crossover active")
                        else:
                            score -= 1
                            reasons.append("🔴 **Short-Term Momentum (MACD):** Bearish compression under signal line")
                            
                    if latest_rsi < 30:
                        score += 1.5
                        reasons.append(f"🟢 **Exhaustion (RSI):** Oversold Condition ({latest_rsi:.1f})")
                    elif latest_rsi > 70:
                        score -= 1.5
                        reasons.append(f"🔴 **Exhaustion (RSI):** Overbought Condition ({latest_rsi:.1f})")

                    # Formulate Investment Verdict banner
                    if score >= 2:
                        verdict, bg_color, text_color = "HIGH CONVICTION BUY", "#d4edda", "#155724"
                    elif 0.5 <= score < 2:
                        verdict, bg_color, text_color = "CAUTIOUS ACCUMULATION", "#fff3cd", "#856404"
                    elif -1 <= score < 0.5:
                        verdict, bg_color, text_color = "NEUTRAL HOLD / WATCH", "#e2e3e5", "#383d41"
                    else:
                        verdict, bg_color, text_color = "HIGH RISK AVOID / LIQUIDATE", "#f8d7da", "#721c24"

                    st.markdown(f"""
                    <div style="background-color:{bg_color}; padding:15px; border-radius:8px; border-left:6px solid {text_color}; margin-bottom:15px;">
                        <h4 style="margin:0; color:{text_color}; font-weight:bold;">STRATEGY VERDICT: {verdict} (Score: {score:+.1f})</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Value", f"₹{latest_close:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
                    col2.metric("Relative Strength Index", f"{latest_rsi:.1f}")
                    col3.metric("MACD Divergence", f"{df['MACD_Hist'].iloc[-1]:.2f}")
                    col4.metric("Signal Matrix Weight", f"{score:+.1f}")

                    st.markdown("---")
                    chart_col, sidebar_news_col = st.columns([3, 1.2])
                    
                    with chart_col:
                        chart_df = df[-252:] if len(df) > 252 else df
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='Price Line'))
                        
                        if show_bollinger:
                            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Upper'], line=dict(color='rgba(173,216,230,0.4)', width=1), name='BB Upper'))
                            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Lower'], line=dict(color='rgba(173,216,230,0.4)', width=1), name='BB Lower', fill='tonexty', fillcolor='rgba(173,216,230,0.04)'))
                        
                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_50'], line=dict(color='blue', width=1.2), name='50 SMA'))
                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_200'], line=dict(color='orange', width=1.2), name='200 SMA'))
                        fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0, r=0, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                    with sidebar_news_col:
                        st.subheader("📰 Real-time News Stream")
                        if stock_news:
                            for item in stock_news[:5]:
                                pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%d %b %Y')
                                st.markdown(f"""
                                **[{item.get('title')}]({item.get('link')})**  
                                <small style='color:gray;'>Publisher: {item.get('publisher')} | {pub_time}</small>
                                ---
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No active specific market headlines found for this asset index.")

                # ==========================================
                # TAB 2: ADVANCED QUANT FUNDAMENTAL FACTORS (NEW ADVANCED TOOL)
                # ==========================================
                with tab2:
                    st.subheader("🧬 Multi-Factor Fundamental Quality Matrix")
                    st.write("Deep corporate accounting factors retrieved live from core balance sheets via Yahoo Engine.")
                    
                    # Extract variables safely
                    roe = stock_info.get('returnOnEquity')
                    roa = stock_info.get('returnOnAssets')
                    debt_to_equity = stock_info.get('debtToEquity')
                    current_ratio = stock_info.get('currentRatio')
                    operating_margin = stock_info.get('operatingMargins')
                    beta_val = stock_info.get('beta')
                    peg_ratio = stock_info.get('pegRatio')
                    
                    def fmt_pct(val): return f"{val * 100:.2f}%" if val is not None else "Data Missing"
                    def fmt_num(val, mult=1): return f"{val/mult:.2f}" if val is not None else "Data Missing"

                    # Build Scannable Factor Evaluation Table Matrix
                    factor_data = {
                        "Quant Performance Factor": ["Return on Equity (ROE)", "Return on Assets (ROA)", "Operating Profit Margin", "Debt-to-Equity Ratio", "Current Solvency Ratio", "Systematic Asset Volatility (Beta)", "PEG Valuation Expansion Multiple"],
                        "Current Asset Reading": [fmt_pct(roe), fmt_pct(roa), fmt_pct(operating_margin), fmt_num(debt_to_equity, 100), fmt_num(current_ratio), fmt_num(beta_val), fmt_num(peg_ratio)],
                        "Risk Benchmark Target Threshold": ["> 15.00% Optimal", "> 8.00% Optimal", "> 12.00% High Efficiency", "< 1.00 Low Leverage Risk", "> 1.20 Cash Soundness", "< 1.00 Low Beta Defensive", "< 1.50 Fair Pricing Growth"]
                    }
                    st.table(pd.DataFrame(factor_data))
                    
                    # Core Health Diagnostics Warning Cards
                    st.markdown("### 🔍 Risk Allocation Signals")
                    c1, c2 = st.columns(2)
                    with c1:
                        if debt_to_equity and debt_to_equity > 100:
                            st.warning("⚠️ **Leverage Alert:** Corporate Debt-to-Equity exceeds 1.0. Balance sheet is heavily leveraged.")
                        else:
                            st.success("✅ **Balance Sheet Health:** Long-term liability metrics are securely structured.")
                    with c2:
                        if roe and roe > 0.15:
                            st.success("✅ **Profit Machine Engine:** Return on Equity meets tier-1 institution targets (>15%).")
                        else:
                            st.info("ℹ️ **Capital Efficiency Warning:** Asset yields moderate capital velocity returns on investor equity.")

                # ==========================================
                # TAB 3: VOLATILITY & PIVOT TARGETS (NEW ADVANCED TOOL)
                # ==========================================
                with tab3:
                    st.subheader("📐 Intraday Pivot Target Framework & Volatility Ranges")
                    st.write("Mathematical execution targets calculated using previous complete close distributions.")

                    # Calculate Pivot Points from last historical trading metrics row
                    last_day = df.iloc[-1]
                    h_val = last_day['High']
                    l_val = last_day['Low']
                    c_val = last_day['Close']
                    atr_val = last_day['ATR'] if 'ATR' in df.columns else (h_val - l_val)
                    
                    pivot = (h_val + l_val + c_val) / 3.0
                    r1 = (2 * pivot) - l_val
                    s1 = (2 * pivot) - h_val
                    r2 = pivot + (h_val - l_val)
                    s2 = pivot - (h_val - l_val)
                    r3 = h_val + 2 * (pivot - l_val)
                    s3 = l_val - 2 * (h_val - pivot)

                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        st.markdown("##### 📈 System Overhead Resistance Targets")
                        st.markdown(f"""
                        > **Resistance 3 (R3 Profit Target Matrix):** `₹{r3:,.2f}`  
                        > **Resistance 2 (R2 Macro Ceiling):** `₹{r2:,.2f}`  
                        > **Resistance 1 (R1 Minor Breakout):** `₹{r1:,.2f}`
                        """)
                        st.metric("🎯 Calculated Structural Central Pivot", f"₹{pivot:,.2f}")
                        
                    with p_col2:
                        st.markdown("##### 📉 Downside Support Target Infrastructure")
                        st.markdown(f"""
                        > **Support 1 (S1 Liquidity Pool):** `₹{s1:,.2f}`  
                        > **Support 2 (S2 System Floor):** `₹{s2:,.2f}`  
                        > **Support 3 (S3 Deep Value Rebound):** `₹{s3:,.2f}`
                        """)
                        st.metric("📊 14-Day True Average Volatility Range (ATR)", f"₹{atr_val:.2f}")

                # ==========================================
                # TAB 4: INTRINSIC & FAIR VALUE CALCULATORS
                # ==========================================
                with tab4:
                    st.subheader("💎 Valuation Models & Intrinsic Calculations")
                    
                    yf_pe = stock_info.get('trailingPE', 20.0)
                    yf_eps = stock_info.get('trailingEps', 25.0)
                    yf_bvps = stock_info.get('bookValue', 150.0)
                    yf_mcap = stock_info.get('marketCap', latest_close * 1000000)
                    yf_fcf = stock_info.get('freeCashflow', yf_mcap * 0.05)
                    yf_shares = stock_info.get('sharesOutstanding', yf_mcap / latest_close)

                    # Dynamic protection fallbacks if fields are parsed empty from Yahoo core dictionary
                    if not yf_eps and yf_pe > 0: yf_eps = latest_close / yf_pe
                    if not yf_pe and yf_eps > 0: yf_pe = latest_close / yf_eps

                    st.markdown("### 📊 Live Multiples Pulled From Engine")
                    h1, h2, h3 = st.columns(3)
                    h1.metric("Engine Stream P/E Multiple", f"{yf_pe:.2f}")
                    h2.metric("Engine Book Value Per Share", f"₹{yf_bvps:,.2f}")
                    h3.metric("Engine Normalised EPS", f"₹{yf_eps:.2f}")

                    st.markdown("---")
                    calc_col1, calc_col2 = st.columns(2)

                    with calc_col1:
                        st.markdown("### 🏛️ Graham's Fair Value Model")
                        eps_input = st.number_input("Earnings Per Share (EPS)", value=float(yf_eps), format="%.2f")
                        bvps_input = st.number_input("Book Value Per Share (BVPS)", value=float(yf_bvps), format="%.2f")
                        
                        if eps_input > 0 and bvps_input > 0:
                            graham_fair_value = np.sqrt(22.5 * eps_input * bvps_input)
                            graham_mos = ((graham_fair_value - latest_close) / graham_fair_value) * 100
                            st.markdown(f"#### Calculated Graham Value: **₹{graham_fair_value:,.2f}**")
                            if graham_mos > 20:
                                st.success(f"✅ **Undervalued:** Trading at a **{graham_mos:.1f}%** structural discount.")
                            elif 0 <= graham_mos <= 20:
                                st.info(f"💛 **Fair Value:** Margins match fair valuation limits (**{graham_mos:.1f}%**).")
                            else:
                                st.warning(f"⚠️ **Overvalued:** Price is **{abs(graham_mos):.1f}%** higher than model limits.")

                    with calc_col2:
                        st.markdown("### 🌀 Discounted Cash Flow (DCF) Model")
                        fcf_input = st.number_input("Free Cash Flow (FCF in ₹)", value=float(yf_fcf))
                        shares_input = st.number_input("Shares Outstanding", value=float(yf_shares))
                        
                        g_rate = st.slider("Expected Growth Rate (g) (%)", min_value=-5.0, max_value=40.0, value=12.0, step=0.5)
                        d_rate = st.slider("Discount Rate (r) (%)", min_value=5.0, max_value=25.0, value=11.0, step=0.5)
                        t_rate = st.slider("Terminal Growth Rate (gn) (%)", min_value=1.0, max_value=8.0, value=4.5, step=0.1)

                        if d_rate <= t_rate:
                            st.error("Execution Blunder: Discount Rate (r) must be configured higher than Terminal Rate.")
                        else:
                            projected_fcf = []
                            present_values = []
                            temp_fcf = fcf_input
                            for year in range(1, 6):
                                temp_fcf = temp_fcf * (1 + (g_rate / 100))
                                discount_factor = 1 / ((1 + (d_rate / 100)) ** year)
                                present_values.append(temp_fcf * discount_factor)
                                
                            sum_pv_fcf = sum(present_values)
                            terminal_value = (temp_fcf * (1 + (t_rate / 100))) / ((d_rate - t_rate) / 100)
                            pv_terminal_value = terminal_value / ((1 + (d_rate / 100)) ** 5)
                            
                            dcf_fair_value = (sum_pv_fcf + pv_terminal_value) / shares_input
                            dcf_mos = ((dcf_fair_value - latest_close) / dcf_fair_value) * 100
                            
                            st.markdown(f"#### Calculated DCF Fair Value: **₹{dcf_fair_value:,.2f}**")
                            if dcf_mos > 20:
                                st.success(f"✅ **Intrinsic Disconnect:** **{dcf_mos:.1f}%** margin discount calculated.")
                            else:
                                st.warning(f"⚠️ **Premium Pricing:** Value trades **{abs(dcf_mos):.1f}%** over intrinsic models.")

                # ==========================================
                # TAB 5: ENGINE BACKTESTER
                # ==========================================
                with tab5:
                    st.subheader("Historical Strategy Simulation Run")
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        strategy_choice = st.selectbox("Select Core Vector:", ["SMA Crossover (50 vs 200)", "MACD Line Crossover"])
                    with b_col2:
                        initial_capital = st.number_input("Starting Capital Allocation (₹)", min_value=100, max_value=10000000, value=100000)
                    
                    bt_df = df.dropna(subset=['SMA_200']).copy() if strategy_choice == "SMA Crossover (50 vs 200)" else df.dropna(subset=['Signal_Line']).copy()
                    
                    if bt_df.empty:
                        st.error("Insufficient historical lookback range available to execute logic calculations.")
                    else:
                        position, cash, shares, portfolio_history, trade_count = 0, initial_capital, 0, [], 0
                        
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
                        bt_df['Buy_Hold_Value'] = (initial_capital / bt_df['Close'].iloc[0]) * bt_df['Close']
                        
                        final_strategy_val = bt_df['Strategy_Value'].iloc[-1]
                        final_bh_val = bt_df['Buy_Hold_Value'].iloc[-1]
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Strategy Terminal Worth", f"₹{final_strategy_val:,.2f}")
                        m2.metric("Strategy Pure ROI", f"{((final_strategy_val - initial_capital)/initial_capital)*100:.2f}%")
                        m3.metric("Buy & Hold Baseline Return", f"{((final_bh_val - initial_capital)/initial_capital)*100:.2f}%")
                        
                        equity_fig = go.Figure()
                        equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Strategy_Value'], line=dict(color='green', width=2), name='Strategy Curves'))
                        equity_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Buy_Hold_Value'], line=dict(color='grey', width=1, dash='dash'), name='Benchmark Curves'))
                        st.plotly_chart(equity_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Global runtime error compiling analytics frames: {e}")
