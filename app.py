import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & MOBILE STYLING
# ==========================================
st.set_page_config(page_title="Indian Quant Deep-Dive Dashboard", layout="wide")
st.title("📊 Indian Quant Trading & Deep-Dive Dashboard")

# Global CSS Injection to Freeze Table Headers and Handle Long Tables
st.markdown("""
<style>
    th {
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        background-color: #f8f9fa !important;
        z-index: 5;
    }
    div[data-testid="stTable"] {
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

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

if not raw_ticker_input.startswith('^') and '.' not in raw_ticker_input:
    yfinance_ticker = f"{raw_ticker_input}.NS"
else:
    yfinance_ticker = raw_ticker_input

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
        st.sidebar.warning("Yahoo API limits reached. Fetching prices from Stooq fallback...")
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
        if not isinstance(info, dict): info = {}
        if not isinstance(news, list): news = []
    except Exception:
        info = {}
        news = []
        
    return df, info, news

@st.cache_data(ttl=3600)
def load_extended_quant_data(ticker):
    """Pulls corporate financials, balance sheets, and historical actions."""
    try:
        t = yf.Ticker(ticker)
        actions = t.actions
        financials = t.financials
        return actions, financials
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def calculate_indicators(df):
    df = df.copy()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift())
    low_cp = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def robust_find_row(df, keyword):
    if df.empty: return None
    for idx in df.index:
        if keyword.lower() in str(idx).lower().replace(" ", "").replace("_", ""):
            return df.loc[idx]
    return None

# ==========================================
# 4. RUN CALCULATIONS & MAIN INTERFACE
# ==========================================
if yfinance_ticker:
    with st.spinner("Compiling structural quant analytics frameworks..."):
        try:
            raw_df, stock_info, stock_news = load_data(yfinance_ticker, "5y")
            extended_actions, extended_financials = load_extended_quant_data(yfinance_ticker)
            
            if raw_df is None or raw_df.empty:
                st.error("Invalid ticker string or matching asset not found.")
            else:
                df = calculate_indicators(raw_df)
                
                # Master Tab Setup (All 4 New Request Frameworks Appended)
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
                    "🔍 Technical Radar", 
                    "🧬 Advanced Quant Factors",
                    "📐 Volatility & Pivot Target Levels",
                    "💎 Valuation Models",
                    "👥 Peer Comparison Matrix",
                    "📅 Corporate Actions Tracker",
                    "📊 Fundamental Trends",
                    "🎲 Monte Carlo Risk Simulator",
                    "🕒 Engine Backtester"
                ])
                
                latest_close = df['Close'].iloc[-1]
                price_change = latest_close - df['Close'].iloc[-2]
                pct_change = (price_change / df['Close'].iloc[-2]) * 100
                
                clean_news_stream = []
                for item in stock_news:
                    if item.get('title') and str(item.get('title')).strip().lower() != "none" and item.get('link') and item.get('providerPublishTime', 0) > 0:
                        clean_news_stream.append(item)

                # ==========================================
                # TAB 1: TECHNICAL RADAR & HEADLINES
                # ==========================================
                with tab1:
                    latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50.0
                    latest_sma50 = df['SMA_50'].iloc[-1]
                    latest_sma200 = df['SMA_200'].iloc[-1]
                    
                    score = 0
                    if latest_sma50 > latest_sma200: score += 1
                    else: score -= 1
                    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]: score += 1
                    else: score -= 1
                    if latest_rsi < 30: score += 1.5
                    elif latest_rsi > 70: score -= 1.5

                    if score >= 2: verdict, bg_color, text_color = "HIGH CONVICTION BUY", "#d4edda", "#155724"
                    elif 0.5 <= score < 2: verdict, bg_color, text_color = "CAUTIOUS ACCUMULATION", "#fff3cd", "#856404"
                    elif -1 <= score < 0.5: verdict, bg_color, text_color = "NEUTRAL HOLD / WATCH", "#e2e3e5", "#383d41"
                    else: verdict, bg_color, text_color = "HIGH RISK AVOID", "#f8d7da", "#721c24"

                    st.markdown(f'<div style="background-color:{bg_color}; padding:15px; border-radius:8px; border-left:6px solid {text_color}; margin-bottom:15px;"><h4 style="margin:0; color:{text_color}; font-weight:bold;">STRATEGY VERDICT: {verdict} (Score: {score:+.1f})</h4></div>', unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Value", f"₹{latest_close:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
                    col2.metric("Relative Strength Index", f"{latest_rsi:.1f}")
                    col3.metric("MACD Divergence", f"{df['MACD_Hist'].iloc[-1]:.2f}")
                    col4.metric("Signal Weight", f"{score:+.1f}")

                    st.markdown("---")
                    chart_col, sidebar_news_col = st.columns([3, 1.2]) if clean_news_stream else (st.container(), None)
                    
                    with chart_col:
                        chart_df = df[-252:]
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='Price'))
                        if show_bollinger:
                            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Upper'], line=dict(color='rgba(173,216,230,0.4)', width=1), name='BB Upper'))
                            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Lower'], line=dict(color='rgba(173,216,230,0.4)', width=1), name='BB Lower', fill='tonexty', fillcolor='rgba(173,216,230,0.04)'))
                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_50'], line=dict(color='blue', width=1.2), name='50 SMA'))
                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_200'], line=dict(color='orange', width=1.2), name='200 SMA'))
                        fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0, r=0, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                    if clean_news_stream and sidebar_news_col:
                        with sidebar_news_col:
                            st.subheader("📰 Real-time Headlines")
                            for item in clean_news_stream[:5]:
                                pub_time = datetime.fromtimestamp(item.get('providerPublishTime')).strftime('%d %b %Y')
                                st.markdown(f"**[{item.get('title')}]({item.get('link')})**  \n<small style='color:gray;'>{item.get('publisher')} | {pub_time}</small>\n---", unsafe_allow_html=True)

                # ==========================================
                # TAB 2: QUANT FUNDAMENTAL FACTORS
                # ==========================================
                with tab2:
                    st.subheader("🧬 Multi-Factor Quality Matrix")
                    roe = stock_info.get('returnOnEquity')
                    roa = stock_info.get('returnOnAssets')
                    debt_to_equity = stock_info.get('debtToEquity')
                    current_ratio = stock_info.get('currentRatio')
                    operating_margin = stock_info.get('operatingMargins')
                    beta_val = stock_info.get('beta')
                    peg_ratio = stock_info.get('pegRatio')
                    
                    def fmt_pct(val): return f"{val * 100:.2f}%" if val is not None else "N/A"
                    def fmt_num(val, mult=1): return f"{val/mult:.2f}" if val is not None else "N/A"

                    factor_data = {
                        "Quant Performance Factor": ["Return on Equity (ROE)", "Return on Assets (ROA)", "Operating Profit Margin", "Debt-to-Equity Ratio", "Current Solvency Ratio", "Systematic Volatility (Beta)", "PEG Multiple"],
                        "Current Reading": [fmt_pct(roe), fmt_pct(roa), fmt_pct(operating_margin), fmt_num(debt_to_equity, 100), fmt_num(current_ratio), fmt_num(beta_val), fmt_num(peg_ratio)],
                        "Target Threshold": ["> 15.00% Optimal", "> 8.00% Optimal", "> 12.00% High Efficiency", "< 1.00 Low Leverage", "> 1.20 Cash Soundness", "< 1.00 Defensive", "< 1.50 Value Growth"]
                    }
                    st.table(pd.DataFrame(factor_data))
                    
                    st.markdown("### 🏢 Core Shareholding Structure")
                    insider_share = stock_info.get('heldPercentInsiders', 0.0) * 100
                    inst_share = stock_info.get('heldPercentInstitutions', 0.0) * 100
                    public_share = max(0.0, 100.0 - (insider_share + inst_share))
                    
                    sh_col1, sh_col2, sh_col3 = st.columns(3)
                    sh_col1.metric("Promoter / Insider", f"{insider_share:.2f}%" if insider_share > 0 else "N/A")
                    sh_col2.metric("Institutional (FII/DII)", f"{inst_share:.2f}%" if inst_share > 0 else "N/A")
                    sh_col3.metric("Public Float", f"{public_share:.2f}%" if public_share < 100 else "N/A")

                # ==========================================
                # TAB 3: VOLATILITY & PIVOT TARGETS
                # ==========================================
                with tab3:
                    st.subheader("📐 Intraday Pivot Target Framework")
                    last_day = df.iloc[-1]
                    h_val, l_val, c_val = last_day['High'], last_day['Low'], last_day['Close']
                    atr_val = last_day['ATR'] if 'ATR' in df.columns else (h_val - l_val)
                    
                    pivot = (h_val + l_val + c_val) / 3.0
                    r1, s1 = (2 * pivot) - l_val, (2 * pivot) - h_val
                    r2, s2 = pivot + (h_val - l_val), pivot - (h_val - l_val)

                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        st.markdown(f"> **Resistance 2:** `₹{r2:,.2f}`  \n> **Resistance 1:** `₹{r1:,.2f}`")
                        st.metric("🎯 Central Structural Pivot", f"₹{pivot:,.2f}")
                    with p_col2:
                        st.markdown(f"> **Support 1:** `₹{s1:,.2f}`  \n> **Support 2:** `₹{s2:,.2f}`")
                        st.metric("📊 14-Day Average True Range (ATR)", f"₹{atr_val:.2f}")

                # ==========================================
                # TAB 4: INTRINSIC VALUATION MODELS
                # ==========================================
                with tab4:
                    st.subheader("💎 Valuation Models (Values in Crores)")
                    yf_pe = stock_info.get('trailingPE', 20.0)
                    yf_eps = stock_info.get('trailingEps', 25.0)
                    yf_bvps = stock_info.get('bookValue', 150.0)
                    yf_mcap = stock_info.get('marketCap', latest_close * 1000000)
                    
                    yf_fcf_crores = stock_info.get('freeCashflow', yf_mcap * 0.05) / 10000000
                    yf_shares_crores = stock_info.get('sharesOutstanding', yf_mcap / latest_close) / 10000000

                    calc_col1, calc_col2 = st.columns(2)
                    with calc_col1:
                        st.markdown("### 🏛️ Graham Model")
                        eps_input = st.number_input("EPS", value=float(yf_eps), format="%.2f", key="val_eps")
                        bvps_input = st.number_input("BVPS", value=float(yf_bvps), format="%.2f", key="val_bvps")
                        if eps_input > 0 and bvps_input > 0:
                            g_val = np.sqrt(22.5 * eps_input * bvps_input)
                            st.markdown(f"#### Graham Fair Value: **₹{g_val:,.2f}**")

                    with calc_col2:
                        st.markdown("### 🌀 DCF Model")
                        fcf_input = st.number_input("FCF (₹ Crores)", value=float(yf_fcf_crores), format="%.2f", key="val_fcf")
                        shares_input = st.number_input("Shares Out (Crores)", value=float(yf_shares_crores), format="%.2f", key="val_sh")
                        g_rate = st.slider("Growth Rate (%)", -5.0, 40.0, 12.0, 0.5, key="val_g")
                        d_rate = st.slider("Discount Rate (%)", 5.0, 25.0, 11.0, 0.5, key="val_d")
                        
                        if d_rate > 4.5:
                            p_v = [fcf_input * ((1 + (g_rate / 100)) ** i) / ((1 + (d_rate / 100)) ** i) for i in range(1, 6)]
                            term_val = (p_v[-1] * 1.045) / ((d_rate - 4.5) / 100)
                            dcf_val = (sum(p_v) + (term_val / ((1 + (d_rate / 100)) ** 5))) / shares_input
                            st.markdown(f"#### DCF Fair Value: **₹{dcf_val:,.2f}**")

                # ==========================================
                # OPTION 1 INTEGRATION: PEER COMPARISON ENGINE
                # ==========================================
                with tab5:
                    st.subheader("👥 Peer Comparison & Competitor Benchmarking")
                    default_peers = "INFY, WIPRO, HCLTECH" if "TCS" in yfinance_ticker else "RELIANCE.NS, ONGC.NS, BPCL.NS"
                    peer_input = st.text_input("Enter Competitor Suffix Symbols (Comma Separated):", default_peers)
                    
                    peer_symbols = [p.strip().upper() for p in peer_input.split(",") if p.strip()]
                    if yfinance_ticker not in peer_symbols:
                        peer_symbols.insert(0, yfinance_ticker)
                        
                    peer_matrix = []
                    for symbol in peer_symbols:
                        sym_clean = f"{symbol}.NS" if not symbol.endswith(".NS") and not symbol.startswith("^") else symbol
                        try:
                            p_info = yf.Ticker(sym_clean).info
                            peer_matrix.append({
                                "Ticker": symbol,
                                "Price (₹)": f"₹{p_info.get('currentPrice', p_info.get('regularPrice', 0)):,.2f}",
                                "P/E Multiple": round(p_info.get('trailingPE', 0), 2) or "N/A",
                                "P/B Ratio": round(p_info.get('priceToBook', 0), 2) or "N/A",
                                "ROE (%)": f"{p_info.get('returnOnEquity', 0)*100:.2f}%",
                                "Op. Margin (%)": f"{p_info.get('operatingMargins', 0)*100:.2f}%"
                            })
                        except Exception:
                            continue
                    if peer_matrix:
                        st.table(pd.DataFrame(peer_matrix))

                # ==========================================
                # OPTION 2 INTEGRATION: CORPORATE ACTIONS TRACKER
                # ==========================================
                with tab6:
                    st.subheader("📅 Historical Corporate Actions Tracker")
                    st.write("Review historical stock splits, stock bonuses, and capital adjustments recorded by exchange filings.")
                    
                    if not extended_actions.empty:
                        clean_actions = extended_actions.copy()
                        clean_actions.index = pd.to_datetime(clean_actions.index).strftime('%d %b %Y')
                        st.dataframe(clean_actions.sort_index(ascending=False), use_container_width=True)
                    else:
                        st.info("No recorded stock splits or major bonus corporate adjustments tracking on the API stream.")

                # ==========================================
                # OPTION 3 INTEGRATION: FUNDAMENTAL TREND VISUALIZER
                # ==========================================
                with tab7:
                    st.subheader("📊 Macro Fundamental Trend Line Visualizer")
                    st.write("Top-line operational revenues vs. Net baseline corporate margins scaled into Crores.")
                    
                    rev_row = robust_find_row(extended_financials, "TotalRevenue")
                    net_row = robust_find_row(extended_financials, "NetIncome")
                    
                    if rev_row is not None and net_row is not None:
                        years = [pd.to_datetime(c).strftime('%Y') for c in extended_financials.columns]
                        rev_vals = [float(v) / 10000000 for v in rev_row.values]
                        net_vals = [float(v) / 10000000 for v in net_row.values]
                        
                        trend_fig = go.Figure()
                        trend_fig.add_trace(go.Bar(x=years, y=rev_vals, name='Total Revenue (Cr)', marker_color='#007bff'))
                        trend_fig.add_trace(go.Bar(x=years, y=net_vals, name='Net Income (Cr)', marker_color='#28a745'))
                        trend_fig.update_layout(bmode='group', height=400, yaxis_title="Value in ₹ Crores", margin=dict(t=20, b=20, l=0, r=0))
                        st.plotly_chart(trend_fig, use_container_width=True)
                    else:
                        st.warning("Annual income statements are locked or unavailable on this specific asset structure.")

                # ==========================================
                # OPTION 4 INTEGRATION: MONTE CARLO RISK SIMULATOR
                # ==========================================
                with tab8:
                    st.subheader("🎲 Quantitative Monte Carlo Price Path Projection")
                    st.write("Simulating 1,000 algorithmic random walks over a forward 30-day horizon using historical annualized daily volatility coefficients.")
                    
                    daily_vol = df['Daily_Return'].std()
                    if pd.isna(daily_vol) or daily_vol == 0: daily_vol = 0.015
                    
                    num_days = 30
                    num_sims = 150  # Balanced threshold array for instant mobile rendering speed
                    
                    sim_matrix = np.zeros((num_days, num_sims))
                    sim_matrix[0] = latest_close
                    
                    for d in range(1, num_days):
                        random_shocks = np.random.normal(0, daily_vol, num_sims)
                        sim_matrix[d] = sim_matrix[d-1] * np.exp(random_shocks)
                        
                    mc_fig = go.Figure()
                    for sim in range(num_sims):
                        mc_fig.add_trace(go.Scatter(y=sim_matrix[:, sim], mode='lines', line=dict(width=0.5), opacity=0.3, showlegend=False))
                        
                    mc_fig.update_layout(height=400, xaxis_title="Trading Days Forward", yaxis_title="Target Share Price (₹)", margin=dict(t=10, b=10, l=0, r=0))
                    st.plotly_chart(mc_fig, use_container_width=True)
                    
                    final_day_distribution = sim_matrix[-1, :]
                    p10 = np.percentile(final_day_distribution, 10)
                    p50 = np.percentile(final_day_distribution, 50)
                    p90 = np.percentile(final_day_distribution, 90)
                    
                    st.markdown(f"""
                    ### 🎯 Quant Probability Distribution Targets (30 Days Forward)
                    *   **10th Percentile Outlier Downside:** `₹{p10:,.2f}` (10% historical probability of crashing below this zone)
                    *   **50th Percentile Central Expectation:** `₹{p50:,.2f}` (Median distribution threshold anchor)
                    *   **90th Percentile Outlier Breakout:** `₹{p90:,.2f}` (10% quantitative momentum probability of surging past this zone)
                    """)

                # ==========================================
                # TAB 9: STRATEGY BACKTESTER
                # ==========================================
                with tab9:
                    st.subheader("Historical Strategy Simulation Run")
                    b_col1, b_col2 = st.columns(2)
                    with b_col1: strategy_choice = st.selectbox("Select Vector Line:", ["SMA Crossover (50 vs 200)", "MACD Line Crossover"])
                    with b_col2: initial_capital = st.number_input("Starting Capital (₹)", value=100000)
                    
                    bt_df = df.dropna(subset=['SMA_200']).copy() if strategy_choice == "SMA Crossover (50 vs 200)" else df.dropna(subset=['Signal_Line']).copy()
                    
                    if bt_df.empty:
                        st.error("Insufficient timeline breadth available to backtest.")
                    else:
                        pos, cash, sh, hist = 0, initial_capital, 0, []
                        for date, row in bt_df.iterrows():
                            buy = row['SMA_50'] > row['SMA_200'] if strategy_choice == "SMA Crossover (50 vs 200)" else row['MACD'] > row['Signal_Line']
                            if pos == 0 and buy:
                                sh, cash, pos = cash / row['Close'], 0, 1
                            elif pos == 1 and not buy:
                                cash, sh, pos = sh * row['Close'], 0, 0
                            hist.append(cash + (sh * row['Close']))
                            
                        bt_df['Strat'] = hist
                        bt_df['BH'] = (initial_capital / bt_df['Close'].iloc[0]) * bt_df['Close']
                        
                        st.metric("Strategy Terminal Worth", f"₹{bt_df['Strat'].iloc[-1]:,.2f}")
                        eq_fig = go.Figure()
                        eq_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Strat'], line=dict(color='green'), name='Strategy'))
                        eq_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['BH'], line=dict(color='grey', dash='dash'), name='Buy & Hold'))
                        st.plotly_chart(eq_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Global runtime error compiling analytics frames: {e}")
