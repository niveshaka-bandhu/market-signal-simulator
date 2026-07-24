import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM FONTS CSS
# ==========================================
st.set_page_config(page_title="Indian Quant Verdict Dashboard", layout="wide")

# Custom CSS to downscale massive headers and clean up paddings
st.markdown("""
<style>
    /* Scale down default large fonts */
    h1 {
        font-size: 26px !important;
        font-weight: 700 !important;
        margin-bottom: 10px !important;
    }
    h2 {
        font-size: 20px !important;
        font-weight: 600 !important;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }
    h3 {
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-top: 10px !important;
    }
    
    /* Freeze table headers dynamically */
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

st.title("📊 Institutional Quant Verdict & Analytics Dashboard")

# ==========================================
# 2. HORIZONTAL CONTROL TOP-BAR (NO SIDEBAR)
# ==========================================
with st.container(border=True):
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.8, 1.0, 1.6])
    
    with ctrl_col1:
        dashboard_view = st.radio(
            "🎯 Workspace Selection:",
            ["📈 Market View & Full Verdict", "🧬 Quantitative Deep-Dive"],
            horizontal=True
        )
        
    with ctrl_col2:
        # Checkbox alignment spacer
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        show_bollinger = st.checkbox("Show Bollinger Bands", value=True)
        
    with ctrl_col3:
        raw_ticker_input = st.text_input(
            "🔍 Search Indian Ticker (e.g., RELIANCE, TCS, HDFCBANK, LICI):", 
            "RELIANCE"
        ).upper().strip()

# Exchange Suffixing Logic
if not raw_ticker_input.startswith('^') and '.' not in raw_ticker_input:
    yfinance_ticker = f"{raw_ticker_input}.NS"
else:
    yfinance_ticker = raw_ticker_input

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
        st.warning("Yahoo API limits reached. Fetching prices from Stooq fallback...")
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
# 4. DATA COMPILATION RUNNER
# ==========================================
if yfinance_ticker:
    with st.spinner("Executing structural algorithmic calculations..."):
        try:
            raw_df, stock_info, stock_news = load_data(yfinance_ticker, "5y")
            extended_actions, extended_financials = load_extended_quant_data(yfinance_ticker)
            
            if raw_df is None or raw_df.empty:
                st.error("Invalid ticker string or matching asset not found.")
            else:
                df = calculate_indicators(raw_df)
                
                latest_close = df['Close'].iloc[-1]
                price_change = latest_close - df['Close'].iloc[-2]
                pct_change = (price_change / df['Close'].iloc[-2]) * 100
                
                clean_news_stream = []
                for item in stock_news:
                    if item.get('title') and str(item.get('title')).strip().lower() != "none" and item.get('link') and item.get('providerPublishTime', 0) > 0:
                        clean_news_stream.append(item)

                # ==========================================
                # MULTI-DIMENSIONAL VERDICT ALGORITHM ENGINE
                # ==========================================
                bull_points = []
                bear_points = []
                
                latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50.0
                latest_sma50 = df['SMA_50'].iloc[-1]
                latest_sma200 = df['SMA_200'].iloc[-1]
                latest_macd_hist = df['MACD_Hist'].iloc[-1]
                
                if latest_sma50 > latest_sma200: bull_points.append("Golden Cross confirmed: 50 SMA is riding structurally above the 200 SMA.")
                else: bear_points.append("Death Cross structure: 50 SMA is trailing below the 200 SMA showing macro technical pressure.")
                
                if latest_rsi < 35: bull_points.append(f"RSI reads highly oversold at {latest_rsi:.1f}, signaling tactical exhaustion.")
                elif latest_rsi > 70: bear_points.append(f"RSI reads overbought at {latest_rsi:.1f}, flashing immediate distribution risk.")
                
                if latest_macd_hist > 0: bull_points.append("MACD histogram shows positive expansion above signal threshold lines.")
                else: bear_points.append("MACD momentum shows near-term bearish convergence.")
                
                roe = stock_info.get('returnOnEquity')
                operating_margin = stock_info.get('operatingMargins')
                if roe and roe >= 0.15: bull_points.append(f"High Capital Return Efficiency: ROE sits optimally at {roe*100:.2f}%.")
                elif roe and roe < 0.10: bear_points.append(f"Depressed Capital Return Efficiency: ROE trails below baseline expectations at {roe*100:.2f}%.")
                
                if operating_margin and operating_margin > 0.12: bull_points.append(f"Strong operational baseline health with core margins at {operating_margin*100:.2f}%.")
                
                debt_to_equity = stock_info.get('debtToEquity')
                beta_val = stock_info.get('beta')
                if debt_to_equity and debt_to_equity > 150: bear_points.append(f"Leverage Flag: High Debt-to-Equity balance noted at {debt_to_equity/100:.2f}.")
                elif debt_to_equity and debt_to_equity <= 100: bull_points.append("Protected Capital Base: Leverage models track safely with clean debt levels.")
                
                yf_eps = stock_info.get('trailingEps', 0)
                yf_bvps = stock_info.get('bookValue', 0)
                graham_val = np.sqrt(22.5 * yf_eps * yf_bvps) if (yf_eps and yf_bvps and yf_eps > 0 and yf_bvps > 0) else None
                
                if graham_val and latest_close < graham_val:
                    mos = ((graham_val - latest_close) / graham_val) * 100
                    bull_points.append(f"Under-valued on Graham Intrinsic formulas. Trading with a {mos:.1f}% Margin of Safety.")
                elif graham_val and latest_close > (graham_val * 1.4):
                    bear_points.append("Trading at a significant premium above historical Graham Intrinsic multiples.")

                total_signals = len(bull_points) + len(bear_points)
                bull_ratio = len(bull_points) / total_signals if total_signals > 0 else 0.5
                
                if bull_ratio >= 0.75:
                    master_verdict = "STRATEGIC ACCUMULATION (STRONG BUY)"
                    box_bg, border_color, accent_text = "#d4edda", "#28a745", "#155724"
                    summary_desc = "The algorithmic model flags clear structural backing across multiple domains. Valuations offer a strong buffer, operational health scales cleanly above institutional hurdles, and tactical momentum signals near-term upside velocity."
                elif 0.55 <= bull_ratio < 0.75:
                    master_verdict = "TACTICAL ACCUMULATION (MILD BUY / HOLD)"
                    box_bg, border_color, accent_text = "#fff3cd", "#ffc107", "#856404"
                    summary_desc = "The company retains high core asset quality, but near-term momentum requires careful risk allocation or waiting for mild positional entry liquidations before executing major buy tickets."
                elif 0.35 <= bull_ratio < 0.55:
                    master_verdict = "NEUTRAL WAIT / TRACKING CONTEXT"
                    box_bg, border_color, accent_text = "#e2e3e5", "#6c757d", "#383d41"
                    summary_desc = "Conflicting vector paths detected. Fundamental strengths are currently being offset by poor macro price momentum or premium valuation hurdles. Maintain a neutral posture on the asset."
                else:
                    master_verdict = "RISK AVOIDANCE ORDER (UNDERPERFORM / SELL)"
                    box_bg, border_color, accent_text = "#f8d7da", "#dc3545", "#721c24"
                    summary_desc = "Defensive frameworks triggered. High structural leverage, degrading technical baselines, or extremely overstretched multiples indicate significant downside projection risk paths."

                # ==========================================
                # VIEW 1: MARKET VIEW & COMPREHENSIVE VERDICT
                # ==========================================
                if dashboard_view == "📈 Market View & Full Verdict":
                    st.write(f"## 📈 Strategic Asset Intelligence Center ({raw_ticker_input})")
                    
                    # Core High-Impact Verdict Block
                    st.markdown(f"""
                    <div style="background-color:{box_bg}; border-left:6px solid {border_color}; padding:15px; border-radius:8px;">
                        <h2 style="margin:0 0 5px 0; color:{accent_text}; font-weight:bold; font-size:18px !important;">🔍 SYSTEM DISPATCH: {master_verdict}</h2>
                        <p style="margin:0; color:{accent_text}; font-size:14px; line-height:1.4;"><strong>Executive Summary Analysis:</strong> {summary_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Split Columns for Bull vs Bear Factors
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.write("### ✅ Positive Structural Drivers (Bulls)")
                        if bull_points:
                            for p in bull_points: st.markdown(f" * {p}")
                        else: st.write("No distinct positive algorithmic signals triggered.")
                    with v_col2:
                        st.write("### ⚠️ Structural Risk Warnings (Bears)")
                        if bear_points:
                            for p in bear_points: st.markdown(f" * {p}")
                        else: st.write("No severe structural risk vectors flagged.")
                        
                    st.markdown("---")
                    
                    # Primary Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Current Value", f"₹{latest_close:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
                    m2.metric("Relative Strength Index", f"{latest_rsi:.1f}")
                    m3.metric("MACD Divergence", f"{df['MACD_Hist'].iloc[-1]:.2f}")
                    m4.metric("Bull Convergence Weight", f"{bull_ratio*100:.1f}%")

                    st.markdown("---")
                    
                    # Charting Framework & News
                    chart_col, sidebar_news_col = st.columns([3, 1.2]) if clean_news_stream else (st.container(), None)
                    
                    with chart_col:
                        st.write("### 📊 Price Action Engine")
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
                            st.write("### 📰 Real-time Headlines")
                            for item in clean_news_stream[:5]:
                                pub_time = datetime.fromtimestamp(item.get('providerPublishTime')).strftime('%d %b %Y')
                                st.markdown(f"**[{item.get('title')}]({item.get('link')})**  \n<small style='color:gray;'>{item.get('publisher')} | {pub_time}</small>\n---", unsafe_allow_html=True)

                    st.markdown("---")
                    core_tab1, core_tab2 = st.tabs(["📐 Intraday Pivot Target Framework", "💎 Intrinsic Valuation Models"])
                    
                    with core_tab1:
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

                    with core_tab2:
                        yf_mcap = stock_info.get('marketCap', latest_close * 1000000)
                        yf_fcf_crores = stock_info.get('freeCashflow', yf_mcap * 0.05) / 10000000
                        yf_shares_crores = stock_info.get('sharesOutstanding', yf_mcap / latest_close) / 10000000

                        calc_col1, calc_col2 = st.columns(2)
                        with calc_col1:
                            st.write("### 🏛️ Graham Valuation Model")
                            # FIX: Dynamically keying widget by yfinance_ticker forces refresh when company changes
                            eps_input = st.number_input("EPS", value=float(yf_eps) if yf_eps else 10.0, format="%.2f", key=f"eps_{yfinance_ticker}")
                            bvps_input = st.number_input("BVPS", value=float(yf_bvps) if yf_bvps else 100.0, format="%.2f", key=f"bvps_{yfinance_ticker}")
                            if eps_input > 0 and bvps_input > 0:
                                g_val = np.sqrt(22.5 * eps_input * bvps_input)
                                st.markdown(f"Calculated Graham Value: **₹{g_val:,.2f}**")

                        with calc_col2:
                            st.write("### 🌀 Multi-Stage DCF Model (₹ Crores)")
                            # FIX: Dynamically keying widget by yfinance_ticker forces refresh when company changes
                            fcf_input = st.number_input("FCF (₹ Crores)", value=float(yf_fcf_crores) if yf_fcf_crores else 500.0, format="%.2f", key=f"fcf_{yfinance_ticker}")
                            shares_input = st.number_input("Shares Out (Crores)", value=float(yf_shares_crores) if yf_shares_crores else 50.0, format="%.2f", key=f"sh_{yfinance_ticker}")
                            g_rate = st.slider("Growth Rate (%)", -5.0, 40.0, 12.0, 0.5, key=f"g_{yfinance_ticker}")
                            d_rate = st.slider("Discount Rate (%)", 5.0, 25.0, 11.0, 0.5, key=f"d_{yfinance_ticker}")
                            
                            if d_rate > 4.5:
                                p_v = [fcf_input * ((1 + (g_rate / 100)) ** i) / ((1 + (d_rate / 100)) ** i) for i in range(1, 6)]
                                term_val = (p_v[-1] * 1.045) / ((d_rate - 4.5) / 100)
                                dcf_val = (sum(p_v) + (term_val / ((1 + (d_rate / 100)) ** 5))) / shares_input
                                st.markdown(f"Calculated DCF Fair Value: **₹{dcf_val:,.2f}**")

                # ==========================================
                # VIEW 2: QUANTITATIVE DEEP-DIVE PAGE
                # ==========================================
                elif dashboard_view == "🧬 Quantitative Deep-Dive":
                    st.write(f"## 🧬 Structured Analytics Deep-Dive Workspace ({raw_ticker_input})")
                    
                    quant_tab1, quant_tab2, quant_tab3, quant_tab4, quant_tab5, quant_tab6 = st.tabs([
                        "🧬 Multi-Factor Quality Matrix",
                        "👥 Peer Benchmarking Engine",
                        "📅 Corporate Action Logs",
                        "📊 Fundamental Margin Trends",
                        "🎲 Monte Carlo Distributions",
                        "🕒 Historical Strategy Backtester"
                    ])
                    
                    with quant_tab1:
                        st.write("### 🧬 Accounting & Solvency Factor Grid")
                        operating_margin = stock_info.get('operatingMargins')
                        
                        def fmt_pct(val): return f"{val * 100:.2f}%" if val is not None else "N/A"
                        def fmt_num(val, mult=1): return f"{val/mult:.2f}" if val is not None else "N/A"

                        factor_data = {
                            "Quant Performance Factor": ["Return on Equity (ROE)", "Return on Assets (ROA)", "Operating Profit Margin", "Debt-to-Equity Ratio", "Current Solvency Ratio", "Systematic Volatility (Beta)", "PEG Multiple"],
                            "Current Reading": [fmt_pct(roe), fmt_pct(stock_info.get('returnOnAssets')), fmt_pct(operating_margin), fmt_num(debt_to_equity, 100) if debt_to_equity else "N/A", fmt_num(stock_info.get('currentRatio')), fmt_num(beta_val), fmt_num(stock_info.get('pegRatio'))],
                            "Target Threshold": ["> 15.00% Optimal", "> 8.00% Optimal", "> 12.00% High Efficiency", "< 1.00 Low Leverage", "> 1.20 Cash Soundness", "< 1.00 Defensive", "< 1.50 Value Growth"]
                        }
                        st.table(pd.DataFrame(factor_data))
                        
                        st.write("### 🏢 Equity Block Allocation Matrix")
                        insider_share = stock_info.get('heldPercentInsiders', 0.0) * 100
                        inst_share = stock_info.get('heldPercentInstitutions', 0.0) * 100
                        public_share = max(0.0, 100.0 - (insider_share + inst_share))
                        
                        sh_col1, sh_col2, sh_col3 = st.columns(3)
                        sh_col1.metric("Promoter Block", f"{insider_share:.2f}%" if insider_share > 0 else "N/A")
                        sh_col2.metric("Institutional Float (FII/DII)", f"{inst_share:.2f}%" if inst_share > 0 else "N/A")
                        sh_col3.metric("Estimated Public Float", f"{public_share:.2f}%" if public_share < 100 else "N/A")

                    with quant_tab2:
                        st.write("### 👥 Competitor Evaluation Matrix")
                        # Sector intelligent default peers
                        sector = stock_info.get('sector', '')
                        if 'Technology' in sector or 'Software' in sector:
                            default_peers = "INFY.NS, WIPRO.NS, HCLTECH.NS, TCS.NS"
                        elif 'Financial' in sector or 'Bank' in sector:
                            default_peers = "HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, KOTAKBANK.NS"
                        elif 'Energy' in sector or 'Utilities' in sector:
                            default_peers = "RELIANCE.NS, ONGC.NS, BPCL.NS, IOC.NS"
                        elif 'Auto' in sector:
                            default_peers = "TATAMOTORS.NS, MARUTI.NS, M&M.NS"
                        elif 'Healthcare' in sector or 'Pharma' in sector:
                            default_peers = "SUNPHARMA.NS, CIPLA.NS, DRREDDY.NS"
                        else:
                            default_peers = "RELIANCE.NS, TCS.NS, HDFCBANK.NS"

                        # FIX: Dynamic keying ensures text input resets to new sector defaults when ticker changes
                        peer_input = st.text_input("Enter Peer Symbols (Comma Separated):", default_peers, key=f"peer_input_{yfinance_ticker}")
                        
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

                    with quant_tab3:
                        st.write("### 📅 Adjustments & Historical Share Splits")
                        if not extended_actions.empty:
                            clean_actions = extended_actions.copy()
                            clean_actions.index = pd.to_datetime(clean_actions.index).strftime('%d %b %Y')
                            st.dataframe(clean_actions.sort_index(ascending=False), use_container_width=True)
                        else:
                            st.info("No recorded corporate actions or adjustments tracked on this engine timeline.")

                    with quant_tab4:
                        st.write("### 📊 Top-line vs Balance Income Vectors")
                        rev_row = robust_find_row(extended_financials, "TotalRevenue")
                        net_row = robust_find_row(extended_financials, "NetIncome")
                        
                        if rev_row is not None and net_row is not None:
                            years = [pd.to_datetime(c).strftime('%Y') for c in extended_financials.columns]
                            rev_vals = [float(v) / 10000000 for v in rev_row.values]
                            net_vals = [float(v) / 10000000 for v in net_row.values]
                            
                            trend_fig = go.Figure()
                            trend_fig.add_trace(go.Bar(x=years, y=rev_vals, name='Total Revenue (Cr)', marker_color='#007bff'))
                            trend_fig.add_trace(go.Bar(x=years, y=net_vals, name='Net Income (Cr)', marker_color='#28a745'))
                            trend_fig.update_layout(barmode='group', height=400, yaxis_title="Value in ₹ Crores", margin=dict(t=20, b=20, l=0, r=0))
                            st.plotly_chart(trend_fig, use_container_width=True)
                        else:
                            st.warning("Annual income statements are unavailable on this specific asset structure.")

                    with quant_tab5:
                        st.write("### 🎲 30-Day Randomized Variance Walk Simulations")
                        daily_vol = df['Daily_Return'].std()
                        if pd.isna(daily_vol) or daily_vol == 0: daily_vol = 0.015
                        
                        num_days, num_sims = 30, 150
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
                        *   **10th Percentile Outlier Downside Risk Boundary:** `₹{p10:,.2f}`
                        *   **50th Percentile Central Distribution Anchor:** `₹{p50:,.2f}`
                        *   **90th Percentile Outlier Breakout Boundary:** `₹{p90:,.2f}`
                        """)

                    with quant_tab6:
                        st.write("### 🕒 Historical Strategy Simulation Run")
                        b_col1, b_col2 = st.columns(2)
                        with b_col1: strategy_choice = st.selectbox("Select Strategy Rule Line:", ["SMA Crossover (50 vs 200)", "MACD Line Crossover"])
                        with b_col2: initial_capital = st.number_input("Starting Capital (₹)", value=100000)
                        
                        bt_df = df.dropna(subset=['SMA_200']).copy() if strategy_choice == "SMA Crossover (50 vs 200)" else df.dropna(subset=['Signal_Line']).copy()
                        
                        if bt_df.empty:
                            st.error("Insufficient timeline breadth available to backtest indicators.")
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
                            eq_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Strat'], line=dict(color='green'), name='Strategy Run'))
                            eq_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['BH'], line=dict(color='grey', dash='dash'), name='Buy & Hold Base'))
                            st.plotly_chart(eq_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Global execution framework runtime error: {e}")
