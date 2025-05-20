import streamlit as st
import pandas as pd
from datetime import datetime, time
import yfinance as yf
import plotly.graph_objs as go
import requests
from streamlit_autorefresh import st_autorefresh

# 🔒 Pushover credentials are loaded from Streamlit Secrets
#   Configure these in your Streamlit Cloud settings under "Secrets":
#   pushover_user_key  = "<your Pushover User Key>"
#   pushover_api_token = "<your Pushover API Token>"
PUSHOVER_USER_KEY = st.secrets.get("pushover_user_key", "")
PUSHOVER_API_TOKEN = st.secrets.get("pushover_api_token", "")

def send_pushover_notification(title, message):
    """
    Send a push notification via Pushover.
    Requires pushover_user_key and pushover_api_token in Streamlit secrets.
    """
    if PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN:
        data = {
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": title,
            "message": message
        }
        requests.post("https://api.pushover.net/1/messages.json", data=data)

# 📰 News Sentiment Analysis Setup
NEWS_API_KEY = st.secrets.get("news_api_key", "")

# Optional: use VADER for lightweight sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def get_headlines(symbol):
    """Fetch latest headlines for a given symbol using NewsAPI."""
    if not NEWS_API_KEY:
        return []
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={symbol}&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    )
    articles = requests.get(url).json().get("articles", [])
    return [f"{a['title']}: {a.get('description','')}" for a in articles]

def headline_sentiment_score(headlines):
    """Compute average compound sentiment score for a list of headlines."""
    if not headlines:
        return 0.0
    scores = [sia.polarity_scores(h)["compound"] for h in headlines]
    return sum(scores) / len(scores)

# --- 🛠️ 6️⃣ Monte Carlo & Black-Scholes for Hit Probabilities ---
import numpy as np
from scipy.stats import norm

@st.cache_data
def mc_hit_probability(S0, target, T_days, vol_annual, n_sims=10000, n_steps=100):
    """
    Estimate probability that a GBM path starting at S0
    hits `target` within T_days trading days.
    """
    dt = (T_days/252) / n_steps
    drift = -0.5 * vol_annual**2 * dt
    diffusion = vol_annual * np.sqrt(dt)
    hits = 0
    for _ in range(n_sims):
        increments = drift + diffusion * np.random.randn(n_steps)
        path = S0 * np.exp(np.cumsum(increments))
        if path.max() >= target:
            hits += 1
    return hits / n_sims

@st.cache_data
def bs_itm_prob(S0, K, T_years, vol):
    """
    Compute Black–Scholes probability S_T > K at expiry T_years.
    """
    d2 = (np.log(S0/K) - 0.5 * vol**2 * T_years) / (vol * np.sqrt(T_years))
    return norm.cdf(d2)


# --- Streamlit Page Setup ---
st.set_page_config(page_title="PLTR Day Trade Tracker", layout="wide")

# --- Hard Stop Control ---
# Use query_params API for persistence
pause_default = st.query_params.get("pause", ["0"])[0] == "1"
pause_all = st.sidebar.checkbox(
    "Pause All App Logic",
    value=pause_default,
    key="pause_all",
    help="When checked, the app will halt all data updates, notifications, and charts."
)
if pause_all:
    st.query_params = {"pause": ["1"]}
    st.sidebar.info("⏸️ App is paused. Uncheck to resume.")
    st.stop()
else:
    st.query_params = {"pause": ["0"]}


# --- Sidebar Controls (always visible) ---
with st.sidebar:
    st.header("🔧 Auto-Refresh Settings")
    auto_refresh = st.checkbox(
        "Enable Auto-Refresh (15 seconds)",
        value=True,
        help="Uncheck to pause all data updates, charts, and alerts"
    )
    resume_time = st.time_input(
        "Resume auto-refresh at:",
        value=None,
        help="Automatically re-enable refresh at this time"
    )

# Auto-refresh logic
now = datetime.now().time()
if resume_time and now >= resume_time:
    auto_refresh = True
if auto_refresh:
    st_autorefresh(interval=15_000, limit=None, key="refresh_timer")

# --- Main Page ---
st.title("📈 PLTR Day Trade Tracker with Charts & Alerts")

# --- 1️⃣ Load or Initialize Trade Data ---
@st.cache_data
def load_data():
    return pd.DataFrame(columns=[
        "Symbol", "Date", "Trade Type", "Entry Price", "Exit Price",
        "Stop Loss", "Target Price", "Actual Result",
        "Confidence", "Notes"
    ])

df = load_data()

# --- 2️⃣ Symbol Selection & Trade Logging Form ---
# Enter symbols to track
symbol_input = st.text_input(
    "Enter symbols to track (comma-separated)",
    value="PLTR,TSLA,AAPL",
    help="Type ticker symbols separated by commas."
)
symbols = [s.strip().upper() for s in symbol_input.split(",") if s.strip()]

with st.form("trade_form"):
    st.subheader("Log a New Trade")
    # Choose which symbol this trade applies to
    trade_symbol = st.selectbox("Symbol", symbols)
    trade_type = st.selectbox("Trade Type", ["Breakout", "Breakdown", "Range", "Call", "Put"])
    entry = st.number_input("Entry Price", format="%.2f")
    exit_price = st.number_input("Exit Price", format="%.2f")
    stop = st.number_input("Stop Loss", format="%.2f")
    target = st.number_input("Target Price", format="%.2f")
    result = st.selectbox("Trade Result", ["Pending", "Win", "Loss", "Breakeven"])
    confidence = st.slider("Confidence Level", 0, 100, 50)
    notes = st.text_area("Notes")
    submitted = st.form_submit_button("Add Trade")

    if submitted:
        new_row = pd.DataFrame([{
            "Symbol": trade_symbol,
            "Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Trade Type": trade_type,
            "Entry Price": entry,
            "Exit Price": exit_price,
            "Stop Loss": stop,
            "Target Price": target,
            "Actual Result": result,
            "Confidence": confidence,
            "Notes": notes
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        st.success("Trade added successfully!")

# --- 3️⃣ Fetch & Compute Indicators for Each Symbol --- & Compute Indicators for Each Symbol ---
for symbol in symbols:
    st.markdown(f"## 📊 {symbol} Analysis")
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="5d", interval="5m")
    if history.empty or len(history) < 2:
        # Fallback to 15m if 5m data insufficient
        history = ticker.history(period="5d", interval="15m")

    # Compute EMA and RSI
    history['EMA9'] = history['Close'].ewm(span=9, adjust=False).mean()
    history['EMA21'] = history['Close'].ewm(span=21, adjust=False).mean()
    delta = history['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    history['RSI'] = 100 - (100 / (1 + rs))

    # Compute MACD and Signal
    history['EMA12'] = history['Close'].ewm(span=12, adjust=False).mean()
    history['EMA26'] = history['Close'].ewm(span=26, adjust=False).mean()
    history['MACD'] = history['EMA12'] - history['EMA26']
    history['Signal'] = history['MACD'].ewm(span=9, adjust=False).mean()
    
    # Compute average volume
    history['AvgVol'] = history['Volume'].rolling(20).mean()

    # --- 4️⃣ Debug Info ---
    # 🧠 Pattern Recognition
    # Last close price
    last_close = history['Close'].iloc[-1]
    # Breakout scan
    prior_high = history['High'].rolling(window=20).max().shift(1)
    breakout = last_close > prior_high.iloc[-1] if not prior_high.empty else False
    # Cup and handle detection (naive)
    window = history['Close'].tail(30)
    cup = False
    if len(window) == 30:
        trough_idx = window.idxmin()
        trough_pos = list(window.index).index(trough_idx)
        cup = (window.iloc[0] > window.min() < window.iloc[-1]) and (10 <= trough_pos <= 20)
    # Triangle detection (naive)
    highs = history['High'].tail(20)
    lows = history['Low'].tail(20)
    rng = highs - lows
    triangle = False
    if len(rng) >= 20:
        triangle = (rng.iloc[0] > rng.iloc[-1]) and all(rng.iloc[i] > rng.iloc[i+1] for i in range(len(rng)-1))
    if cup:
        pattern_msg = "Cup-and-Handle pattern detected"
    elif triangle:
        pattern_msg = "Triangle pattern detected"
    elif breakout:
        pattern_msg = "Breakout of 20-period high"
    else:
        pattern_msg = "No pattern detected"
    st.write(f"🧠 Pattern Recognition: {pattern_msg}")
    st.markdown("### 📊 Debug Info")
    st.write("Last 5 Rows of History:")
    st.dataframe(history.tail(5))
    st.write("Shape:", history.shape)
    st.write("Columns:", history.columns.tolist())

    # 📰 5️⃣ News Sentiment Analysis
    headlines = get_headlines(symbol)
    sentiment = headline_sentiment_score(headlines)
    st.write(f"📰 News Sentiment (avg score): {sentiment:.2f}")
    # Optionally adjust confidence based on sentiment
    if sentiment > 0.2:
        confidence = min(100, confidence + 10)
    elif sentiment < -0.2:
        confidence = max(0, confidence - 10)

# --- 5️⃣ Strategy Logic & Chart --- & Chart --- & Chart ---
    if not auto_refresh:
        st.info("⏸ Strategy logic is paused while Auto-Refresh is off.")
    else:
        valid = history.dropna(subset=["EMA9", "EMA21", "RSI"])
        if valid.empty:
            st.warning("⚠️ Not enough valid data to compute predictions.")
        else:
            latest = valid.iloc[-1]
            current_price = latest['Close']

            # Live price metric
            st.metric(label="🔔 Live Price", value=f"${current_price:.2f}")

            # Trend, RSI, MACD, Volume
            trend_bias = "Bullish (CALL)" if latest['EMA9'] > latest['EMA21'] else "Bearish (PUT)"
            rsi_val = round(latest['RSI'], 2)
            macd_val = round(latest['MACD'], 2)
            signal_val = round(latest['Signal'], 2)
            vol_val = latest['Volume']
            avg_vol_val = round(latest['AvgVol'], 0)

            # Confidence Level
            confidence_msg = "Moderate"
            if trend_bias.startswith("Bullish"):
                if macd_val > signal_val and vol_val > avg_vol_val and rsi_val < 65:
                    confidence_msg = "Very High"
                elif macd_val > signal_val and rsi_val < 65:
                    confidence_msg = "High"
                elif 45 <= rsi_val <= 55:
                    confidence_msg = "Low"
            else:
                if macd_val < signal_val and vol_val > avg_vol_val and rsi_val > 35:
                    confidence_msg = "Very High"
                elif macd_val < signal_val and rsi_val > 35:
                    confidence_msg = "High"
                elif 45 <= rsi_val <= 55:
                    confidence_msg = "Low"

            # Display prediction
            st.subheader(f"📌 Suggested Strategy: {trend_bias} — {confidence_msg} Confidence")
            st.write(f"📊 RSI: {rsi_val}")
            st.write(f"📊 MACD: {macd_val} vs Signal: {signal_val}")
            st.write(f"📊 Volume: {vol_val} vs AvgVol: {avg_vol_val}")

            # Push notification on trend change
            if 'last_trend_bias' not in st.session_state or st.session_state.last_trend_bias != trend_bias:
                send_pushover_notification("Strategy Update", f"{symbol} trend: {trend_bias} at ${current_price:.2f}")
                st.session_state.last_trend_bias = trend_bias

            # Chart with markers
            fig = go.Figure(data=[go.Candlestick(
                x=history.index,
                open=history['Open'], high=history['High'],
                low=history['Low'], close=history['Close']
            )])
            fig.add_trace(go.Scatter(x=history.index, y=history['EMA9'], mode='lines', name='EMA9'))
            fig.add_trace(go.Scatter(x=history.index, y=history['EMA21'], mode='lines', name='EMA21'))
            fig.add_hline(y=entry, line=dict(color='blue', dash='dot'), annotation_text='Entry', annotation_position='top left')
            fig.add_hline(y=target, line=dict(color='green', dash='dash'), annotation_text='Target', annotation_position='top right')
            fig.add_hline(y=stop, line=dict(color='red', dash='dash'), annotation_text='Stop', annotation_position='bottom right')
            fig.update_layout(title=f"{symbol} Chart", height=300)
            st.plotly_chart(fig, use_container_width=True)

# --- 6️⃣ Summary of Trades by Symbol ---
st.subheader("📌 Trade Summary by Symbol")
if df.empty:
    st.info("No trades logged yet.")
else:
    fig_summary = df.copy()
    fig_summary['Profit/Loss'] = df['Profit/Loss'] if 'Profit/Loss' in df.columns else df.apply(
        lambda row: round((row['Exit Price'] - row['Entry Price']), 2) if row['Trade Type'] in ['Call','Breakout'] else round((row['Entry Price']-row['Exit Price']),2), axis=1
    )
    grouped = fig_summary.groupby('Symbol').agg(
        Total_Trades=('Symbol','count'),
        Wins=('Actual Result', lambda x: (x=='Win').sum()),
        Losses=('Actual Result', lambda x: (x=='Loss').sum()),
        Breakevens=('Actual Result', lambda x: (x=='Breakeven').sum()),
        Net_PL=('Profit/Loss','sum')
    )
    for symbol, row in grouped.iterrows():
        st.markdown(f"### {symbol}")
        st.write(f"Total Trades: {row['Total_Trades']}")
        st.write(f"Wins: {row['Wins']}")
        st.write(f"Losses: {row['Losses']}")
        st.write(f"Breakevens: {row['Breakevens']}")
        st.write(f"Net P/L: {round(row['Net_PL'],2)}")

# --- Download CSV ---
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download All Trades", data=csv, file_name='trades.csv', mime='text/csv')