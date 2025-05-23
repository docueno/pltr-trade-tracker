import streamlit as st
import pandas as pd
from datetime import datetime, time, timedelta
import yfinance as yf
import plotly.graph_objs as go
import requests
from streamlit_autorefresh import st_autorefresh
import numpy as np
from scipy.stats import norm
import os
from pytz import timezone
# import GoogleNews  # Commented out temporarily
import praw
import json

# 🔒 Pushover credentials are loaded from Streamlit Secrets
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
ALPHA_VANTAGE_API_KEY = st.secrets.get("alpha_vantage_api_key", "")
FINNHUB_API_KEY = st.secrets.get("finnhub_api_key", "")
REDDIT_CLIENT_ID = st.secrets.get("reddit_client_id", "")
REDDIT_CLIENT_SECRET = st.secrets.get("reddit_client_secret", "")
REDDIT_USER_AGENT = st.secrets.get("reddit_user_agent", "pltr_trade_tracker/1.0 by /u/unknown")  # Default user agent

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Source weights for sentiment averaging
SOURCE_WEIGHTS = {
    "NewsAPI": 0.4,      # Adjusted weight to compensate for missing Google News
    "Alpha Vantage": 0.4,
    "Finnhub": 0.1,
    "Reddit": 0.1        # Lower weight for user-generated content
}

def get_headlines_newsapi(symbol):
    """Fetch latest headlines for a given symbol using NewsAPI."""
    if not NEWS_API_KEY:
        return []
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={symbol}&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    )
    try:
        articles = requests.get(url).json().get("articles", [])
        return [(f"{a['title']}: {a.get('description','')}", "NewsAPI") for a in articles]
    except:
        return []

def get_headlines_alpha_vantage(symbol):
    """Fetch latest headlines for a given symbol using Alpha Vantage."""
    if not ALPHA_VANTAGE_API_KEY:
        return []
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=NEWS_SENTIMENT&tickers={symbol}&limit=5&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    try:
        response = requests.get(url).json()
        articles = response.get("feed", [])
        return [(f"{a['title']}: {a.get('summary','')}", "Alpha Vantage") for a in articles]
    except:
        return []

# def get_headlines_google_news(symbol):  # Commented out temporarily
#     """Fetch latest headlines for a given symbol using GoogleNews library."""
#     try:
#         googlenews = GoogleNews(lang='en', period='7d')
#         googlenews.search(symbol)
#         articles = googlenews.results()[:5]  # Limit to 5 articles
#         return [(f"{a['title']}: {a.get('desc','')}", "Google News") for a in articles]
#     except:
#         return []

def get_headlines_finnhub(symbol):
    """Fetch latest headlines for a given symbol using Finnhub."""
    if not FINNHUB_API_KEY:
        return []
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    url = (
        f"https://finnhub.io/api/v1/company-news?"
        f"symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
    )
    try:
        articles = requests.get(url).json()[:5]  # Limit to 5 articles
        return [(f"{a['headline']}: {a.get('summary','')}", "Finnhub") for a in articles]
    except:
        return []

def get_headlines_reddit(symbol):
    """Fetch latest posts mentioning the symbol from r/wallstreetbets using Reddit API."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        st.warning("Reddit API credentials not found. Please update Streamlit secrets with reddit_client_id and reddit_client_secret.")
        return []
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        subreddit = reddit.subreddit("wallstreetbets")
        posts = []
        for submission in subreddit.search(symbol, limit=5):
            posts.append((f"{submission.title}: {submission.selftext[:200]}", "Reddit"))
        return posts
    except Exception as e:
        st.error(f"Error fetching Reddit data: {str(e)}")
        return []

def get_headlines(symbol):
    """Aggregate headlines from multiple sources with their source labels."""
    headlines = []
    headlines.extend(get_headlines_newsapi(symbol))
    headlines.extend(get_headlines_alpha_vantage(symbol))
    # headlines.extend(get_headlines_google_news(symbol))  # Commented out temporarily
    headlines.extend(get_headlines_finnhub(symbol))
    headlines.extend(get_headlines_reddit(symbol))
    # Remove duplicates based on headline text
    seen = set()
    unique_headlines = []
    for headline, source in headlines:
        if headline not in seen:
            seen.add(headline)
            unique_headlines.append((headline, source))
    return unique_headlines

def headline_sentiment_score(headlines_with_source):
    """Compute weighted average sentiment score and per-source scores."""
    if not headlines_with_source:
        return 0.0, {}
    
    per_source_scores = {}
    for headline, source in headlines_with_source:
        score = sia.polarity_scores(headline)["compound"]
        if source not in per_source_scores:
            per_source_scores[source] = []
        per_source_scores[source].append(score)
    
    # Average sentiment per source
    per_source_avg = {source: sum(scores)/len(scores) for source, scores in per_source_scores.items()}
    
    # Weighted overall sentiment
    weighted_sum = 0.0
    total_weight = 0.0
    for source, avg_score in per_source_avg.items():
        weight = SOURCE_WEIGHTS.get(source, 0.1)
        weighted_sum += avg_score * weight
        total_weight += weight
    
    overall_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0
    return overall_sentiment, per_source_avg

# --- 🛠️ Monte Carlo & Black-Scholes for Hit Probabilities ---
@st.cache_data
def mc_hit_probability(S0, target, T_days, vol_annual, n_sims=10000, n_steps=100):
    """
    Estimate probability that a GBM path starting at S0
    hits `target` above (for CALL) or stays below (for PUT) within T_days trading days.
    Returns probability of hitting above target; invert for PUT logic.
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
def bs_itm_prob(S0, K, T_years, vol, dividend_yield=0.0, risk_free_rate=0.04):
    """
    Compute Black-Scholes probability S_T > K at expiry T_years (for CALL) or < K (for PUT with 1 - result).
    Adjusted for dividends.
    """
    d2 = (np.log(S0/K) + (risk_free_rate - dividend_yield - 0.5 * vol**2) * T_years) / (vol * np.sqrt(T_years))
    return norm.cdf(d2)

# Check market status (NYSE hours: 9:30 AM - 4:00 PM EDT)
def is_market_open():
    eastern = timezone('US/Eastern')
    now = datetime.now(eastern)
    return now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0)

# Adjust expiration date to the nearest Friday (weekly options)
def get_next_friday(start_date, days_to_expiration):
    current_date = start_date
    trading_days = 0
    while trading_days < days_to_expiration:
        current_date += timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            trading_days += 1
    # Adjust to the nearest Friday
    while current_date.weekday() != 4:  # Friday = 4
        current_date += timedelta(days=1)
    return current_date

# --- Fetch and Analyze SPY Data for Market Trends ---
@st.cache_data
def get_spy_trend():
    """Fetch SPY data and compute trend indicators."""
    spy_ticker = yf.Ticker("SPY")
    spy_history = spy_ticker.history(period="10d", interval="5m", prepost=True)
    if spy_history.empty or len(spy_history) < 2:
        spy_history = spy_ticker.history(period="10d", interval="15m", prepost=True)
    
    # Compute EMA and RSI for SPY
    spy_history['EMA9'] = spy_history['Close'].ewm(span=9, adjust=False).mean()
    spy_history['EMA21'] = spy_history['Close'].ewm(span=21, adjust=False).mean()
    delta = spy_history['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    spy_history['RSI'] = 100 - (100 / (1 + rs))
    
    latest_spy = spy_history.iloc[-1]
    spy_trend = "Bullish" if latest_spy['EMA9'] > latest_spy['EMA21'] else "Bearish"
    spy_rsi = round(latest_spy['RSI'], 2)
    spy_current_price = latest_spy['Close']
    
    return spy_trend, spy_rsi, spy_current_price, spy_history

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Day Trade Tracker with Charts & Alerts", layout="centered")

# --- Hard Stop Control ---
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

# --- Sidebar Controls ---
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
st.title("📈 Day Trade Tracker with Charts & Alerts")

# Market status message
if not is_market_open():
    st.info("⚠️ Market is currently closed (outside 9:30 AM - 4:00 PM EDT). Recommendations are based on the latest available data.")

# --- Display SPY Market Trend ---
spy_trend, spy_rsi, spy_current_price, spy_history = get_spy_trend()
st.markdown("### 📉 Market Trend (SPY - S&P 500)")
st.write(f"**SPY Trend:** {spy_trend}")
st.metric(label="🔔 SPY Last Close", value=f"${spy_current_price:.2f}")
st.write(f"**SPY RSI:** {spy_rsi}")
# SPY Chart
spy_fig = go.Figure(data=[go.Candlestick(
    x=spy_history.index,
    open=spy_history['Open'], high=spy_history['High'],
    low=spy_history['Low'], close=spy_history['Close']
)])
spy_fig.add_trace(go.Scatter(x=spy_history.index, y=spy_history['EMA9'], mode='lines', name='EMA9'))
spy_fig.add_trace(go.Scatter(x=spy_history.index, y=spy_history['EMA21'], mode='lines', name='EMA21'))
spy_fig.update_layout(title="SPY Chart", height=300)
st.plotly_chart(spy_fig, use_container_width=True)

# --- 1️⃣ Load or Initialize Trade Data ---
@st.cache_data
def load_data():
    return pd.DataFrame(columns=[
        "Symbol", "Date", "Trade Type", "Entry Price", "Exit Price",
        "Stop Loss", "Target Price", "Days to Expiration", "Implied Volatility",
        "Actual Result", "Confidence", "Notes"
    ])

df = load_data()

# --- 2️⃣ Initialize Prediction History ---
prediction_file = "prediction_history.csv"
if 'prediction_df' not in st.session_state:
    if os.path.exists(prediction_file):
        st.session_state.prediction_df = pd.read_csv(prediction_file)
    else:
        st.session_state.prediction_df = pd.DataFrame(columns=[
            "Symbol", "Timestamp", "Trade Type", "Target Price", "Days to Expiration",
            "Monte Carlo Probability", "Black-Scholes Probability"
        ])

# --- 3️⃣ Symbol Selection & Trade Logging Form ---
symbols_input = st.text_input("Enter symbols to track (comma-separated):", value="")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

with st.form("trade_form"):
    st.subheader("Log a New Trade")
    trade_symbol = st.selectbox("Symbol", symbols)
    trade_type = st.selectbox("Trade Type", ["Breakout", "Breakdown", "Range", "Call", "Put"])
    entry = st.number_input("Entry Price", format="%.2f")
    exit_price = st.number_input("Exit Price", format="%.2f")
    stop = st.number_input("Stop Loss", format="%.2f")
    target = st.number_input("Target Price", format="%.2f")
    days_to_expiration = st.number_input("Days to Expiration", min_value=1, max_value=252, value=5)
    implied_vol = st.number_input("Implied Volatility (%)", min_value=0.0, max_value=100.0, value=30.0, format="%.2f") / 100
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
            "Days to Expiration": days_to_expiration,
            "Implied Volatility": implied_vol,
            "Actual Result": result,
            "Confidence": confidence,
            "Notes": notes
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        st.success("Trade added successfully!")

# --- 4️⃣ Fetch & Compute Indicators for Each Symbol ---
if 'latest_probs' not in st.session_state:
    st.session_state.latest_probs = {}

for symbol in symbols:
    st.markdown(f"## 📊 {symbol} Analysis")
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="10d", interval="5m", prepost=True)
    if history.empty or len(history) < 2:
        history = ticker.history(period="10d", interval="15m", prepost=True)

    # Fetch dividend yield
    dividend_yield = ticker.info.get('dividendYield', 0.0) or 0.0

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

    # Compute historical volatility (fallback)
    log_returns = np.log(history['Close'] / history['Close'].shift(1))
    vol_annual = np.std(log_returns) * np.sqrt(252) if not log_returns.empty else 0.3
    if log_returns.empty:
        st.warning(f"⚠️ Insufficient data for volatility calculation for {symbol}. Using default 30%.")

    # --- 5️⃣ Debug Info ---
    last_close = history['Close'].iloc[-1]
    prior_high = history['High'].rolling(window=20).max().shift(1)
    breakout = last_close > prior_high.iloc[-1] if not prior_high.empty else False
    window = history['Close'].tail(30)
    cup = False
    if len(window) == 30:
        trough_idx = window.idxmin()
        trough_pos = list(window.index).index(trough_idx)
        cup = (window.iloc[0] > window.min() < window.iloc[-1]) and (10 <= trough_pos <= 20)
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
    # Cache pattern for display outside market hours
    if 'pattern_msg' not in st.session_state:
        st.session_state.pattern_msg = {}
    st.session_state.pattern_msg[symbol] = pattern_msg
    st.write(f"🧠 Pattern Recognition: {st.session_state.pattern_msg.get(symbol, 'No pattern detected')}")
    st.markdown("### 📊 Debug Info")
    st.write("Last 5 Rows of History:")
    st.dataframe(history.tail(5))
    st.write("Shape:", history.shape)
    st.write("Columns:", history.columns.tolist())

    # 📰 News Sentiment Analysis
    headlines_with_source = get_headlines(symbol)
    overall_sentiment, per_source_scores = headline_sentiment_score(headlines_with_source)
    st.markdown("### 📰 News Sentiment Analysis")
    st.write(f"**Overall Sentiment (Weighted Average):** {overall_sentiment:.2f}")
    st.write("**Sentiment by Source:**")
    for source, score in per_source_scores.items():
        st.write(f"- {source}: {score:.2f} (Weight: {SOURCE_WEIGHTS.get(source, 0.1)})")
    
    if headlines_with_source:
        st.write("**Recent Headlines by Source:**")
        # Group headlines by source
        headlines_by_source = {}
        for headline, source in headlines_with_source:
            if source not in headlines_by_source:
                headlines_by_source[source] = []
            headlines_by_source[source].append(headline)
        
        for source, source_headlines in headlines_by_source.items():
            st.write(f"**{source}:**")
            for h in source_headlines[:3]:  # Limit to 3 per source
                st.write(f"- {h}")
    else:
        st.write("No recent headlines found.")

    if overall_sentiment > 0.2:
        confidence = min(100, confidence + 10)
    elif overall_sentiment < -0.2:
        confidence = max(0, confidence - 10)

    # --- 6️⃣ Strategy Logic & Chart ---
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

            # Adjust confidence based on SPY trend
            confidence_msg = "Moderate"
            if trend_bias.startswith("Bullish"):
                if spy_trend == "Bullish":
                    confidence += 10  # Boost confidence if SPY aligns
                    confidence_msg = "High (Market Alignment)"
                elif spy_trend == "Bearish":
                    confidence -= 10  # Reduce confidence if SPY contradicts
                    confidence_msg = "Low (Market Divergence)"
                if macd_val > signal_val and vol_val > avg_vol_val and rsi_val < 65:
                    confidence_msg = "Very High"
                elif macd_val > signal_val and rsi_val < 65:
                    confidence_msg = "High"
                elif 45 <= rsi_val <= 55:
                    confidence_msg = "Low"
            else:
                if spy_trend == "Bearish":
                    confidence += 10  # Boost confidence if SPY aligns
                    confidence_msg = "High (Market Alignment)"
                elif spy_trend == "Bullish":
                    confidence -= 10  # Reduce confidence if SPY contradicts
                    confidence_msg = "Low (Market Divergence)"
                if macd_val < signal_val and vol_val > avg_vol_val and rsi_val > 35:
                    confidence_msg = "Very High"
                elif macd_val < signal_val and rsi_val > 35:
                    confidence_msg = "High"
                elif 45 <= rsi_val <= 55:
                    confidence_msg = "Low"
            confidence = max(0, min(100, confidence))  # Keep confidence within 0-100

            # Display prediction with market context
            st.subheader(f"📌 Suggested Strategy: {trend_bias} — {confidence_msg}")
            st.write(f"📊 RSI: {rsi_val}")
            st.write(f"📊 MACD: {macd_val} vs Signal: {signal_val}")
            st.write(f"📊 Volume: {vol_val} vs AvgVol: {avg_vol_val}")
            st.write(f"**Market Context:** SPY is {spy_trend}, RSI at {spy_rsi}. Confidence adjusted accordingly.")

            # --- 7️⃣ Probability Analysis ---
            symbol_trades = df[df['Symbol'] == symbol]
            target = symbol_trades['Target Price'].iloc[-1] if not symbol_trades.empty else None
            entry = symbol_trades['Entry Price'].iloc[-1] if not symbol_trades.empty else None
            stop = symbol_trades['Stop Loss'].iloc[-1] if not symbol_trades.empty else None
            trade_type = symbol_trades['Trade Type'].iloc[-1] if not symbol_trades.empty else None
            days_to_expiration = symbol_trades['Days to Expiration'].iloc[-1] if not symbol_trades.empty else 1
            implied_vol = symbol_trades['Implied Volatility'].iloc[-1] if not symbol_trades.empty else vol_annual

            if symbol not in st.session_state.latest_probs:
                st.session_state.latest_probs[symbol] = {
                    "target": None,
                    "days_to_expiration": None,
                    "prob_mc": None,
                    "prob_bs": None
                }

            if target is not None and not np.isnan(implied_vol):
                # Monte Carlo Probability (hit probability below target for PUT, above for CALL)
                prob_mc = mc_hit_probability(current_price, target, days_to_expiration, implied_vol)
                if trade_type in ['Put', 'PUT', 'Breakdown']:
                    prob_mc = 1 - prob_mc  # Probability of hitting below target for PUT
                # Black-Scholes Probability (expiration probability below target for PUT, above for CALL)
                prob_bs = bs_itm_prob(current_price, target, days_to_expiration/252, implied_vol, dividend_yield)
                if trade_type in ['Put', 'PUT', 'Breakdown']:
                    prob_bs = 1 - prob_bs  # Probability of being below target at expiration for PUT
                
                st.session_state.latest_probs[symbol] = {
                    "target": target,
                    "days_to_expiration": days_to_expiration,
                    "prob_mc": prob_mc,
                    "prob_bs": prob_bs
                }

                new_prediction = pd.DataFrame([{
                    "Symbol": symbol,
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Trade Type": trade_type,
                    "Target Price": target,
                    "Days to Expiration": days_to_expiration,
                    "Monte Carlo Probability": prob_mc * 100,
                    "Black-Scholes Probability": prob_bs * 100
                }])
                st.session_state.prediction_df = pd.concat(
                    [st.session_state.prediction_df, new_prediction], ignore_index=True
                )
                st.session_state.prediction_df.to_csv(prediction_file, index=False)

                if prob_mc > 0.7 or prob_bs > 0.7:
                    send_pushover_notification(
                        f"High Probability Alert for {symbol}",
                        f"High chance of hitting ${target:.2f} ({trade_type}): MC {prob_mc*100:.2f}%, BS {prob_bs*100:.2f}%"
                    )

            st.subheader(f"📈 Probability Analysis for {symbol}")
            latest_prob = st.session_state.latest_probs.get(symbol, {})
            if latest_prob["target"] is not None:
                action = "below" if trade_type in ['Put', 'PUT', 'Breakdown'] else "above"
                st.write(f"Monte Carlo Probability of hitting ${latest_prob['target']:.2f} {action} within {latest_prob['days_to_expiration']} days: {latest_prob['prob_mc']*100:.2f}%")
                st.write(f"Black-Scholes Probability of being {action} ${latest_prob['target']:.2f} at expiration: {latest_prob['prob_bs']*100:.2f}%")
            else:
                st.warning("⚠️ No target price or valid volatility data available for probability analysis.")

            if 'last_trend_bias' not in st.session_state or st.session_state.last_trend_bias != trend_bias:
                send_pushover_notification("Strategy Update", f"{symbol} trend: {trend_bias} at ${current_price:.2f}")
                st.session_state.last_trend_bias = trend_bias

            # --- 8️⃣ Contract Suggestion ---
            if latest is not None and not np.isnan(current_price):
                # Calculate target based on pattern
                if cup:
                    cup_high = window.iloc[0]  # Left rim
                    cup_low = window.min()     # Bottom
                    cup_depth = cup_high - cup_low
                    target_price = round(cup_high + cup_depth, 2) if trend_bias.startswith("Bullish") else round(cup_low - cup_depth, 2)
                    contract_type = "CALL" if trend_bias.startswith("Bullish") else "PUT"
                    expiration_date = get_next_friday(datetime.now(), days_to_expiration)
                    expiration_str = expiration_date.strftime('%b %d, %Y')
                    strike_price = round(target_price)  # Nearest strike
                    # Approximate probabilities
                    trading_days = sum(1 for i in range((expiration_date - datetime.now()).days + 1) if (datetime.now() + timedelta(days=i)).weekday() < 5)
                    prob_mc = mc_hit_probability(current_price, target_price, trading_days, implied_vol)
                    if contract_type == "PUT":
                        prob_mc = 1 - prob_mc
                    prob_bs = bs_itm_prob(current_price, target_price, trading_days/252, implied_vol, dividend_yield)
                    if contract_type == "PUT":
                        prob_bs = 1 - prob_bs
                    # Adjust rationale based on SPY trend
                    market_note = f"Market ({spy_trend}) supports this move." if (spy_trend == "Bullish" and contract_type == "CALL") or (spy_trend == "Bearish" and contract_type == "PUT") else f"Market ({spy_trend}) contradicts this move."
                    rationale = f"{pattern_msg} supports a {contract_type.lower()} to ${target_price:.2f}. {trend_bias} trend and sentiment {overall_sentiment:.2f} align. {market_note}"
                    risk_tip = f"Set stop-loss at ${round(cup_low, 2) if contract_type == 'CALL' else round(cup_high, 2)}."
                elif triangle:
                    triangle_high = highs.iloc[0]
                    triangle_low = lows.iloc[-1]
                    triangle_depth = triangle_high - triangle_low
                    target_price = round(triangle_low - triangle_depth, 2) if trend_bias.startswith("Bearish") else round(triangle_high + triangle_depth, 2)
                    contract_type = "PUT" if trend_bias.startswith("Bearish") else "CALL"
                    expiration_date = get_next_friday(datetime.now(), days_to_expiration)
                    expiration_str = expiration_date.strftime('%b %d, %Y')
                    strike_price = round(target_price)
                    # Approximate probabilities
                    trading_days = sum(1 for i in range((expiration_date - datetime.now()).days + 1) if (datetime.now() + timedelta(days=i)).weekday() < 5)
                    prob_mc = mc_hit_probability(current_price, target_price, trading_days, implied_vol)
                    if contract_type == "PUT":
                        prob_mc = 1 - prob_mc
                    prob_bs = bs_itm_prob(current_price, target_price, trading_days/252, implied_vol, dividend_yield)
                    if contract_type == "PUT":
                        prob_bs = 1 - prob_bs
                    market_note = f"Market ({spy_trend}) supports this move." if (spy_trend == "Bullish" and contract_type == "CALL") or (spy_trend == "Bearish" and contract_type == "PUT") else f"Market ({spy_trend}) contradicts this move."
                    rationale = f"{pattern_msg} suggests a {contract_type.lower()} to ${target_price:.2f}. {trend_bias} trend supports this. {market_note}"
                    risk_tip = f"Set stop-loss at ${round(triangle_high, 2) if contract_type == 'PUT' else round(triangle_low, 2)}."
                elif breakout:
                    target_price = round(prior_high.iloc[-1] + (prior_high.iloc[-1] - lows.tail(20).min()), 2)
                    contract_type = "CALL"
                    expiration_date = get_next_friday(datetime.now(), days_to_expiration)
                    expiration_str = expiration_date.strftime('%b %d, %Y')
                    strike_price = round(target_price)
                    # Approximate probabilities
                    trading_days = sum(1 for i in range((expiration_date - datetime.now()).days + 1) if (datetime.now() + timedelta(days=i)).weekday() < 5)
                    prob_mc = mc_hit_probability(current_price, target_price, trading_days, implied_vol)
                    prob_bs = bs_itm_prob(current_price, target_price, trading_days/252, implied_vol, dividend_yield)
                    market_note = f"Market ({spy_trend}) supports this move." if spy_trend == "Bullish" else f"Market ({spy_trend}) contradicts this move."
                    rationale = f"Breakout of 20-period high supports a CALL to ${target_price:.2f}. {trend_bias} trend aligns. {market_note}"
                    risk_tip = f"Set stop-loss at ${round(lows.tail(20).min(), 2)}."
                else:
                    target_price, contract_type, expiration_date, expiration_str, strike_price, prob_mc, prob_bs, rationale, risk_tip = None, None, None, None, None, None, None, "No clear pattern for recommendation.", None

                if target_price is not None:
                    st.subheader(f"📝 Recommended Contract for {symbol}")
                    st.write(f"Contract: {symbol} {expiration_str} ${strike_price:.2f} {contract_type}")
                    st.write(f"Target Price: ${target_price:.2f}")
                    st.write(f"Monte Carlo Probability: {prob_mc*100:.2f}%")
                    st.write(f"Black-Scholes Probability: {prob_bs*100:.2f}%")
                    st.write(f"Rationale: {rationale}")
                    st.write(f"Risk Management: {risk_tip}")

            # Chart with markers
            fig = go.Figure(data=[go.Candlestick(
                x=history.index,
                open=history['Open'], high=history['High'],
                low=history['Low'], close=history['Close']
            )])
            fig.add_trace(go.Scatter(x=history.index, y=history['EMA9'], mode='lines', name='EMA9'))
            fig.add_trace(go.Scatter(x=history.index, y=history['EMA21'], mode='lines', name='EMA21'))
            if entry is not None and not np.isnan(entry):
                fig.add_hline(y=entry, line=dict(color='blue', dash='dot'), annotation_text='Entry', annotation_position='top left')
            if target is not None and not np.isnan(target):
                fig.add_hline(y=target, line=dict(color='green', dash='dash'), annotation_text='Target', annotation_position='top right')
            if stop is not None and not np.isnan(stop):
                fig.add_hline(y=stop, line=dict(color='red', dash='dash'), annotation_text='Stop', annotation_position='bottom right')
            fig.update_layout(title=f"{symbol} Chart", height=300)
            st.plotly_chart(fig, use_container_width=True)

# --- 9️⃣ Prediction History ---
st.subheader("📜 Prediction History")
if st.session_state.prediction_df.empty:
    st.info("No predictions logged yet.")
else:
    for symbol in symbols:
        symbol_preds = st.session_state.prediction_df[st.session_state.prediction_df['Symbol'] == symbol]
        if not symbol_preds.empty:
            st.markdown(f"### {symbol}")
            st.dataframe(
                symbol_preds.sort_values(by="Timestamp", ascending=False)[[
                    "Timestamp", "Trade Type", "Target Price", "Days to Expiration",
                    "Monte Carlo Probability", "Black-Scholes Probability"
                ]].style.format({
                    "Target Price": "{:.2f}",
                    "Monte Carlo Probability": "{:.2f}%",
                    "Black-Scholes Probability": "{:.2f}%"
                })
            )

# --- 🔟 Summary of Trades by Symbol ---
st.subheader("📌 Trade Summary by Symbol")
if df.empty:
    st.info("No trades logged yet.")
else:
    fig_summary = df.copy()
    fig_summary['Profit/Loss'] = fig_summary.apply(
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

pred_csv = st.session_state.prediction_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Prediction History", data=pred_csv, file_name='prediction_history.csv', mime='text/csv')