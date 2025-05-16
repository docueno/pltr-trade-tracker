import streamlit as st
import pandas as pd
from datetime import datetime
import yfinance as yf
import plotly.graph_objs as go
import requests

# 🔒 Pushover credentials (set in Streamlit Secrets)
# In Streamlit Cloud: Settings → Secrets
# Add these two lines:
# pushover_user_key = "YOUR_USER_KEY"
# pushover_api_token = "YOUR_APP_TOKEN"
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

# --- Streamlit Page Setup ---
st.set_page_config(page_title="PLTR Day Trade Tracker", layout="wide")
st.title("📈 PLTR Day Trade Tracker with Charts & Alerts")

# --- 1️⃣ Load or Initialize Trade Data ---
@st.cache_data
def load_data():
    return pd.DataFrame(columns=[
        "Date", "Trade Type", "Entry Price", "Exit Price",
        "Stop Loss", "Target Price", "Actual Result",
        "Confidence", "Notes"
    ])

df = load_data()

# --- 2️⃣ Trade Logging Form ---
with st.form("trade_form"):
    st.subheader("Log a New Trade")
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
        new_row = pd.DataFrame.from_dict({
            "Date": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            "Trade Type": [trade_type],
            "Entry Price": [entry],
            "Exit Price": [exit_price],
            "Stop Loss": [stop],
            "Target Price": [target],
            "Actual Result": [result],
            "Confidence": [confidence],
            "Notes": [notes]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        st.success("Trade added successfully!")

# --- 3️⃣ Fetch Market Data and Calculate Indicators ---
ticker = yf.Ticker("PLTR")
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

# --- 4️⃣ Debug Info (Visible on Main Page) ---
st.markdown("### 📊 Debug Info")
st.write("Last 5 Rows of History:")
st.dataframe(history.tail(5))
st.write("Shape:", history.shape)
st.write("Columns:", history.columns.tolist())

# --- 5️⃣ Strategy Logic ---
# Select last fully populated candle
valid = history.dropna(subset=["EMA9", "EMA21", "RSI"])
if valid.empty:
    st.warning("⚠️ Not enough valid data to compute predictions.")
else:
    latest = valid.iloc[-1]
    current_price = latest['Close']

    # Live Price
    st.metric(label="🔔 Live PLTR Price", value=f"${current_price:.2f}")

    # Trend and RSI
    trend_bias = "Bullish (CALL)" if latest['EMA9'] > latest['EMA21'] else "Bearish (PUT)"
    rsi_val = round(latest['RSI'], 2)

    # Confidence Level
    confidence_msg = "Moderate"
    if trend_bias.startswith("Bullish") and rsi_val < 65:
        confidence_msg = "High"
    elif trend_bias.startswith("Bearish") and rsi_val > 35:
        confidence_msg = "High"
    elif 45 <= rsi_val <= 55:
        confidence_msg = "Low"

    # Display Prediction
    st.subheader(f"📌 Suggested Strategy: {trend_bias} — {confidence_msg} Confidence")
    st.write(f"📊 RSI: {rsi_val}")

    # Push Notification on Trend Change
    if 'last_trend_bias' not in st.session_state or st.session_state.last_trend_bias != trend_bias:
        send_pushover_notification("Strategy Update", f"Trend shifted: {trend_bias} at ${current_price:.2f}")
        st.session_state.last_trend_bias = trend_bias

# --- 6️⃣ Display Trades, Charts, P/L, Summary ---
if not df.empty:
    df['Profit/Loss'] = df.apply(lambda row: round((row['Exit Price'] - row['Entry Price']), 2)
                                   if row['Trade Type'] in ['Call', 'Breakout']
                                   else round((row['Entry Price'] - row['Exit Price']), 2), axis=1)

for _, row in df.iterrows():
    col1, col2 = st.columns([2, 3])
    with col1:
        st.write(f"**{row['Trade Type']}** | Entry: ${row['Entry Price']} → Target: ${row['Target Price']} | Stop: ${row['Stop Loss']}")
        st.write(f"📋 Notes: {row['Notes']}")
        st.write(f"Status: {row['Actual Result']} | P/L: {row.get('Profit/Loss', 0)}")
        if row['Actual Result'] == 'Pending':
            if trend_bias.startswith("Bullish") and current_price >= row['Target Price']:
                st.success(f"📈 Target hit at ${current_price:.2f}")
            elif trend_bias.startswith("Bearish") and current_price <= row['Target Price']:
                st.error(f"⚠️ Stop hit at ${current_price:.2f}")
    with col2:
        fig = go.Figure(data=[go.Candlestick(
            x=history.index, open=history['Open'], high=history['High'],
            low=history['Low'], close=history['Close'])])
        fig.add_trace(go.Scatter(x=history.index, y=history['EMA9'], mode='lines', name='EMA9'))
        fig.add_trace(go.Scatter(x=history.index, y=history['EMA21'], mode='lines', name='EMA21'))
        fig.update_layout(title=f"PLTR Chart - Entry ${row['Entry Price']}", height=300)
        st.plotly_chart(fig, use_container_width=True)

# Download & Summary
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", data=csv, file_name='pltr_trades.csv', mime='text/csv')

st.subheader("📌 Summary")
if not df.empty:
    st.write(f"Total Trades: {len(df)}")
    st.write(f"Wins: {len(df[df['Actual Result']=='Win'])}")
    st.write(f"Losses: {len(df[df['Actual Result']=='Loss'])}")
    st.write(f"Breakevens: {len(df[df'Actual Result']=='Breakeven'])}")
    st.write(f"Total Net P/L: {round(df['Profit/Loss'].sum(),2)}")
else:
    st.info("No trades logged yet.")
