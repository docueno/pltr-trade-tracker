import streamlit as st
import pandas as pd
from datetime import datetime
import yfinance as yf
import plotly.graph_objs as go
import requests

# ✅ Safely load secrets with .get() to avoid KeyErrors
PUSHOVER_USER_KEY = st.secrets.get("pushover_user_key", "")
PUSHOVER_API_TOKEN = st.secrets.get("pushover_api_token", "")

def send_pushover_notification(title, message):
    if PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN:
        data = {
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": title,
            "message": message
        }
        requests.post("https://api.pushover.net/1/messages.json", data=data)

st.set_page_config(page_title="PLTR Trade Tracker", layout="wide")
st.title("📈 PLTR Day Trade Tracker with Charts & Alerts")

# Test notification button
if st.button("📨 Send Test Notification to iPhone"):
    send_pushover_notification("✅ Test Notification", "Pushover is successfully connected!")
    st.success("Test notification sent to your iPhone.")

@st.cache_data
def load_data():
    return pd.DataFrame(columns=[
        "Date", "Trade Type", "Entry Price", "Exit Price", "Stop Loss",
        "Target Price", "Actual Result", "Notes"
    ])

df = load_data()

with st.form("trade_form"):
    st.subheader("Log a New Trade")
    trade_type = st.selectbox("Trade Type", ["Breakout", "Breakdown", "Range", "Call", "Put"])
    entry = st.number_input("Entry Price", format="%.2f")
    exit_price = st.number_input("Exit Price", format="%.2f")
    stop = st.number_input("Stop Loss", format="%.2f")
    target = st.number_input("Target Price", format="%.2f")
    result = st.selectbox("Trade Result", ["Pending", "Win", "Loss", "Breakeven"])
    notes = st.text_area("Notes")
    submitted = st.form_submit_button("Add Trade")

    if submitted:
        new_row = pd.DataFrame.from_dict({
            "Date": [datetime.today().strftime('%Y-%m-%d')],
            "Trade Type": [trade_type],
            "Entry Price": [entry],
            "Exit Price": [exit_price],
            "Stop Loss": [stop],
            "Target Price": [target],
            "Actual Result": [result],
            "Notes": [notes]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        st.success("Trade added successfully!")

df["Profit/Loss"] = df.apply(lambda row:
    round((row["Exit Price"] - row["Entry Price"]), 2) if row["Trade Type"] in ["Call", "Breakout"]
    else round((row["Entry Price"] - row["Exit Price"]), 2) if row["Trade Type"] in ["Put", "Breakdown"]
    else 0, axis=1)

ticker = yf.Ticker("PLTR")
current_price = ticker.history(period="1d", interval="1m").iloc[-1]["Close"]
st.metric(label="🔔 Live PLTR Price", value=f"${current_price:.2f}")

for index, row in df.iterrows():
    col1, col2 = st.columns([2, 3])
    with col1:
        st.write(f"**{row['Trade Type']}** | Entry: ${row['Entry Price']} → Target: ${row['Target Price']} | Stop: ${row['Stop Loss']}")
        st.write(f"📋 Notes: {row['Notes']}")
        st.write(f"Status: {row['Actual Result']} | P/L: {row['Profit/Loss']}")

        if row["Actual Result"] == "Pending":
            if (row["Trade Type"] in ["Call", "Breakout"] and current_price >= row["Target Price"]) or \
               (row["Trade Type"] in ["Put", "Breakdown"] and current_price <= row["Target Price"]):
                st.success(f"📈 Target hit at ${current_price:.2f}")
                send_pushover_notification("PLTR Target Hit", f"{row['Trade Type']} trade hit target at ${current_price:.2f}")
            elif (row["Trade Type"] in ["Call", "Breakout"] and current_price <= row["Stop Loss"]) or \
                 (row["Trade Type"] in ["Put", "Breakdown"] and current_price >= row["Stop Loss"]):
                st.error(f"⚠️ Stop loss hit at ${current_price:.2f}")
                send_pushover_notification("PLTR Stop Hit", f"{row['Trade Type']} trade hit stop at ${current_price:.2f}")

    with col2:
        history = ticker.history(period="2d", interval="5m")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=history.index,
            open=history['Open'],
            high=history['High'],
            low=history['Low'],
            close=history['Close'],
            name='PLTR'
        ))
        fig.add_hline(y=row['Entry Price'], line=dict(color='blue', dash='dot'))
        fig.update_layout(title=f"PLTR Chart - Entry ${row['Entry Price']}", height=300)
        st.plotly_chart(fig, use_container_width=True)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", data=csv, file_name='pltr_day_trade_tracker.csv', mime='text/csv')

st.subheader("📌 Summary")
if not df.empty:
    st.write(f"Total Trades: {len(df)}")
    st.write(f"Wins: {len(df[df['Actual Result'] == 'Win'])}")
    st.write(f"Losses: {len(df[df['Actual Result'] == 'Loss'])}")
    st.write(f"Breakevens: {len(df[df['Actual Result'] == 'Breakeven'])}")
    st.write(f"Pending: {len(df[df['Actual Result'] == 'Pending'])}")
    st.write(f"Total Net P/L: {round(df['Profit/Loss'].sum(), 2)} points")
else:
    st.info("No trades logged yet.")
