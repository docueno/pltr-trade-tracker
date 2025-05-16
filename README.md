# PLTR Day Trade Tracker App

This is a Streamlit app for tracking Palantir (PLTR) intraday trades, complete with:
- Live chart previews
- Auto-calculated P/L
- Pushbullet alerts on price triggers

## 🚀 To run locally
1. Install Python 3.10 or higher
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.streamlit/secrets.toml` file:
   ```
   pushbullet_api_key = "your-pushbullet-api-key-here"
   ```
4. Run the app:
   ```
   streamlit run pltr_trade_tracker_app.py
   ```

## 🌐 To deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to https://share.streamlit.io/
3. Paste your secret key in the Secrets section
