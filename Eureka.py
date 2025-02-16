#%% Import Libraries
import sys
import os
import pandas as pd
import requests
import joblib
import numpy as np
import matplotlib.pyplot as plt
from data import preprocess_data
from dotenv import load_dotenv

#%% Stream Live Data
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\currency_data\trade_signals.csv'
data = pd.read_csv(file_path, header=0)
data = data.tail(1000)

#%% Generate Features
df = preprocess_data.generate_features(data)
df_plot = df.tail(240)
features = df.drop(columns=['Close'])
target = df[['Close']]

#%% Load Models
regression_model = joblib.load("./models/regression_model.pkl")
long_classification_model = joblib.load("./models/long_classification_model.pkl")
short_classification_model = joblib.load("./models/short_classification_model.pkl")

#%% Predictions
price_forecast = regression_model.predict(features)
pred_confidence = np.std(price_forecast) * 1.96  # 95% confidence interval

long_signals = long_classification_model.predict(features)
short_signals = short_classification_model.predict(features)

# long_proba = long_classification_model.predict_proba(features)[:, 1]
# short_proba = short_classification_model.predict_proba(features)[:, 1]
# df['long_signals'] = (long_proba > 0.6).astype(int)  # Adjust threshold if needed
# df['short_signals'] = (short_proba > 0.6).astype(int)

# Store Predictions in DataFrame
df['forecast'] = price_forecast
df['confidence_upper'] = df['forecast'] + pred_confidence
df['confidence_lower'] = df['forecast'] - pred_confidence
df['long_signals'] = long_signals
df['short_signals'] = short_signals

#%% Graphing (Only Last 240 Steps)
plt.figure(figsize=(12,6))
plt.plot(df_plot['datetime'], df_plot['Close'], label='Actual Price', color='blue')
plt.plot(df_plot['datetime'], df_plot['forecast'], label='Forecasted Price', color='orange', linestyle='dashed')

# Confidence Interval Shading
plt.fill_between(df_plot['datetime'], df_plot['confidence_lower'], df_plot['confidence_upper'], color='orange', alpha=0.2, label='Confidence Interval')

# Plot Buy/Sell Signals
for i in range(len(df_plot)):
    if df_plot['long_signals'].iloc[i] == 1:
        plt.scatter(df_plot['datetime'].iloc[i], df_plot['Close'].iloc[i], color='green', marker='^', s=100, label='Buy Signal' if i == 0 else "")
    elif df_plot['short_signals'].iloc[i] == 1:
        plt.scatter(df_plot['datetime'].iloc[i], df_plot['Close'].iloc[i], color='red', marker='v', s=100, label='Sell Signal' if i == 0 else "")

plt.legend(loc='upper left')
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Forecast with Buy/Sell Signals')
plt.grid()

#%% Send Signal to Discord (Only if Buy/Sell Signal Detected)
load_dotenv()
bot_token = os.getenv("DISCORD_BOT_TOKEN")
channel_id = os.getenv("DISCORD_TRADE_CHANNEL_ID")

if df['long_signals'].any() or df['short_signals'].any():
    plt.savefig("trade_signal.png")

    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {bot_token}"}
    files = {"file": open("trade_signal.png", "rb")}
    payload = {"content": "Trade signal detected!"}

    response = requests.post(url, headers=headers, files=files, data=payload)
    print(f"Discord Response: {response.status_code}, {response.text}")

plt.show()
