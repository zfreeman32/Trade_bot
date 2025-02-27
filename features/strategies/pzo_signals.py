import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Price Zone Oscillator (PZO) Strategy
def pzo_signals(stock_df, length=14, ema_length=60):
    signals = pd.DataFrame(index=stock_df.index)
    pzo = ((stock_df['Close'] - stock_df['Close'].rolling(window=length).mean()) / stock_df['Close'].rolling(window=length).std())*100
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=length).adx()
    ema = stock_df['Close'].ewm(span=ema_length).mean()

    buy_signal_col = 'pzo_buy_signal'
    sell_signal_col = 'pzo_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0

    # ADX > 18, price > EMA, and PZO cross "-40" level or surpass "+15" level from below
    signals.loc[(adx > 18) & (stock_df['Close'] > ema) & (
        (pzo.shift(1) < -40) & (pzo > -40) |
        ((pzo.shift(1) < 0) & (pzo > 0) & (pzo > 15))), buy_signal_col] = 1
    
    # ADX < 18, and PZO cross "-40" or "+15" level from below
    signals.loc[(adx <= 18) & (
        (pzo.shift(1) < -40) & (pzo > -40) |
        (pzo.shift(1) < 15) & (pzo > 15)), buy_signal_col] = 1

    return signals
