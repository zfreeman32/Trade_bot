import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from rsistrat.py
def rsi_signals(df, length=14, overbought=70, oversold=30, rsi_average_type='simple'):
    close_price = df['Close']

    if rsi_average_type == 'simple':
        rsi = momentum.RSIIndicator(close_price, window=length).rsi()
    elif rsi_average_type == 'exponential':
        rsi = close_price.ewm(span=length, min_periods=length - 1).mean()

    signals = pd.DataFrame(index=df.index)
    signals['RSI'] = rsi

    signals['RSI_strat_buy_signal'] = 0
    signals['RSI_strat_sell_signal'] = 0
    signals.loc[(signals['RSI'] > oversold) & (signals['RSI'].shift(1) <= oversold), 'RSI_strat_buy_signal'] = 1
    signals.loc[(signals['RSI'] < overbought) & (signals['RSI'].shift(1) >= overbought), 'RSI_strat_sell_signal'] = 1
    signals.drop(['RSI'], axis=1, inplace=True)
    return signals
