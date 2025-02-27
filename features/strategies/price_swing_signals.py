import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from priceswing.py
def price_swing_signals(stock_df, swing_type="RSI", length=20, exit_length=20, deviations=2, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Bollinger Bands
    indicator_bb = volatility.BollingerBands(close=stock_df['Close'], window=length, window_dev=deviations)
    stock_df['bb_bbm'] = indicator_bb.bollinger_mavg()
    stock_df['bb_bbh'] = indicator_bb.bollinger_hband()
    stock_df['bb_bbl'] = indicator_bb.bollinger_lband()

    buy_signal_col = 'price_swing_buy_signal'
    sell_signal_col = 'price_swing_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0

    if swing_type == "bollinger":
        # Use Bollinger Bands crossover swing type
        signals.loc[stock_df['Close'] > stock_df['bb_bbh'], sell_signal_col] = 1
        signals.loc[stock_df['Close'] < stock_df['bb_bbl'], buy_signal_col] = 1

    elif swing_type == "RSI":
        # Use RSI crossover swing type
        rsi = momentum.RSIIndicator(close=stock_df['Close'], window=length)
        signals['rsi'] = rsi.rsi()
        signals.loc[signals['rsi'] > overbought, sell_signal_col] = 1
        signals.loc[signals['rsi'] < oversold, buy_signal_col] = 1
        signals.drop(['rsi'], axis=1, inplace=True)
        
    return signals
