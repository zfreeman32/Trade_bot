import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from eightmonthavg.py
def eight_month_avg_signals(stock_df, length=8):
    signals = pd.DataFrame(index=stock_df.index)  # Ensure index matches stock_df
    signals['sma'] = stock_df['Close'].rolling(window=length).mean()
    signals['eight_month_avg_buy_signal'] = 0
    signals['eight_month_avg_sell_signal'] = 0

    buy_condition = (stock_df['Close'] > signals['sma']) & (stock_df['Close'].shift(1) <= signals['sma'].shift(1))
    sell_condition = (stock_df['Close'] < signals['sma']) & (stock_df['Close'].shift(1) >= signals['sma'].shift(1))

    # Ensure the conditions' index matches stock_df.index before applying
    buy_condition = buy_condition.reindex(stock_df.index, fill_value=False)
    sell_condition = sell_condition.reindex(stock_df.index, fill_value=False)

    signals.loc[buy_condition, 'eight_month_avg_buy_signal'] = 1
    signals.loc[sell_condition, 'eight_month_avg_sell_signal'] = 1

    signals.drop(['sma'], axis=1, inplace=True)
    return signals
