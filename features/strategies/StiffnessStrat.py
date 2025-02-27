import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from stiffnessstrat.py
def StiffnessStrat(df, length=84, average_length=20, exit_length=84, num_dev=2, entry_stiffness_level=90, exit_stiffness_level=50, market_index='Close'):
    
    # Get stiffness and market trend
    sma = trend.SMAIndicator(df['Close'], window=int(average_length)).sma_indicator()
    bollinger = volatility.BollingerBands(df['Close'], window=int(average_length), window_dev=num_dev)
    upper_band = bollinger.bollinger_hband()
    condition = (df['Close'] > sma + upper_band)
    df['stiffness'] = condition.rolling(window=100).sum() / 100 * 100

    if market_index not in df.columns:
        print(f"Warning: {market_index} not found in dataset. Using 'Close' instead.")
        market_index = 'Close'

    df['ema'] = trend.EMAIndicator(df[market_index]).ema_indicator()
    uptrend = (df['ema'] > df['ema'].shift()) & (df['ema'].shift() > df['ema'].shift(2))
    df['uptrend'] = uptrend
    
    # Entry and exit conditions
    entry = (df['uptrend'] & (df['stiffness'] > entry_stiffness_level)).shift()
    exit = ((df['stiffness'] < exit_stiffness_level) | ((df['stiffness'].shift().rolling(window=exit_length).count() == exit_length))).shift()
    
    df['Buy_Signal'] = np.where(entry, 'buy', 'neutral') 
    df['Sell_Signal'] = np.where(exit, 'sell', 'neutral') 

    # Combine into one signal column
    df['stiffness_strat_buy_signal'] = 0
    df['stiffness_strat_sell_signal'] = 0
    df.loc[df['Buy_Signal'] == 'buy', 'stiffness_strat_buy_signal'] = 1
    df.loc[df['Sell_Signal'] == 'sell', 'stiffness_strat_sell_signal'] = 1

    return df[['stiffness_strat_sell_signal', 'stiffness_strat_buy_signal']]
