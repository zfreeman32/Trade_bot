import pandas as pd
import numpy as np
from ta import trend, volatility

def get_stiffness_indicator(df, length=100, average_length=20, num_dev=2):
    sma = trend.SMAIndicator(df['Close'], window=int(average_length)).sma_indicator()
    bollinger = volatility.BollingerBands(df['Close'], window=int(average_length), window_dev=num_dev)
    upper_band = bollinger.bollinger_hband()
    condition = (df['Close'] > sma + upper_band)
    df['stiffness'] = condition.rolling(window=length).sum() / length * 100
    return df

def get_market_trend(df, market_index='Close', length=2):
    """
    Computes market trend using EMA of the given market index column.
    If market_index does not exist, it defaults to using 'Close' prices.
    """
    if market_index not in df.columns:
        print(f"Warning: {market_index} not found in dataset. Using 'Close' instead.")
        market_index = 'Close'

    df['ema'] = trend.EMAIndicator(df[market_index]).ema_indicator()
    uptrend = (df['ema'] > df['ema'].shift()) & (df['ema'].shift() > df['ema'].shift(2))
    df['uptrend'] = uptrend
    return df 

def StiffnessStrat(df, length=84, average_length=20, exit_length=84, num_dev=2, entry_stiffness_level=90, exit_stiffness_level=50, market_index='Close'):
    
    # Get stiffness and market trend
    df = get_stiffness_indicator(df, length=length, average_length=average_length, num_dev=num_dev)
    df = get_market_trend(df, market_index=market_index)
    
    # Entry and exit conditions
    entry = (df['uptrend'] & (df['stiffness'] > entry_stiffness_level)).shift()
    exit = ((df['stiffness'] < exit_stiffness_level) | ((df['stiffness'].shift().rolling(window=exit_length).count() == exit_length))).shift()
    
    df['Buy_Signal'] = np.where(entry, 'buy', 'neutral') 
    df['Sell_Signal'] = np.where(exit, 'sell', 'neutral') 

    # Combine into one signal column
    df['stiffness_strat_signal'] = 'neutral'
    df.loc[df['Buy_Signal'] == 'buy', 'stiffness_strat_signal'] = 'buy'
    df.loc[df['Sell_Signal'] == 'sell', 'stiffness_strat_signal'] = 'sell'

    return df[['stiffness_strat_signal']]

