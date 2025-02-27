import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def stoch_signals(stock_df, fast_period=10, slow_period=20):
    """
    Computes Stochastic Oscillator crossover signals.

    Returns:
    A DataFrame with 'stoch_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Stochastic Oscillator
    stoch = momentum.StochasticOscillator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=14, smooth_window=3)
    signals['%K'] = stoch.stoch()
    signals['%D'] = stoch.stoch_signal()

    # Generate crossover signals
    signals['stoch_buy_signal'] = 0
    signals['stoch_sell_signal'] = 0
    signals.loc[
        (signals['%K'] > signals['%D']) & (signals['%K'].shift(1) <= signals['%D'].shift(1)), 'stoch_buy_signal'
    ] = 1
    
    signals.loc[
        (signals['%K'] < signals['%D']) & (signals['%K'].shift(1) >= signals['%D'].shift(1)), 'stoch_sell_signal'
    ] = 1

    # Drop temporary columns
    signals.drop(['%K', '%D'], axis=1, inplace=True)

    return signals

#%% 
# True Strength Index (TSI) Strategy
