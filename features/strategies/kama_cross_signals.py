import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def kama_cross_signals(stock_df, fast_period=10, slow_period=20):
    """
    Computes KAMA crossover and price cross signals.

    Returns:
    A DataFrame with 'kama_cross_signal' and 'kama_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute Fast and Slow KAMA
    fast_kama = momentum.kama(stock_df['Close'], window=fast_period)
    slow_kama = momentum.kama(stock_df['Close'], window=slow_period)

    # Generate crossover signals
    signals['kama_cross_buy_signal'] = 0
    signals['kama_cross_sell_signal'] = 0
    signals.loc[(fast_kama > slow_kama) & (fast_kama.shift(1) <= slow_kama.shift(1)) & (stock_df['Close'] > fast_kama), 
                'kama_cross_buy_signal'] = 1
    
    signals.loc[(fast_kama < slow_kama) & (fast_kama.shift(1) >= slow_kama.shift(1)) & (stock_df['Close'] < fast_kama), 
                'kama_cross_sell_signal'] = 1

    signals.loc[(stock_df['Close'] > fast_kama) & (stock_df['Close'].shift(1) <= fast_kama.shift(1)), 'kama_cross_buy_signal'] = 1
    signals.loc[(stock_df['Close'] < fast_kama) & (stock_df['Close'].shift(1) >= fast_kama.shift(1)), 'kama_cross_sell_signal'] = 1

    return signals

#%% 
# Stochastic Oscillator Strategy
