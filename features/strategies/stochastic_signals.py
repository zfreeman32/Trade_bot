import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from stochastic.py
def stochastic_signals(stock_df, k=14, d=3, overbought=80, oversold=20):
    signals = pd.DataFrame(index=stock_df.index)
    stoch = momentum.StochasticOscillator(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=k, smooth_window=d)
    
    signals['stoch_k'] = stoch.stoch()
    signals['stoch_d'] = stoch.stoch_signal()
    
    signals['stochastic_strat_buy_signal'] = 0
    signals['stochastic_strat_sell_signal'] = 0
    # Create signal when 'stoch_k' crosses above 'stoch_d'
    signals.loc[signals['stoch_k'] > signals['stoch_d'], 'stochastic_strat_buy_signal'] = 1
    # Create signal when 'stoch_k' crosses below 'stoch_d'
    signals.loc[signals['stoch_k'] < signals['stoch_d'], 'stochastic_strat_sell_signal'] = 1
    
    # Create states of 'overbought' and 'oversold'
    signals['overbought'] = signals['stoch_k'] > overbought
    signals['oversold'] = signals['stoch_k'] < oversold
    signals.drop(['stoch_k', 'stoch_d', 'overbought','oversold'], axis=1, inplace=True)

    return signals
