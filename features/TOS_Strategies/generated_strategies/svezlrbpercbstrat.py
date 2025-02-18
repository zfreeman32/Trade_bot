import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands

def sve_zl_rb_perc_b_strat(stock_df, stddev_length=10, ema_length=10, num_dev=2, k_period=14, slowing_period=3):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Initialising Bollinger Bands
    bb = BollingerBands(close=stock_df['Close'], window=stddev_length, window_dev=num_dev)
    
    # Adding to signals df
    signals['percent_b'] = bb.bollinger_pband()
    
    # Initialising Stochastic Oscillator
    stoch = StochasticOscillator(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=k_period, smooth_window=slowing_period)
    
    # Adding to signals df
    signals['stochastic'] = stoch.stoch_signal()
    
    # Creating the signals
    signals['sve_zl_rb_perc_b_strat'] = 'neutral'
    signals.loc[(signals['percent_b'] > signals['percent_b'].shift(1)) & (signals['stochastic'] > signals['stochastic'].shift(1)), 'sve_zl_rb_perc_b_strat'] = 'long'
    signals.loc[(signals['percent_b'] < signals['percent_b'].shift(1)) & (signals['stochastic'] < signals['stochastic'].shift(1)), 'sve_zl_rb_perc_b_strat'] = 'short'
    
    signals.drop(['percent_b', 'stochastic'], axis=1, inplace=True)

    return signals