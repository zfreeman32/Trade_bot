import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def Awesome_Oscillator_signals(stock_df):
    """
    Computes Awesome Oscillator zero-crossing signals.

    Returns:
    A DataFrame with 'Ao_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Awesome Oscillator
    ao_indicator = momentum.AwesomeOscillatorIndicator(high=stock_df['High'], low=stock_df['Low'])
    signals['AO'] = ao_indicator.awesome_oscillator()

    # Generate signals on zero-crossing
    signals['Ao_buy_signal'] = 0
    signals['Ao_sell_signal'] = 0
    signals.loc[(signals['AO'].shift(1) < 0) & (signals['AO'] >= 0), 'Ao_buy_signal'] = 1
    signals.loc[(signals['AO'].shift(1) >= 0) & (signals['AO'] < 0), 'Ao_sell_signal'] = 1

    return signals

#%% 
# KAMA Cross Strategy
