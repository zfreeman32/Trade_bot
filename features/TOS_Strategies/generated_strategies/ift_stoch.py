import pandas as pd
import numpy as np
import talib
from ta import trend

# IFT_Stoch Strategy using 'talib'
def ift_stoch_signals(df, length=14, slowing_length=3, over_bought=60, over_sold=30, sma_length=165):
    signals = pd.DataFrame(index=df.index)
    
    # Calculate Stochastic Oscillator using 'talib'
    stoch_k, stoch_d = talib.STOCH(df['High'], df['Low'], df['Close'], 
                                   fastk_period=length, slowk_period=slowing_length, slowd_period=slowing_length)

    signals['IFTStoch'] = 0.5 * ((stoch_k - 50) * 2) ** 2
    
    # Calculate SMA
    sma = trend.SMAIndicator(df['Close'], sma_length)
    signals['SMA'] = sma.sma_indicator()
    
    # Initialize all signals with neutral
    signals['ift_stoch_signals'] = 'neutral'
    
    # Generate signals according to given conditions
    signals.loc[(signals['IFTStoch'] > over_sold) & (signals['IFTStoch'].shift(1) <= over_sold), 'ift_stoch_signals'] = 'buy_to_open'
    signals.loc[(signals['IFTStoch'] < over_bought) & (signals['IFTStoch'].shift(1) >= over_bought) & (df['Close'] < signals['SMA']), 'ift_stoch_signals'] = 'sell_to_open'
    signals.loc[(signals['IFTStoch'] < over_bought) & (signals['IFTStoch'].shift(1) >= over_bought), 'ift_stoch_signals'] = 'sell_to_close'
    signals.loc[(signals['IFTStoch'] > over_sold) & (signals['IFTStoch'].shift(1) <= over_sold), 'ift_stoch_signals'] = 'buy_to_close'
    
    signals.drop(['IFTStoch', 'SMA'], axis=1, inplace=True)
    return signals

