import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

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
    
    # Create buy and sell signal columns
    signals['ift_stoch_buy_signal'] = np.where((signals['IFTStoch'] > over_sold) & (signals['IFTStoch'].shift(1) <= over_sold), 1, 0)
    signals['ift_stoch_sell_signal'] = np.where((signals['IFTStoch'] < over_bought) & (signals['IFTStoch'].shift(1) >= over_bought) & (df['Close'] < signals['SMA']), 1, 0)
    
    signals.drop(['IFTStoch', 'SMA'], axis=1, inplace=True)
    
    return signals
