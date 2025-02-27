import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from bbdivergencestrat.py
def BB_Divergence_Strat(dataframe, secondary_data=None):
    signal = []
    high = dataframe['High']
    low = dataframe['Low']
    close = dataframe['Close']

    # Calculate all the necessary indicators
    dataframe['MADivergence3d'] = talib.MAX(dataframe['MADivergence'], timeperiod=3) if 'MADivergence' in dataframe else 0
    dataframe['MIDivergence3d'] = talib.MIN(dataframe['MIDivergence'], timeperiod=3) if 'MIDivergence' in dataframe else 0
    dataframe['ROC'] = talib.ROC(close, timeperiod=3)
    
    # Calculate MACD
    dataframe['MACD'], dataframe['MACDsignal'], dataframe['MACDhist'] = talib.MACD(close)
    
    # Calculate Stochastic
    dataframe['SOK'], dataframe['SOD'] = talib.STOCH(high, low, close)

    # Handle secondary data (if available)
    if secondary_data is not None:
        if 'MADivergence' in secondary_data and 'MIDivergence' in secondary_data:
            secondary_data['MADivergence3d'] = talib.MAX(secondary_data['MADivergence'], timeperiod=3)
            secondary_data['MIDivergence3d'] = talib.MIN(secondary_data['MIDivergence'], timeperiod=3)

    # Define Buy and Sell Conditions
    conditions_buy = (
        (dataframe['MADivergence3d'] > 20) &
        (dataframe['MACD'] > dataframe['MACDsignal']) &
        (dataframe['ROC'] > 0) &
        (dataframe['SOK'] < 85) 
    )
    
    conditions_sell = (
        (dataframe['MIDivergence3d'] < -20) &
        (dataframe['MACD'] < dataframe['MACDsignal']) &
        (dataframe['ROC'] < 0) &
        (dataframe['SOK'] > 85)
    )

    # Assign Buy and Sell signals
    dataframe['BB_Divergence_Strat_buy_signal'] = 0
    dataframe['BB_Divergence_Strat_sell_signal'] = 0
    dataframe.loc[conditions_buy, 'BB_Divergence_Strat_buy_signal'] = 1
    dataframe.loc[conditions_sell, 'BB_Divergence_Strat_sell_signal'] = 1
    
    return dataframe[['BB_Divergence_Strat_buy_signal', 'BB_Divergence_Strat_sell_signal']]
