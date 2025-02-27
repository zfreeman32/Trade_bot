import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Aroon Strategy
def aroon_strategy(stock_df, window=25):
    """
    Computes Aroon trend strength and direction.

    Returns:
    A DataFrame with 'aroon_Trend_Strength', 'aroon_direction_signal', and 'aroon_signal'.
    """
    # Create Aroon Indicator
    aroon = trend.AroonIndicator(stock_df['Close'], stock_df['Low'], window=window)
    
    # Create a DataFrame to store signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['Aroon_Up'] = aroon.aroon_up()
    signals['Aroon_Down'] = aroon.aroon_down()

    # Determine trend strength
    signals['aroon_Trend_signal'] = 'weak'
    signals.loc[(signals['Aroon_Up'] >= 70) | (signals['Aroon_Down'] >= 70), 'aroon_Trend_Strength'] = 'strong'

    # Determine direction signal
    signals['aroon_direction_signal'] = 'bearish'
    signals.loc[signals['Aroon_Up'] > signals['Aroon_Down'], 'aroon_direction_signal'] = 'bullish'

    # Generate trading signal
    signals['aroon_buy_signal'] = 0
    signals['aroon_sell_signal'] = 0
    signals.loc[
        (signals['aroon_direction_signal'] == 'bullish') & 
        (signals['aroon_direction_signal'].shift(1) == 'bearish'), 'aroon_buy_signal'
    ] = 1
    
    signals.loc[
        (signals['aroon_direction_signal'] == 'bearish') & 
        (signals['aroon_direction_signal'].shift(1) == 'bullish'), 'aroon_sell_signal'
    ] = 1

    # Drop temporary columns
    signals.drop(['Aroon_Up', 'Aroon_Down', 'aroon_Trend_signal', 'aroon_direction_signal'], axis=1, inplace=True)

    return signals
