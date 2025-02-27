import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def ichimoku_signals(stock_df, window1=9, window2=26):
    """
    Computes Ichimoku trend direction and crossover signals.

    Returns:
    A DataFrame with 'ichi_signal' and 'ichi_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create Ichimoku Indicator
    ichimoku = trend.IchimokuIndicator(stock_df['High'], stock_df['Low'], window1, window2)
    signals['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
    signals['kijun_sen'] = ichimoku.ichimoku_base_line()
    signals['senkou_span_a'] = ichimoku.ichimoku_a()
    signals['senkou_span_b'] = ichimoku.ichimoku_b()

    # Generate crossover signals
    signals['ichi_buy_signal'] = 0
    signals['ichi_sell_signal'] = 0
    signals.loc[(signals['tenkan_sen'] > signals['kijun_sen']) & 
                (signals['tenkan_sen'].shift(1) <= signals['kijun_sen'].shift(1)), 'ichi_buy_signal'] = 1
    
    signals.loc[(signals['tenkan_sen'] < signals['kijun_sen']) & 
                (signals['tenkan_sen'].shift(1) >= signals['kijun_sen'].shift(1)), 'ichi_sell_signal'] = 1

    # Drop temporary columns
    signals.drop(['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b'], axis=1, inplace=True)

    return signals

#%% 
# Know Sure Thing (KST) Strategy
