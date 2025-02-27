import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def kst_signals(stock_df, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15, nsig=9):
    """
    Computes KST trend direction and crossover signals.

    Returns:
    A DataFrame with 'kst_signal' and 'kst_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create KST Indicator
    kst = trend.KSTIndicator(stock_df['Close'], roc1, roc2, roc3, roc4, window1, window2, window3, window4, nsig)
    signals['KST'] = kst.kst()
    signals['KST_Signal'] = kst.kst_sig()

    # Generate crossover signals
    signals['kst_buy_signal'] = 0
    signals['kst_sell_signal'] = 0
    signals.loc[(signals['KST'] > signals['KST_Signal']) & 
                (signals['KST'].shift(1) <= signals['KST_Signal'].shift(1)), 'kst_buy_signal'] = 1
    
    signals.loc[(signals['KST'] < signals['KST_Signal']) & 
                (signals['KST'].shift(1) >= signals['KST_Signal'].shift(1)), 'kst_sell_signal'] = 1

    # Drop temporary columns
    signals.drop(['KST', 'KST_Signal'], axis=1, inplace=True)

    return signals

#%% 
# Moving Average Convergence Divergence (MACD) Strategy
