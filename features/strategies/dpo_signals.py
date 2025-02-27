import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def dpo_signals(stock_df, window=20):
    """
    Computes DPO trend direction and crossover signals.

    Returns:
    A DataFrame with 'dpo_direction_Signal' and 'dpo_Signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create DPO Indicator
    dpo = trend.DPOIndicator(stock_df['Close'], window)
    signals['DPO'] = dpo.dpo()

    # Determine market direction
    signals['dpo_overbought_signal'] = 0
    signals['dpo_oversold_signal'] = 0
    signals.loc[signals['DPO'] > 0, 'dpo_overbought_signal'] = 1
    signals.loc[signals['DPO'] < 0, 'dpo_oversold_signal'] = 1

    # Generate buy/sell signals based on zero-crossing
    signals['dpo_buy_signal'] = 0
    signals['dpo_sell_signal'] = 0
    signals.loc[(signals['DPO'] > 0) & (signals['DPO'].shift(1) <= 0), 'dpo_buy_signal'] = 1
    signals.loc[(signals['DPO'] < 0) & (signals['DPO'].shift(1) >= 0), 'dpo_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['DPO'], axis=1, inplace=True)

    return signals

#%% 
# Exponential Moving Average (EMA) Crossover Strategy
