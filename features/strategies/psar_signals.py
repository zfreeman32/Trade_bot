import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# PSAR 
def psar_signals(stock_df, step=0.02, max_step=0.2):
    """
    Computes Parabolic SAR trend direction and signals.

    Returns:
    A DataFrame with 'psar_direction' and 'psar_signal' columns.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute PSAR values
    psar_indicator = trend.PSARIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], step=step, max_step=max_step)
    psar_values = psar_indicator.psar()

    # Generate buy/sell signals based on crossovers
    signals['psar_buy_signal'] = 0
    signals['psar_sell_signal'] = 0
    signals.loc[(stock_df['Close'] > psar_values) & (stock_df['Close'].shift(1) <= psar_values.shift(1)), 'psar_buy_signal'] = 1
    signals.loc[(stock_df['Close'] < psar_values) & (stock_df['Close'].shift(1) >= psar_values.shift(1)), 'psar_sell_signal'] = 1

    return signals
