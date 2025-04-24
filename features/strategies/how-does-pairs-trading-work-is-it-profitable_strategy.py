
import pandas as pd
from ta import momentum

# TSI Momentum Strategy
def tsi_momentum_signals(stock_df, window_slow=25, window_fast=13):
    """
    Generates trading signals based on the True Strength Index (TSI) momentum strategy.

    Parameters:
    stock_df (DataFrame): DataFrame containing stock price data with a 'Close' column.
    window_slow (int): The slow period for TSI calculation. Default is 25.
    window_fast (int): The fast period for TSI calculation. Default is 13.

    Returns:
    DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    tsi = momentum.TSIIndicator(stock_df['Close'], window_slow, window_fast)
    signals['TSI'] = tsi.tsi()
    signals['tsi_signal'] = 'neutral'
    
    # Generate long signals when TSI crosses above zero
    signals.loc[(signals['TSI'] > 0) & (signals['TSI'].shift(1) <= 0), 'tsi_signal'] = 'long'
    
    # Generate short signals when TSI crosses below zero
    signals.loc[(signals['TSI'] < 0) & (signals['TSI'].shift(1) >= 0), 'tsi_signal'] = 'short'
    
    signals.drop(['TSI'], axis=1, inplace=True)
    return signals
