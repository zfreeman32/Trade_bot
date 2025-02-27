import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def eom_signals(stock_df, window=14):
    """
    Computes EOM trend signals.

    Returns:
    A DataFrame with 'eom_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute EOM
    eom = volume.EaseOfMovementIndicator(stock_df['High'], stock_df['Low'], stock_df['Volume'], window)
    signals['EOM'] = eom.ease_of_movement()

    # Generate signals
    signals['eom_buy_signal'] = 0
    signals['eom_sell_signal'] = 0
    signals.loc[signals['EOM'] > 0, 'eom_buy_signal'] = 1
    signals.loc[signals['EOM'] < 0, 'eom_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['EOM'], axis=1, inplace=True)

    return signals

