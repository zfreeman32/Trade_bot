
import pandas as pd
import numpy as np

# Treasury Bond Trading Strategy
def treasury_bond_signals(bond_df, short_window=10, long_window=30):
    """
    Generate trading signals for Treasury bonds based on moving averages.

    Parameters:
    bond_df (pd.DataFrame): Dataframe containing bond price data with a 'Close' column.
    short_window (int): Short moving average window.
    long_window (int): Long moving average window.

    Returns:
    pd.DataFrame: Dataframe containing trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=bond_df.index)
    signals['Short_MA'] = bond_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = bond_df['Close'].rolling(window=long_window).mean()
    signals['Signal'] = 0

    signals['Signal'][short_window:] = np.where(
        signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1, 0
    )
    signals['Position'] = signals['Signal'].diff()

    # Generate trade signals
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['Position'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['Position'] == -1, 'trade_signal'] = 'short'

    # Drop temporary columns
    signals.drop(['Short_MA', 'Long_MA', 'Signal', 'Position'], axis=1, inplace=True)

    return signals
