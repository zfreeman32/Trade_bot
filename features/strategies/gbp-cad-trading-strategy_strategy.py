
import pandas as pd
from ta import momentum, trend

# GBP/CAD Trading Strategy
def gbp_cad_signals(gbp_cad_df, short_window=5, long_window=20):
    """
    Generates trading signals for the GBP/CAD currency pair based on moving averages.

    Parameters:
    gbp_cad_df (pd.DataFrame): DataFrame containing 'Close' prices of GBP/CAD.
    short_window (int): Short moving average window.
    long_window (int): Long moving average window.

    Returns:
    pd.DataFrame: DataFrame with trading signals.
    """
    signals = pd.DataFrame(index=gbp_cad_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = gbp_cad_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = gbp_cad_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    
    # Create trading positions
    signals['position'] = signals['signal'].diff()
    signals['trading_signal'] = 'neutral'
    
    # Long and short signals
    signals.loc[signals['position'] == 1, 'trading_signal'] = 'long'  # Buy signal
    signals.loc[signals['position'] == -1, 'trading_signal'] = 'short' # Sell signal
    
    signals.drop(['short_mavg', 'long_mavg', 'signal', 'position'], axis=1, inplace=True)
    
    return signals
