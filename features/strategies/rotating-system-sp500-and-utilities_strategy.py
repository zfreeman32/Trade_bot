
import pandas as pd

# Weekly Rotating System Between S&P 500 and Utilities
def spy_xlu_rotation_signals(spy_df, xlu_df):
    """
    Generate trading signals for a weekly rotating system between SPY and XLU based on past performance.
    
    Parameters:
    - spy_df: DataFrame containing SPY historical price data with a 'Close' column.
    - xlu_df: DataFrame containing XLU historical price data with a 'Close' column.
    
    Returns:
    - signals: DataFrame containing signals ('long', 'short', 'neutral') for each week.
    """
    # Ensure both dataframes have the same index (dates)
    df = pd.DataFrame(index=spy_df.index)
    df['SPY_Close'] = spy_df['Close']
    df['XLU_Close'] = xlu_df['Close']

    # Calculate weekly returns
    df['SPY_Return'] = df['SPY_Close'].pct_change(periods=5)  # 5 trading days in a week
    df['XLU_Return'] = df['XLU_Close'].pct_change(periods=5)

    # Generate signals based on performance
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 'neutral'  # Default to neutral

    # Determine which asset performed better and set signals
    signals.loc[df['SPY_Return'] > df['XLU_Return'], 'signal'] = 'long'  # Buy SPY
    signals.loc[df['XLU_Return'] > df['SPY_Return'], 'signal'] = 'short'  # Buy XLU

    return signals[['signal']]
