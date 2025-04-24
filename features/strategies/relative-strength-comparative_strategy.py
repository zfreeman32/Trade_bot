
import pandas as pd

# Relative Strength Comparative (RSC) Strategy
def rsc_signals(stock_df, benchmark_df, period=14):
    """
    Generate trading signals based on the Relative Strength Comparative (RSC) strategy.

    Parameters:
    stock_df (pd.DataFrame): DataFrame with stock price history, must include 'Close'.
    benchmark_df (pd.DataFrame): DataFrame with benchmark index price history, must include 'Close'.
    period (int): Period over which to calculate the RSC.

    Returns:
    pd.DataFrame: DataFrame containing the columns 'RSC' and 'rsc_signal', where 'rsc_signal'
                  indicates 'long', 'short', or 'neutral' trading signals.
    """
    
    # Calculate the returns
    stock_return = stock_df['Close'].pct_change(periods=period)
    benchmark_return = benchmark_df['Close'].pct_change(periods=period)
    
    # Compute RSC
    rsc = stock_return / benchmark_return
    rsc_signal = pd.Series(index=rsc.index, dtype='object')
    
    # Generate trading signals based on RSC's previous values
    rsc_signal.loc[(rsc > 1) & (rsc.shift(1) <= 1)] = 'long'   # Stock outperforms benchmark
    rsc_signal.loc[(rsc < 1) & (rsc.shift(1) >= 1)] = 'short'  # Stock underperforms benchmark
    rsc_signal.fillna('neutral', inplace=True)                   # Fill remaining positions as neutral
    
    # Prepare the results DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['RSC'] = rsc
    signals['rsc_signal'] = rsc_signal
    
    return signals
