
import pandas as pd

# SPY Down Day Mean Reversion Strategy
def spy_down_day_signals(stock_df):
    """
    Generates trading signals based on the S&P 500 (SPY) down day mean reversion strategy.
    
    The function checks the closing prices of SPY to determine if the previous day was a down day.
    If it was down, a 'long' signal is created for the next day. If it was an up day, a 'short' 
    signal is generated for the next day. Otherwise, the signal is marked as 'neutral'.
    
    Parameters:
    stock_df (DataFrame): A pandas DataFrame containing daily stock data with a 'Close' column.

    Returns:
    DataFrame: A DataFrame containing trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['Prev_Close'] = stock_df['Close'].shift(1)
    signals['Current_Close'] = stock_df['Close']
    
    # Identify down days
    signals['is_down_day'] = signals['Prev_Close'] > signals['Current_Close']
    
    # Assign trading signals
    signals['signal'] = 'neutral'  # Default signal
    signals.loc[signals['is_down_day'], 'signal'] = 'long'
    signals.loc[~signals['is_down_day'] & (signals['Prev_Close'] < signals['Current_Close']), 'signal'] = 'short'
    
    # Cleanup
    signals.drop(['Prev_Close', 'Current_Close', 'is_down_day'], axis=1, inplace=True)
    return signals
