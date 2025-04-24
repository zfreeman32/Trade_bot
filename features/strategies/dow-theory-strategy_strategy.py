
import pandas as pd

# Dow Theory Trading Strategy
def dow_theory_signals(index_df, high_column='High', low_column='Low', timeframe='14D'):
    """
    Generate trading signals based on the Dow Theory.

    Parameters:
    index_df : pd.DataFrame - Historical data with 'Date', 'High', 'Low'.
    high_column : str - Name of the column containing high prices.
    low_column : str - Name of the column containing low prices.
    timeframe : str - The timeframe for checking the previous high/low.

    Returns:
    pd.DataFrame - DataFrame with signals 'long', 'short', 'neutral'.
    """
    signals = pd.DataFrame(index=index_df.index)
    signals['signal'] = 'neutral'
    
    # Calculate previous high and low for the given timeframe
    index_df['previous_high'] = index_df[high_column].shift(1)
    index_df['previous_low'] = index_df[low_column].shift(1)
    
    # Iterate through the DataFrame to identify signals
    for i in range(1, len(index_df)):
        current_high = index_df[high_column].iloc[i]
        current_low = index_df[low_column].iloc[i]
        previous_high = index_df['previous_high'].iloc[i]
        previous_low = index_df['previous_low'].iloc[i]
        
        if current_high > previous_high and index_df[low_column].iloc[i] > previous_low:
            signals['signal'].iloc[i] = 'long'
        elif current_high < previous_high and index_df[low_column].iloc[i] < previous_low:
            signals['signal'].iloc[i] = 'short'

    return signals
