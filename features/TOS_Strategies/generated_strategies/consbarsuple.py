import pandas as pd

def cons_bars_up_le_signals(stock_df, consec_bars_up=4, price='Close'):
    # Initialize the signal DataFrame with the same index as stock_df
    signals = pd.DataFrame(index=stock_df.index)
    signals['Signal'] = 0.0

    # Create a boolean mask indicating where price bars are consecutively increasing
    mask = stock_df[price].diff() > 0
    
    # Calculate the rolling sum of increasing bars
    rolling_sum = mask.rolling(window=consec_bars_up).sum()
    
    # Finally, as in template, long signals where the count of increasing bars exceeds consec_bars_up
    signals['Signal'][rolling_sum >= consec_bars_up] = 1.0

    # Take difference of signals to identify specific 'long' points
    signals['Signal'] = signals['Signal'].diff()

    # Replace any NaNs with 0
    signals['Signal'].fillna(0.0, inplace=True)

    # Generate trading orders
    signals['cons_bars_up_le_signal'] = 'Hold'
    signals.loc[signals['Signal'] > 0, 'cons_bars_up_le_signal'] = 'Long'
    signals.drop(columns=['Signal'], inplace=True)
    return signals