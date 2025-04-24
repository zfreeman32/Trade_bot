
import pandas as pd

# Scale-In Trading Strategy
def scale_in_signals(stock_df, scale_amount=0.1, threshold=0.02):
    """
    This function implements a scale-in trading strategy.
    
    Parameters:
    - stock_df: DataFrame with 'Close' price column.
    - scale_amount: The fraction of total capital to invest with each scale-in.
    - threshold: The percentage drop from the last entry price to trigger a scale-in.

    Returns:
    - signals: DataFrame with trading signals ('long', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    position_size = 0
    total_investment = 0
    last_entry_price = None

    for i in range(len(stock_df)):
        price = stock_df['Close'].iloc[i]
        
        # Check if we should scale in
        if last_entry_price is not None and (last_entry_price - price) / last_entry_price >= threshold:
            # Scale in
            investment = scale_amount * (1 - position_size)
            position_size += scale_amount
            total_investment += investment
            last_entry_price = (last_entry_price * (position_size - scale_amount) + price * scale_amount) / position_size
            signals['signal'].iloc[i] = 'long'
        elif last_entry_price is None:
            # Initial entry
            last_entry_price = price
            position_size += scale_amount
            total_investment += scale_amount
            signals['signal'].iloc[i] = 'long'

    return signals
