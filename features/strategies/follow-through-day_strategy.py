
import pandas as pd

# Follow Through Day Trading Strategy
def follow_through_day_signals(stock_df, volume_multiplier=1.5):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Price Change'] = stock_df['Close'].diff()
    signals['Volume Change'] = stock_df['Volume'].diff()
    
    # 1. Identify the rally attempts (Day 1, Day 2, Day 3)
    signals['Rally Attempt'] = (signals['Price Change'] > 0).astype(int)
    
    # 2. Ensure that we have a valid rally attempt sequence
    signals['Valid Rally'] = (signals['Rally Attempt'].rolling(window=3).sum() == 3).astype(int)

    # 3. Identify the Follow Through Day (at least Day 4 of a rally attempt)
    signals['Follow Through'] = (
        (signals['Valid Rally'].shift(3) == 1) &  # Previous three days valid
        (signals['Rally Attempt'] == 1) &  # Current day is a rally
        (signals['Price Change'] > 0) &  # Today's price is higher than yesterday's
        (signals['Volume Change'] > signals['Volume'].shift(1) * volume_multiplier)  # Increased volume
    ).astype(int)

    # 4. Generate signals
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['Follow Through'] == 1, 'trading_signal'] = 'long'

    # Optional: Adding conditions for shorting or exiting
    # Define conditions for a short signal or exit here if needed

    return signals[['trading_signal']]
