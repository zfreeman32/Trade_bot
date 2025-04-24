
import pandas as pd

# Tesla Stock Price Boom Strategy
def tesla_stock_boom_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    
    # Assuming the buy signal is triggered on the 6th trading day after a specific event
    # For simplicity, let's say we trigger a buy if the price rises for 3 consecutive days
    price_change = stock_df['Close'].pct_change()
    consecutive_increase = (price_change > 0).rolling(window=3).sum()

    # Generating the buy signal
    signals.loc[consecutive_increase == 3, 'signal'] = 'long'
    
    # Sell after 6 trading days (assuming we keep the same signal until then)
    signals['exit_signal'] = signals['signal'].shift(-6)
    
    # Mark selling point for the 'long' signal
    signals.loc[signals['exit_signal'] == 'long', 'signal'] = 'short'
    
    return signals[['signal']]
