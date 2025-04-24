
import pandas as pd

# Best 6 Months Timing Model Strategy
def best_6_months_timing_signals(stock_df):
    # Ensure the DataFrame has a DateTime index
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df.set_index('Date', inplace=True)

    signals = pd.DataFrame(index=stock_df.index)
    
    # Create month column for easier filtering
    signals['Month'] = stock_df.index.month
    
    # Initialize the signal column
    signals['signal'] = 'neutral'
    
    # Define the trading periods for the Best 6 Months Timing Model (typically May to October)
    signals.loc[(signals['Month'] >= 5) & (signals['Month'] <= 10), 'signal'] = 'long'
    signals.loc[(signals['Month'] < 5) | (signals['Month'] > 10), 'signal'] = 'short'
    
    return signals[['signal']]