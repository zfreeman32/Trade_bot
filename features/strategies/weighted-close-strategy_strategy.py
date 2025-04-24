
import pandas as pd

# Weighted Close Strategy
def weighted_close_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Weighted Close
    weighted_close = (stock_df['High'] + stock_df['Low'] + (2 * stock_df['Close'])) / 4
    signals['Weighted_Close'] = weighted_close
    
    # Generate signals based on Weighted Close
    signals['signal'] = 'neutral'
    signals.loc[(signals['Weighted_Close'] > stock_df['Close']), 'signal'] = 'short'
    signals.loc[(signals['Weighted_Close'] < stock_df['Close']), 'signal'] = 'long'
    
    return signals
