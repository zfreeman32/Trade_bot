
import pandas as pd

# QS Exit Strategy
def qs_exit_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']
    signals['High'] = stock_df['High']
    
    # Generate sell signals
    signals['qs_exit_signal'] = 'neutral'
    signals.loc[(signals['Close'] > signals['High'].shift(1)), 'qs_exit_signal'] = 'sell'
    
    return signals[['qs_exit_signal']]
