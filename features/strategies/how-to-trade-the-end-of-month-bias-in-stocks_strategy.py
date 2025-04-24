
import pandas as pd

# Login Strategy
def login_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['login_signal'] = 'neutral'
    # The login strategy is not based on actual stock data but rather on the concept. 
    # Assuming we want to generate synthetic signals based on some predefined criteria.

    # For demonstration purposes, let's say we trigger a 'long' signal if the closing price is above the moving average
    moving_avg = stock_df['Close'].rolling(window=20).mean()
    
    signals.loc[stock_df['Close'] > moving_avg, 'login_signal'] = 'long'
    signals.loc[stock_df['Close'] <= moving_avg, 'login_signal'] = 'short'

    return signals
