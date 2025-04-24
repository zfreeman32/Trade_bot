
import pandas as pd
import numpy as np
from ta import trend, momentum

# Weekend Trend Trader Strategy
def weekend_trend_trader_signals(stock_df, short_window=10, long_window=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    
    # Generate trading signals
    signals['trading_signal'] = 'neutral'
    signals.loc[(signals['signal'] == 1) & (signals['signal'].shift(1) == 0), 'trading_signal'] = 'long'
    signals.loc[(signals['signal'] == 0) & (signals['signal'].shift(1) == 1), 'trading_signal'] = 'short'
    
    return signals[['trading_signal']]
