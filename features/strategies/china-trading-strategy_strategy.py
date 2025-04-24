
import pandas as pd
import numpy as np

# FXI Trading Strategy
def fxi_trading_strategy(stock_df, short_window=20, long_window=50):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Short_MA'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = stock_df['Close'].rolling(window=long_window).mean()

    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1, 0)   
    signals['position'] = signals['signal'].diff()

    signals['trading_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'trading_signal'] = 'long'    # Buy signal
    signals.loc[signals['position'] == -1, 'trading_signal'] = 'short'   # Sell signal
    
    signals.drop(['Short_MA', 'Long_MA', 'signal', 'position'], axis=1, inplace=True)
    return signals
