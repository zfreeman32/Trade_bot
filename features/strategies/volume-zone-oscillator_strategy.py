
import pandas as pd
import numpy as np

# Volume Zone Oscillator (VZO) Strategy
def vzo_signals(stock_df, volume_window=14, ema_window=20):
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Volume Position (VPos) and Total Volume (TV)
    stock_df['Price_Change'] = stock_df['Close'].diff()
    stock_df['VPos_Change'] = np.where(stock_df['Price_Change'] > 0, stock_df['Volume'], 
                                        np.where(stock_df['Price_Change'] < 0, -stock_df['Volume'], 0))
    stock_df['VPos'] = stock_df['VPos_Change'].rolling(window=volume_window).sum()
    stock_df['TV'] = stock_df['Volume'].rolling(window=volume_window).mean()

    # Calculate Volume Zone Oscillator (VZO)
    stock_df['VZO'] = (stock_df['VPos'] / stock_df['TV']) * 100

    # Generate signals
    signals['vzo_signal'] = 'neutral'
    signals.loc[(stock_df['VZO'] > 0) & (stock_df['VZO'].shift(1) <= 0), 'vzo_signal'] = 'long'
    signals.loc[(stock_df['VZO'] < 0) & (stock_df['VZO'].shift(1) >= 0), 'vzo_signal'] = 'short'

    return signals
