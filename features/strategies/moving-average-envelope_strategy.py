
import pandas as pd
import numpy as np

# Moving Average Envelope Strategy
def moving_average_envelope_signals(stock_df, ma_window=20, envelope_percentage=0.05):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Moving Average
    stock_df['MA'] = stock_df['Close'].rolling(window=ma_window).mean()
    
    # Calculate the upper and lower envelope
    stock_df['Upper_Envelope'] = stock_df['MA'] * (1 + envelope_percentage)
    stock_df['Lower_Envelope'] = stock_df['MA'] * (1 - envelope_percentage)
    
    # Generate signals
    signals['Signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > stock_df['Upper_Envelope']), 'Signal'] = 'short'
    signals.loc[(stock_df['Close'] < stock_df['Lower_Envelope']), 'Signal'] = 'long'
    
    return signals
