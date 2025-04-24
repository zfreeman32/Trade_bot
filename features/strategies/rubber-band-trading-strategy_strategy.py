
import pandas as pd
from ta.volatility import KeltnerChannel

# Rubber Band Trading Strategy
def rubber_band_strategy(stock_df, window=20, atr_window=10):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Keltner Channel
    keltner = KeltnerChannel(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=window, window_atr=atr_window)
    
    # Generate signals
    signals['Keltner_High'] = keltner.keltner_channel_hband()
    signals['Keltner_Low'] = keltner.keltner_channel_lband()
    signals['signal'] = 'neutral'
    
    # Long signal conditions
    signals.loc[stock_df['Close'] < signals['Keltner_Low'], 'signal'] = 'long'
    
    # Short signal conditions
    signals.loc[stock_df['Close'] > signals['Keltner_High'], 'signal'] = 'short'
    
    return signals[['signal']]
