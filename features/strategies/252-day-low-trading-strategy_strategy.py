
import pandas as pd

# 252-Day Low Trading Strategy
def low_252_day_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the 252-day low
    signals['252_day_low'] = stock_df['Close'].rolling(window=252).min()
    
    # Generate long signals
    signals['long_signal'] = 0
    signals.loc[stock_df['Close'] <= signals['252_day_low'], 'long_signal'] = 1
    
    # Filter only those signals which would be 'long' on the day of hitting 252-day low
    signals['signal'] = 'neutral'
    signals.loc[signals['long_signal'] == 1, 'signal'] = 'long'
    
    # Optional: Reduce to max ten positions logic could be implemented
    signals['position'] = signals['signal'].replace({'long': 1, 'neutral': 0}).cumsum().clip(upper=10)
    
    signals.drop(['252_day_low', 'long_signal'], axis=1, inplace=True)
    
    return signals
