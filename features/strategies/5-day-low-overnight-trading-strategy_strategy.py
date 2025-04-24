
import pandas as pd

# 5-Day Low Overnight Trading Strategy
def five_day_low_overnight_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the 5-day low
    stock_df['5_day_low'] = stock_df['Close'].rolling(window=5).min()
    
    # Generate signals based on the strategy rules
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] < stock_df['5_day_low']) & (stock_df['Close'] > stock_df['Open']), 'signal'] = 'long'
    
    return signals
