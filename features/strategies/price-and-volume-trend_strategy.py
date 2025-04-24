
import pandas as pd

# Price and Volume Trend (PVT) Strategy
def pvt_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Price and Volume Trend (PVT)
    stock_df['PVT'] = (stock_df['Volume'] * (stock_df['Close'].pct_change().fillna(0) + 1)).cumprod()
    
    # Create signal column
    signals['pvt_signal'] = 'neutral'
    
    # Generate buy/sell signals
    signals.loc[(stock_df['PVT'] > stock_df['PVT'].shift(1)) & (stock_df['PVT'].shift(1) <= stock_df['PVT'].shift(2)), 'pvt_signal'] = 'long'
    signals.loc[(stock_df['PVT'] < stock_df['PVT'].shift(1)) & (stock_df['PVT'].shift(1) >= stock_df['PVT'].shift(2)), 'pvt_signal'] = 'short'
    
    return signals
