
import pandas as pd

# Oversold Market Rebound Strategy
def oversold_market_signals(stock_df, threshold=-10, period=10):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate percentage change over the defined period
    stock_df['returns'] = stock_df['Close'].pct_change(periods=period) * 100
    
    # Identify oversold conditions
    signals['signal'] = 'neutral'
    signals.loc[stock_df['returns'] <= threshold, 'signal'] = 'long'
    
    # Implementing a next-day exit strategy
    signals.loc[signals['signal'] == 'long', 'exit'] = signals['signal'].shift(-1).fillna('neutral')
    
    return signals[['signal', 'exit']]
