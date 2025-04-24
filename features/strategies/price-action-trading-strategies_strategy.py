
import pandas as pd

# Price Action Trading Strategy: Pin Bar
def pin_bar_strategy(stock_df, body_ratio=0.3, wick_ratio=2):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Signal'] = 'neutral'
    
    # Calculate the body and wick sizes
    signals['Body'] = (stock_df['Open'] - stock_df['Close']).abs()
    signals['Upper_Wick'] = stock_df['High'] - stock_df[['Open', 'Close']].max(axis=1)
    signals['Lower_Wick'] = stock_df[['Open', 'Close']].min(axis=1) - stock_df['Low']
    
    # Identify pin bars
    bullish_pin_bar = (signals['Lower_Wick'] > (signals['Body'] * wick_ratio)) & (signals['Upper_Wick'] < (signals['Body'] * body_ratio))
    bearish_pin_bar = (signals['Upper_Wick'] > (signals['Body'] * wick_ratio)) & (signals['Lower_Wick'] < (signals['Body'] * body_ratio))
    
    # Set trading signals based on pin bar conditions
    signals.loc[bullish_pin_bar, 'Signal'] = 'long'
    signals.loc[bearish_pin_bar, 'Signal'] = 'short'
    
    return signals[['Signal']]
