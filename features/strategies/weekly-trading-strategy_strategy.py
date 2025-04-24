
import pandas as pd

# Weekly Trading Strategy
def weekly_trading_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    # Calculate the weekly high and low
    stock_df['Weekly_High'] = stock_df['Close'].rolling(window=5).max()
    stock_df['Weekly_Low'] = stock_df['Close'].rolling(window=5).min()
    
    # Generate trading signals based on price movements
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] > stock_df['Weekly_High'].shift(1), 'signal'] = 'long'  # Buy signal
    signals.loc[stock_df['Close'] < stock_df['Weekly_Low'].shift(1), 'signal'] = 'short'  # Sell signal
  
    return signals
