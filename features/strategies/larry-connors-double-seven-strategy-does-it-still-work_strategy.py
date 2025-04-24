
import pandas as pd

# Larry Connors' Double Seven Trading Strategy
def double_seven_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['close'] = stock_df['Close']
    
    # Calculate the 7-day rolling high and low
    signals['high_7'] = stock_df['Close'].rolling(window=7).max()
    signals['low_7'] = stock_df['Close'].rolling(window=7).min()
    
    # Calculate the previous close to determine entry conditions
    signals['prev_close'] = stock_df['Close'].shift(1)
    
    # Generate signals
    signals['double_seven_signal'] = 'neutral'
    
    # Buy signal: If today's close is less than the 7-day low
    signals.loc[signals['close'] < signals['low_7'], 'double_seven_signal'] = 'long'
    
    # Sell signal: If today's close is greater than the 7-day high
    signals.loc[signals['close'] > signals['high_7'], 'double_seven_signal'] = 'short'
    
    # Drop auxiliary columns
    signals.drop(['high_7', 'low_7', 'prev_close'], axis=1, inplace=True)

    return signals
