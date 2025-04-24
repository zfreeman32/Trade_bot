
import pandas as pd

# Bullish Candlestick Patterns Strategy
def bullish_candlestick_patterns(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate previous day's high and low
    signals['prev_high'] = stock_df['High'].shift(1)
    signals['prev_low'] = stock_df['Low'].shift(1)
    signals['close'] = stock_df['Close']
    
    # Bearish Engulfing Pattern
    signals['bearish_engulfing'] = (
        (stock_df['Open'] < signals['prev_close']) & 
        (stock_df['Close'] < stock_df['Open']) &
        (signals['prev_close'] > stock_df['Open'].shift(1)) &
        (stock_df['Close'] < signals['prev_high'])
    )

    # Three Outside Down Pattern
    signals['three_outside_down'] = (
        (stock_df['Open'] > stock_df['Close'].shift(1)) &
        (stock_df['Close'] < stock_df['Open'].shift(1)) &
        (stock_df['Open'] < stock_df['Close']) &
        (stock_df['Close'] < stock_df['Open'].shift(1))
    )

    # Bullish Harami Pattern
    signals['bullish_harami'] = (
        (stock_df['Open'].shift(1) > stock_df['Close'].shift(1)) & 
        (stock_df['Close'] > stock_df['Open'].shift(1)) & 
        (stock_df['Close'] < stock_df['Open']) &
        (stock_df['Open'] < stock_df['Close'].shift(1))
    )

    # Combine signals
    signals['long'] = signals['bearish_engulfing'] | signals['three_outside_down'] | signals['bullish_harami']

    # Generate signal column
    signals['signal'] = 'neutral'
    signals.loc[signals['long'], 'signal'] = 'long'
    signals.drop(['prev_high', 'prev_low', 'bearish_engulfing', 'three_outside_down', 'bullish_harami'], axis=1, inplace=True)

    return signals[['signal']]
