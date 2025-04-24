
import pandas as pd
from ta import trend

# Trend Following Strategy based on 200-day moving average
def trend_following_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the 200-day moving average
    signals['200_MA'] = trend.SMAIndicator(stock_df['Close'], window=200).sma_indicator()
    
    # Initialize column for signals
    signals['trend_signal'] = 'neutral'
    
    # Generate long and short signals
    signals.loc[(stock_df['Close'] > signals['200_MA']) & (stock_df['Close'].shift(1) <= signals['200_MA'].shift(1)), 'trend_signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['200_MA']) & (stock_df['Close'].shift(1) >= signals['200_MA'].shift(1)), 'trend_signal'] = 'short'
    
    return signals[['trend_signal']]
