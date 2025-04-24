
import pandas as pd

# Day Trading Price Action Strategy
def price_action_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['price_change'] = stock_df['Close'].diff()
    
    # Determine swing highs and lows
    signals['swing_high'] = stock_df['Close'][(stock_df['Close'] > stock_df['Close'].shift(1)) & 
                                               (stock_df['Close'] > stock_df['Close'].shift(-1))]
    signals['swing_low'] = stock_df['Close'][(stock_df['Close'] < stock_df['Close'].shift(1)) & 
                                              (stock_df['Close'] < stock_df['Close'].shift(-1))]
    
    signals['signal'] = 'neutral'
    
    # Generate long signals based on swing lows
    signals.loc[signals['swing_low'].notnull(), 'signal'] = 'long'
    
    # Generate short signals based on swing highs
    signals.loc[signals['swing_high'].notnull(), 'signal'] = 'short'
    
    signals.drop(['price_change', 'swing_high', 'swing_low'], axis=1, inplace=True)
    
    return signals
