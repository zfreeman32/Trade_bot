import pandas as pd
import numpy as np
from ta.trend import SMAIndicator

def golden_triangle_signals(stock_df, average_length=50, confirm_length=20, volume_length=5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    sma_long = SMAIndicator(stock_df['Close'], average_length)
    sma_short = SMAIndicator(stock_df['Close'], confirm_length)
    
    # Identify initial uptrend condition
    signals['uptrend'] = stock_df['Close'] > sma_long.sma_indicator()
    
    # Identify pivot points as local maxima
    signals['pivot'] = ((stock_df['Close'] > stock_df['Close'].shift()) &
                         (stock_df['Close'] > stock_df['Close'].shift(-1)))
    
    # Identify price drop condition
    signals['price_drop'] = stock_df['Close'] < sma_long.sma_indicator()
    
    # Define initial triangle setup condition
    signals['triangle_setup'] = np.where((signals['uptrend'] & 
                                          signals['pivot'] & 
                                          signals['price_drop']).shift().fillna(0), 'yes', 'no')
    
    # Price and volume confirmation
    signals['price_confirm'] = stock_df['Close'] > sma_short.sma_indicator()
    signals['volume_confirm'] = stock_df['Volume'] > stock_df['Volume'].rolling(volume_length).max()
    signals['triangle_confirm'] = np.where((signals['price_confirm'] & 
                                            signals['volume_confirm']).shift().fillna(0), 'yes', 'no')
    
    # Add a simulated Buy order when the triangle is confirmed
    signals['golden_triangle_le'] = np.where((signals['triangle_setup'] == 'yes') & 
                                             (signals['triangle_confirm'] == 'yes'), 'buy', 'wait')

    # Remove intermediate signals used for calculations
    signals = signals.drop(['uptrend', 'pivot', 'price_drop'], axis=1)

    return signals

