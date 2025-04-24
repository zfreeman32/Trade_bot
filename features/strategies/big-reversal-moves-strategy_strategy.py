
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Membership Plans Trading Strategy
def membership_plans_trading_strategy(stock_df, membership_type='Gold'):
    signals = pd.DataFrame(index=stock_df.index)
    
    if membership_type == 'Platinum':
        signals['price_sma'] = stock_df['Close'].rolling(window=50).mean()
        signals['long_signal'] = np.where(stock_df['Close'] > signals['price_sma'], 'long', 'neutral')
        signals['short_signal'] = np.where(stock_df['Close'] < signals['price_sma'], 'short', 'neutral')
    else:  # Default to Gold membership strategy
        signals['price_sma'] = stock_df['Close'].rolling(window=20).mean()
        signals['long_signal'] = np.where(stock_df['Close'] > signals['price_sma'], 'long', 'neutral')
        signals['short_signal'] = np.where(stock_df['Close'] < signals['price_sma'], 'short', 'neutral')
    
    # Combine signals
    signals['membership_signal'] = np.where(signals['long_signal'] == 'long', 'long',
                                 np.where(signals['short_signal'] == 'short', 'short', 'neutral'))
    
    signals.drop(['long_signal', 'short_signal', 'price_sma'], axis=1, inplace=True)
    
    return signals
