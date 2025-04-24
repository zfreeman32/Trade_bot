
import pandas as pd
import numpy as np
from ta import momentum, trend

# Membership-Based Trading Strategy
def membership_based_trading_signals(stock_df, membership='Gold', strategies_selected=10, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Assuming we're using a simple moving average strategy as the backbone
    signals['SMA'] = stock_df['Close'].rolling(window=window).mean()
    
    # Trading signals based on comparison with the SMA
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] > signals['SMA'], 'signal'] = 'long'
    signals.loc[stock_df['Close'] < signals['SMA'], 'signal'] = 'short'
    
    # If the membership is Gold or Platinum, limit to selected strategies
    if membership in ['Gold', 'Platinum']:
        total_strategies = ['momentum', 'mean_reversion', 'sma_crossover']
        selected_strategies = total_strategies[:strategies_selected]
        
        if 'momentum' in selected_strategies:
            momentum_indicator = momentum.RSIIndicator(stock_df['Close'], window=window)
            signals['momentum_RSI'] = momentum_indicator.rsi()
            signals.loc[(signals['momentum_RSI'] > 70), 'signal'] = 'short'
            signals.loc[(signals['momentum_RSI'] < 30), 'signal'] = 'long'
        
        if 'mean_reversion' in selected_strategies:
            signals['mean_reversion'] = (stock_df['Close'] - stock_df['Close'].rolling(window=window).mean()) / stock_df['Close'].rolling(window=window).std()
            signals.loc[signals['mean_reversion'] > 1, 'signal'] = 'short'
            signals.loc[signals['mean_reversion'] < -1, 'signal'] = 'long'

    signals.drop(['SMA'], axis=1, inplace=True)
    return signals
