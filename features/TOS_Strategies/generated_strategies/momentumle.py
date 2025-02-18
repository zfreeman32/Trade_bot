import pandas as pd
import numpy as np
from ta import momentum

# MomentumLE Strategy
def momentumle_signals(stock_df, length=12, price_scale=100):
    # Initialize the signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Momentum
    mom = momentum.roc(stock_df['Close'], window=length)
    
    # Create 'momentumle_signals' column and set to 'neutral'
    signals['momentumle_signals'] = 'neutral'
    
    # Create a Long Entry signal when Momentum becomes a positive value and continues rising
    signals.loc[(mom > 0) & (mom.shift(1) <= 0), 'momentumle_signals'] = 'long'
    
    # Calculate signal price level
    signals['signal_price_level'] = stock_df['High'] + (1 / price_scale)
    
    # If the next Open price is greater than the current High plus one point, 
    # it is considered a price level to generate the signal at
    signals.loc[stock_df['Open'].shift(-1) > signals['signal_price_level'], 'momentumle_signals'] = 'long'
    signals = signals.drop(['signal_price_level'], axis=1)

    return signals
