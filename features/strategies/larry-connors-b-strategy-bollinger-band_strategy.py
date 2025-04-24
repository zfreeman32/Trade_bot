
import pandas as pd
import numpy as np
from ta import volatility

# Larry Connors %b Bollinger Band Strategy
def connors_b_percentage_signals(stock_df, window=5, num_std_dev=2):
    # Calculate the Bollinger Bands
    bollinger = volatility.BollingerBands(stock_df['Close'], window=window, window_dev=num_std_dev)
    stock_df['bb_lower'] = bollinger.bollinger_lband()
    stock_df['bb_upper'] = bollinger.bollinger_hband()
    
    # Calculate %b
    stock_df['%b'] = (stock_df['Close'] - stock_df['bb_lower']) / (stock_df['bb_upper'] - stock_df['bb_lower'])
    
    # Generate signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['%b'] = stock_df['%b']
    signals['connors_signal'] = 'neutral'
    
    # Buy signal: %b < 0 (oversold)
    signals.loc[signals['%b'] < 0, 'connors_signal'] = 'long'
    
    # Sell signal: %b > 1 (overbought)
    signals.loc[signals['%b'] > 1, 'connors_signal'] = 'short'
    
    return signals[['%b', 'connors_signal']]
