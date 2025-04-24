
import pandas as pd
import numpy as np

# Volume Flow Indicator Strategy
def volume_flow_signals(stock_df, window=14, threshold=1.5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate typical price
    typical_price = (stock_df['High'] + stock_df['Low'] + stock_df['Close']) / 3
    # Calculate the change in typical price
    tp_change = typical_price.diff()
    
    # Calculate volume flow
    volume_flow = np.where(tp_change > 0, stock_df['Volume'], -stock_df['Volume'])
    volume_flow_indicator = pd.Series(volume_flow).rolling(window=window).sum()
    
    # Calculate standard deviation for cut-off value
    volatility = np.std(volume_flow_indicator)
    cut_off = threshold * volatility

    # Generate signals based on volume flow indicator and cut-off
    signals['Volume_Flow'] = volume_flow_indicator
    signals['volume_flow_signal'] = 'neutral'
    signals.loc[(signals['Volume_Flow'] > cut_off), 'volume_flow_signal'] = 'long'
    signals.loc[(signals['Volume_Flow'] < -cut_off), 'volume_flow_signal'] = 'short'
    
    signals.drop(['Volume_Flow'], axis=1, inplace=True)
    return signals
