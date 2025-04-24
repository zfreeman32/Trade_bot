
import pandas as pd
import numpy as np

# Fractal Dimension Index (FDI) Strategy
def fdi_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Fractal Dimension Index (FDI)
    price = stock_df['Close']
    log_return = np.log(price / price.shift(1))
    
    # Rolling window of the log returns
    rolling_mean = log_return.rolling(window=window).mean()
    rolling_std = log_return.rolling(window=window).std()
    
    # Calculate FDI
    fdi = 1 + (rolling_std / rolling_mean.abs()).rolling(window=window).mean()
    signals['FDI'] = fdi
    
    # Generate signals based on FDI values
    signals['fdi_signal'] = 'neutral'
    signals.loc[(signals['FDI'] < 1.3), 'fdi_signal'] = 'short'  # Unsustainable trend
    signals.loc[(signals['FDI'] < 1.5) & (signals['FDI'] >= 1.3), 'fdi_signal'] = 'long'  # Trending
    signals.loc[(signals['FDI'] >= 1.5), 'fdi_signal'] = 'neutral'  # Ranging
    
    return signals
