#%%
import pandas as pd
import numpy as np

def rms(data):
    """Calculates the root mean square (RMS)."""
    return np.sqrt(np.mean(data**2, axis=0))

def supersmoother(data, length):
    """Applies the SuperSmoother filter to the input data."""
    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
    c2, c3 = -b1, a1 * a1
    c1 = 1 - c2 - c3

    ss = np.zeros_like(data)
    for i in range(2, len(data)):
        ss[i] = c1 * (data[i] + data[i-1]) / 2 + c2 * ss[i-1] + c3 * ss[i-2]
    
    return ss

def elegant_oscillator(stock_df, rms_length=10, cutoff_length=10, threshold=0.5):
    """Computes the Elegant Oscillator and generates buy/sell signals."""
    signals = pd.DataFrame(index=stock_df.index)
    close_price = stock_df['Close']
    
    # Calculate Root Mean Square (RMS)
    stock_df['rms'] = rms(close_price, rms_length)
    
    # Apply SuperSmoother filter
    stock_df['ss_filter'] = supersmoother(stock_df['rms'], cutoff_length)
    
    # Normalize and apply Inverse Fisher Transform
    min_ss, max_ss = np.min(stock_df['ss_filter']), np.max(stock_df['ss_filter'])
    x = (2 * (stock_df['ss_filter'] - min_ss) / (max_ss - min_ss) - 1)
    stock_df['elegant_oscillator'] = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    # Generate signals
    signals['elegant_oscillator_signal'] = 'neutral'
    signals.loc[(stock_df['elegant_oscillator'] > threshold) & 
                (stock_df['elegant_oscillator'].shift(1) <= threshold), 'elegant_oscillator_signal'] = 'sell'
    signals.loc[(stock_df['elegant_oscillator'] < -threshold) & 
                (stock_df['elegant_oscillator'].shift(1) >= -threshold), 'elegant_oscillator_signal'] = 'buy'

    return signals

