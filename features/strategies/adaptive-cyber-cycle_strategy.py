
import pandas as pd
import numpy as np

# Adaptive Cyber Cycle Strategy
def adaptive_cyber_cycle_signals(stock_df, length=14, threshold=0.5):
    """
    Generates trading signals based on the Adaptive Cyber Cycle indicator.
    
    Parameters:
    stock_df (DataFrame): DataFrame containing stock prices with a 'Close' column.
    length (int): The length of the cycle period.
    threshold (float): The threshold for signal generation.
    
    Returns:
    DataFrame: A DataFrame containing trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Adaptive Cyber Cycle
    price = stock_df['Close']
    
    # Calculating the cycle component using the Fast Fourier Transform (FFT)
    fft = np.fft.fft(price - price.mean())
    cycle_period = np.round(len(price) / (np.argmax(np.abs(fft[1:])) + 1))
    
    # Adaptive cycle calculation based on current market conditions
    adaptive_cycle = np.sin(np.linspace(0, 2 * np.pi, len(price)) * (2 * np.pi / cycle_period))
    
    # Calculate the phase
    phase = adaptive_cycle - adaptive_cycle.mean()
    
    # Generate signals based on phase thresholds
    signals['adaptive_cycle'] = phase
    signals['signal'] = 'neutral'
    signals.loc[(signals['adaptive_cycle'] > threshold) & (signals['adaptive_cycle'].shift(1) <= threshold), 'signal'] = 'long'
    signals.loc[(signals['adaptive_cycle'] < -threshold) & (signals['adaptive_cycle'].shift(1) >= -threshold), 'signal'] = 'short'
    
    return signals[['signal']]
