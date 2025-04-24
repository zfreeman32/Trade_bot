
import pandas as pd
import numpy as np

# Rainbow Oscillator Strategy
def rainbow_oscillator_signals(stock_df, periods=[5, 10, 20, 30, 40, 50, 60]):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    moving_averages = {f'MA_{period}': stock_df['Close'].rolling(window=period).mean() for period in periods}
    
    # Create DataFrame from moving averages
    ma_df = pd.DataFrame(moving_averages)
    
    # Calculate highest high and lowest low of the moving averages
    highest_high = ma_df.max(axis=1)
    lowest_low = ma_df.min(axis=1)
    
    # Calculate Rainbow Oscillator
    rainbow_oscillator = (highest_high - lowest_low) / (highest_high + lowest_low)
    
    signals['Rainbow_Oscillator'] = rainbow_oscillator
    signals['oscillator_signal'] = 'neutral'
    
    # Generate signals based on the Rainbow Oscillator
    signals.loc[(signals['Rainbow_Oscillator'] > 0.1), 'oscillator_signal'] = 'long'
    signals.loc[(signals['Rainbow_Oscillator'] < -0.1), 'oscillator_signal'] = 'short'
    
    return signals
