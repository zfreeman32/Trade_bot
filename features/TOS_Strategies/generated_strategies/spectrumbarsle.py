import pandas as pd
import numpy as np

# SpectrumBarsLE Strategy
def spectrum_bars_le_signals(df, length=10):
    signals = pd.DataFrame(index=df.index)
    signals['close_shift'] = df['Close'].shift(length)
    
    # Define the SpectrumBarsLE conditions:
    # Close price is greater than that from a specified number of bars ago 
    signals['spectrum_bars_le_signals'] = np.where(df['Close'] > signals['close_shift'], 'long', 'neutral')

    # Drop unnecessary columns
    signals.drop(columns='close_shift', inplace=True)
    
    return signals