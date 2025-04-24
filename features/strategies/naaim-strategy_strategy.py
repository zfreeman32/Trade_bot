
import pandas as pd

# NAAIM Exposure Index Trading Strategy
def naaim_exposure_signals(naaim_index_df, threshold_high=60, threshold_low=40):
    signals = pd.DataFrame(index=naaim_index_df.index)
    signals['NAAIM_Index'] = naaim_index_df['NAAIM_Exposure']  # Assuming the DataFrame has a column 'NAAIM_Exposure'

    # Initialize signals
    signals['signal'] = 'neutral'
    
    # Generate trading signals
    signals.loc[signals['NAAIM_Index'] > threshold_high, 'signal'] = 'short'  # Sell signal
    signals.loc[signals['NAAIM_Index'] < threshold_low, 'signal'] = 'long'    # Buy signal
    
    return signals
