
import pandas as pd
import numpy as np

# Bond MOVE Index and TLT Trading Strategy
def bond_move_tlt_signals(tlt_df, move_index_df):
    signals = pd.DataFrame(index=tlt_df.index)
    
    # Calculate the mean and standard deviation of the MOVE index
    move_mean = move_index_df['MOVE'].mean()
    move_std = move_index_df['MOVE'].std()
    
    # Define the high volatility threshold
    high_volatility_threshold = move_mean + 2 * move_std
    
    # Create signal column based on the MOVE index
    signals['signal'] = 'neutral'
    signals.loc[move_index_df['MOVE'] <= high_volatility_threshold, 'signal'] = 'long'
    signals.loc[move_index_df['MOVE'] > high_volatility_threshold, 'signal'] = 'short'
    
    return signals
