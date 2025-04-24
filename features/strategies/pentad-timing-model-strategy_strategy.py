
import pandas as pd
import numpy as np

# PENTAD Timing Model Strategy
def pentad_timing_model_signals(data_spy, data_dji, data_trans, data_adv_decline):
    # Ensure indices are aligned
    data = data_spy.join([data_dji, data_trans, data_adv_decline], how='inner', rsuffix='_dji')
    data.columns = ['SPY_Close', 'DJI_Close', 'Trans_Close', 'Adv_Decline_Close']
    
    # Calculate necessary metrics - weekly returns
    data['SPY_Return'] = data['SPY_Close'].pct_change()
    data['DJI_Return'] = data['DJI_Close'].pct_change()
    data['Trans_Return'] = data['Trans_Close'].pct_change()
    data['Adv_Decline_Return'] = data['Adv_Decline_Close'].pct_change()
    
    # Create signals DataFrame
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 'neutral'
    
    # Define entry and exit signals
    signals.loc[
        (data['SPY_Return'] > 0) & 
        (data['DJI_Return'] > 0) & 
        (data['Trans_Return'] > 0) & 
        (data['Adv_Decline_Return'] > 0), 
        'signal'] = 'long'
        
    signals.loc[
        (data['SPY_Return'] < 0) | 
        (data['DJI_Return'] < 0) | 
        (data['Trans_Return'] < 0) | 
        (data['Adv_Decline_Return'] < 0), 
        'signal'] = 'short'
        
    return signals
