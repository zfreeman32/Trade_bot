
import pandas as pd

# ISM Manufacturing Index Trading Strategy
def ism_pmi_signals(pmi_data):
    """
    Generate trading signals based on ISM Manufacturing Index PMI readings.
    
    Parameters:
    pmi_data (pd.DataFrame): DataFrame containing PMI data with a 'PMI' column.
    
    Returns:
    pd.DataFrame: DataFrame containing the trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=pmi_data.index)
    signals['PMI'] = pmi_data['PMI']
    signals['pmi_signal'] = 'neutral'
    
    # Generate signals based on the PMI readings
    signals.loc[signals['PMI'] > 50, 'pmi_signal'] = 'long'
    signals.loc[signals['PMI'] < 50, 'pmi_signal'] = 'short'
    
    return signals[['PMI', 'pmi_signal']]
