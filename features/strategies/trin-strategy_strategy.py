
import pandas as pd

# TRIN (Arms Index) Trading Strategy
def trin_signals(stock_df, ad_ratio_col='AD_Ratio', ad_volume_col='AD_Volume'):
    """
    Generate trading signals based on the TRIN (Arms Index).
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock data with AD ratio and AD volume.
    ad_ratio_col (str): Name of the column for AD Ratio.
    ad_volume_col (str): Name of the column for AD Volume.
    
    Returns:
    pd.DataFrame: DataFrame containing signals 'long', 'short', 'neutral'.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate TRIN
    signals['TRIN'] = stock_df[ad_volume_col] / stock_df[ad_ratio_col]
    
    # Initialize signal column
    signals['trin_signal'] = 'neutral'
    
    # Generate trading signals
    signals.loc[signals['TRIN'] < 1.0, 'trin_signal'] = 'long'
    signals.loc[signals['TRIN'] > 1.0, 'trin_signal'] = 'short'
    
    return signals[['trin_signal']]
