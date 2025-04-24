
import pandas as pd

# CANSLIM Strategy
def canslim_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Conditions for the CANSLIM strategy
    # C - Current quarterly earnings per share growth > 25%
    signals['C'] = stock_df['EPS_Growth_Qtr'].rolling(window=4).mean() > 25
    
    # A - Annual earnings growth >= 25%
    signals['A'] = stock_df['EPS_Growth_Ann'] >= 25
    
    # N - New product, service, or management
    signals['N'] = stock_df['New_Product_Service_Management'] == 1
    
    # S - Supply and demand (price strength)
    signals['S'] = stock_df['Price_Strength'] >= 1.5  # Example threshold
    
    # L - Leader or laggard (relative strength)
    signals['L'] = stock_df['Relative_Strength'] >= 70  # Example threshold
    
    # I - Institutional sponsorship
    signals['I'] = stock_df['Institutional_Ownership'] > 0.5  # Example threshold
    
    # M - Market direction
    signals['M'] = stock_df['Market_Trend'] == 'Bull'  # Assuming 'Bull' or 'Bear'
    
    # Combine the signals into a final trading signal
    signals['signal'] = 'neutral'
    signals.loc[signals['C'] & signals['A'] & signals['N'] & signals['S'] & signals['L'] & signals['I'] & signals['M'], 'signal'] = 'long'
    
    return signals[['signal']]

