
import pandas as pd
import numpy as np

# Anomaly Detection Strategy
def anomaly_detection_signals(stock_df, threshold=1.5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the returns
    stock_df['Return'] = stock_df['Close'].pct_change()
    
    # Calculate the mean and standard deviation of returns
    mean_return = stock_df['Return'].mean()
    std_return = stock_df['Return'].std()
    
    # Calculate the upper and lower anomaly thresholds
    upper_threshold = mean_return + (threshold * std_return)
    lower_threshold = mean_return - (threshold * std_return)
    
    # Initialize signals
    signals['anomaly_signal'] = 'neutral'
    
    # Identify long and short signals
    signals.loc[stock_df['Return'] > upper_threshold, 'anomaly_signal'] = 'long'
    signals.loc[stock_df['Return'] < lower_threshold, 'anomaly_signal'] = 'short'
    
    return signals
