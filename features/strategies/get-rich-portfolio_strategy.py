
import pandas as pd

# Get Rich Portfolio Strategy
def get_rich_portfolio_signals(asset_classes_df):
    signals = pd.DataFrame(index=asset_classes_df.index)
    
    # Assuming asset_classes_df contains returns for different asset classes
    returns = asset_classes_df.pct_change().dropna()

    # Define thresholds for long and short signals
    long_threshold = 0.015  # e.g., if expected return > 1.5%
    short_threshold = -0.01  # e.g., if expected return < -1%

    # Assign signals based on the mean returns of the asset classes
    signals['Avg_Return'] = returns.mean(axis=1)
    signals['portfolio_signal'] = 'neutral'
    
    # Generate trading signals
    signals.loc[signals['Avg_Return'] > long_threshold, 'portfolio_signal'] = 'long'
    signals.loc[signals['Avg_Return'] < short_threshold, 'portfolio_signal'] = 'short'
    
    return signals[['portfolio_signal']]
