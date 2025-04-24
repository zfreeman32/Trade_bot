
import pandas as pd
import numpy as np

# Dual Momentum Strategy
def dual_momentum_signals(price_df, relative_window=12, absolute_window=6):
    """
    Generate trading signals based on the Dual Momentum strategy.
    
    Parameters:
    - price_df: DataFrame containing 'Close' prices with a DateTime index.
    - relative_window: Window for calculating relative momentum (in months).
    - absolute_window: Window for calculating absolute momentum (in months).
    
    Returns:
    - A DataFrame containing trading signals: 'long', 'short', or 'neutral'.
    """
    signals = pd.DataFrame(index=price_df.index)
    
    # Calculate the returns
    price_df['Returns'] = price_df['Close'].pct_change()
    
    # Calculate absolute momentum
    absolute_momentum = price_df['Close'].rolling(window=absolute_window).mean().shift(1)
    signals['Absolute_Momentum'] = np.where(price_df['Close'] > absolute_momentum, 1, -1)  # 1 for positive momentum, -1 for negative
    
    # Calculate relative momentum (comparing with a benchmark, e.g., S&P 500)
    benchmark_momentum = price_df['Close'].rolling(window=relative_window).mean().shift(1)
    signals['Relative_Momentum'] = np.where(price_df['Close'] > benchmark_momentum, 1, -1)  # 1 for outperforming, -1 for underperforming
    
    # Generate final signals
    signals['Trade_Signal'] = 'neutral'
    signals.loc[(signals['Absolute_Momentum'] == 1) & (signals['Relative_Momentum'] == 1), 'Trade_Signal'] = 'long'
    signals.loc[(signals['Absolute_Momentum'] == -1) & (signals['Relative_Momentum'] == -1), 'Trade_Signal'] = 'short'

    return signals[['Trade_Signal']]
