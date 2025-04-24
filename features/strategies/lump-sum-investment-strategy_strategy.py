
import pandas as pd

# Lump Sum Buy & Hold Strategy
def lump_sum_buy_and_hold_signals(asset_classes, investment_amount):
    """
    Generate signals and performance analysis for the Lump Sum Buy & Hold strategy.

    Parameters:
    asset_classes (list of str): List of 5 asset classes for the portfolio.
    investment_amount (float): Amount to invest in total.

    Returns:
    pd.DataFrame: DataFrame containing the asset classes, allocation, and strategy signal.
    """
    # Define portfolio weights
    weights = [0.2] * 5  # 20% in each asset class
    
    # Create DataFrame for asset allocations
    signals = pd.DataFrame({
        'Asset Class': asset_classes,
        'Weight': weights,
        'Investment Amount': [investment_amount * weight for weight in weights],
        'Signal': ['long'] * 5  # As a buy and hold strategy, we signal 'long' for all assets
    })
    
    return signals

# Example usage:
# asset_classes = ['U.S. Stocks', 'Foreign Stocks', 'U.S. Bonds', 'U.S. REITs', 'World Commodities']
# investment_amount = 10000
# strategy_signals = lump_sum_buy_and_hold_signals(asset_classes, investment_amount)
