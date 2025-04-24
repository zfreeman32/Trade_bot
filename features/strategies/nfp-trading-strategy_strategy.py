
import pandas as pd
import numpy as np

# NFP (Non-Farm Payrolls) Trading Strategy
def nfp_trading_signals(stock_df, nfp_dates, threshold=50000):
    """
    Generates trading signals based on the Non-Farm Payroll (NFP) report.
    
    Parameters:
    stock_df (DataFrame): DataFrame with 'Close' prices and dates as index.
    nfp_dates (list): List of dates when NFP reports were released.
    threshold (int): Minimum change in NFP to consider a bullish/bearish signal.
    
    Returns:
    DataFrame: A DataFrame containing trading signals ('long', 'short', 'neutral').
    """

    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'

    for date in nfp_dates:
        # Get previous day's close
        previous_close = stock_df.loc[date - pd.Timedelta(days=1), 'Close']
        
        # Simulate NFP data for the purpose of this example (in reality, you would fetch this)
        nfp_change = np.random.randint(-100000, 100000)  # Simulated NFP change for example
        
        if nfp_change > threshold:
            signals.loc[date, 'signal'] = 'long'
        elif nfp_change < -threshold:
            signals.loc[date, 'signal'] = 'short'
        
        # Example of how the market might react; you need an actual market movement model
        current_price = previous_close * (1 + (nfp_change / 1000000))  # Arbitrary price change model
        if signals.loc[date, 'signal'] == 'long':
            stock_df.loc[date:, 'Close'] = current_price  # Future prices up
        elif signals.loc[date, 'signal'] == 'short':
            stock_df.loc[date:, 'Close'] = current_price * (0.99)  # Future prices down by 1%

    return signals
