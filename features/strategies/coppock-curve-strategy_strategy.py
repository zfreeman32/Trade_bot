
import pandas as pd
import numpy as np

# Coppock Curve Indicator Strategy
def coppock_curve_signals(stock_df, period1=11, period2=14, smooth=10):
    """
    Generates trading signals based on the Coppock Curve Indicator.
    
    Parameters:
    stock_df (DataFrame): DataFrame containing stock price data with a 'Close' column.
    period1 (int): The first period for the Rate of Change calculation.
    period2 (int): The second period for the Rate of Change calculation.
    smooth (int): The smoothing period for the Coppock Curve.
    
    Returns:
    DataFrame: A DataFrame containing the Coppock Curve and trading signals.
    """
    
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Rate of Change
    roc1 = stock_df['Close'].pct_change(periods=period1) * 100
    roc2 = stock_df['Close'].pct_change(periods=period2) * 100
    
    # Calculate Coppock Curve
    coppock_curve = roc1 + roc2
    signals['coppock_curve'] = coppock_curve.rolling(window=smooth).mean()

    # Generate signals
    signals['coppock_signal'] = 'neutral'
    signals.loc[(signals['coppock_curve'] > 0) & (signals['coppock_curve'].shift(1) <= 0), 'coppock_signal'] = 'long'
    signals.loc[(signals['coppock_curve'] < 0) & (signals['coppock_curve'].shift(1) >= 0), 'coppock_signal'] = 'short'
    
    return signals[['coppock_curve', 'coppock_signal']]
