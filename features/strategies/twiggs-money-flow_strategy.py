
import pandas as pd
import numpy as np

# Twiggs Money Flow Strategy
def twiggs_money_flow_signals(stock_df, ema_period=21, tmf_threshold=0):
    """
    Twiggs Money Flow trading strategy signals.
    
    Parameters:
    stock_df (DataFrame): DataFrame containing stock price data with 'Close' and 'Volume' columns.
    ema_period (int): The period for the EMA calculation.
    tmf_threshold (float): Threshold for Twiggs Money Flow to identify buy/sell signals.

    Returns:
    DataFrame: Signals DataFrame with 'tmf_signal' (long, short, neutral).
    """
    
    # Calculate the Twiggs Money Flow
    stock_df['money_flow'] = ((stock_df['Close'] - stock_df['Low']) - (stock_df['High'] - stock_df['Close'])) / (stock_df['High'] - stock_df['Low']) * stock_df['Volume']
    stock_df['tmf'] = stock_df['money_flow'].ewm(span=ema_period, adjust=False).mean()
    
    # Generate signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['tmf'] = stock_df['tmf']
    signals['tmf_signal'] = 'neutral'
    
    signals.loc[(signals['tmf'] > tmf_threshold) & (signals['tmf'].shift(1) <= tmf_threshold), 'tmf_signal'] = 'long'
    signals.loc[(signals['tmf'] < -tmf_threshold) & (signals['tmf'].shift(1) >= -tmf_threshold), 'tmf_signal'] = 'short'

    return signals[['tmf_signal']]
