
import pandas as pd
from ta.momentum import RSIIndicator

# RSI Trading Strategy
def rsi_trading_signals(stock_df, rsi_period=14, overbought=70, oversold=30):
    """
    Generates trading signals based on the RSI trading strategy.
    
    :param stock_df: DataFrame with stock prices (must include 'Close' column)
    :param rsi_period: The number of periods to calculate RSI
    :param overbought: Overbought level for RSI
    :param oversold: Oversold level for RSI
    :return: DataFrame with trading signals ('long', 'short', 'neutral')
    """
    
    signals = pd.DataFrame(index=stock_df.index)
    rsi = RSIIndicator(stock_df['Close'], window=rsi_period)
    signals['RSI'] = rsi.rsi()
    
    signals['signals'] = 'neutral'
    signals.loc[(signals['RSI'] < oversold), 'signals'] = 'long'   # Buy Signal
    signals.loc[(signals['RSI'] > overbought), 'signals'] = 'short' # Sell Signal

    return signals[['signals']]
