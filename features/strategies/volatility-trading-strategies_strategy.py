
import pandas as pd
import numpy as np
from ta import trend, volatility

# Volatility Trading Strategy based on 200-day moving average
def volatility_trading_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the 200-day moving average
    signals['200_sma'] = trend.SMAIndicator(stock_df['Close'], window=200).sma_indicator()
    
    # Calculate the historical volatility (using a rolling window of 21 days)
    signals['volatility'] = volatility.AverageTrueRange(stock_df['High'], stock_df['Low'], stock_df['Close'], window=21).average_true_range()
    
    # Create conditions for trading signals
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] < signals['200_sma']) & (signals['volatility'] > signals['volatility'].rolling(window=21).mean()), 'signal'] = 'short'
    
    # Return the resulting signals DataFrame with the trading signals
    return signals[['signal']]
