
import pandas as pd
from ta.trend import SMAIndicator

# Fabian Market Timing Model Strategy
def fabian_market_timing_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate 39-week moving averages for S&P 500, Dow Jones, and Utilities sector
    sp500_ma = SMAIndicator(stock_df['S&P500_Close'], window=39).sma_indicator()
    dow_ma = SMAIndicator(stock_df['Dow_Close'], window=39).sma_indicator()
    utilities_ma = SMAIndicator(stock_df['Utilities_Close'], window=39).sma_indicator()

    # Generate 'long', 'short', 'neutral' signals based on moving averages
    signals['S&P500_MA'] = sp500_ma
    signals['Dow_MA'] = dow_ma
    signals['Utilities_MA'] = utilities_ma
    signals['signal'] = 'neutral'

    # Long signal: All three indices must be above their moving averages
    signals.loc[(stock_df['S&P500_Close'] > sp500_ma) & 
                 (stock_df['Dow_Close'] > dow_ma) & 
                 (stock_df['Utilities_Close'] > utilities_ma), 'signal'] = 'long'

    # Short signal: All three indices must be below their moving averages
    signals.loc[(stock_df['S&P500_Close'] < sp500_ma) & 
                 (stock_df['Dow_Close'] < dow_ma) & 
                 (stock_df['Utilities_Close'] < utilities_ma), 'signal'] = 'short'

    return signals[['signal']]
