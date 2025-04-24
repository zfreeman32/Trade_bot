
import pandas as pd

# War Effects Trading Strategy
def war_effects_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Assuming we use an arbitrary time window to measure volatility
    # This can be adjusted based on desired backtesting parameters
    window = 30 
    
    # Calculate the historical volatility
    signals['Volatility'] = stock_df['Close'].rolling(window).std()
    
    # Generate signals based on historical price movement around significant events (arbitrarily defined)
    signals['Price Change'] = stock_df['Close'].pct_change(periods=window)
    
    # When prices decline significantly, it might suggest a long position
    signals['Signal'] = 'neutral'
    signals.loc[(signals['Price Change'] < -0.05) & (signals['Volatility'].shift(1) < signals['Volatility']), 'Signal'] = 'long'
    
    # When prices increase significantly alongside rising volatility, it might suggest a short position
    signals.loc[(signals['Price Change'] > 0.05) & (signals['Volatility'].shift(1) > signals['Volatility']), 'Signal'] = 'short'
    
    return signals[['Signal']]
