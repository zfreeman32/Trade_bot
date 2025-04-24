
import pandas as pd
import numpy as np

# Ultimate Oscillator Trading Strategy
def ultimate_oscillator_signals(stock_df, short_window=7, medium_window=14, long_window=28):
    # Calculate the True Range (TR) and the Average True Range (ATR)
    stock_df['High_Shift'] = stock_df['High'].shift(1)
    stock_df['Low_Shift'] = stock_df['Low'].shift(1)
    stock_df['Close_Shift'] = stock_df['Close'].shift(1)
    stock_df['TR'] = np.maximum(stock_df['High'] - stock_df['Low'], 
                                 np.maximum(abs(stock_df['High'] - stock_df['Close_Shift']),
                                            abs(stock_df['Low'] - stock_df['Close_Shift'])))
    
    # Calculate Buying Pressure (BP)
    stock_df['BP'] = stock_df['Close'] - stock_df['Low_Shift']
    
    # Calculate the averages for each window
    stock_df['avg_BP_short'] = stock_df['BP'].rolling(window=short_window).sum() / stock_df['TR'].rolling(window=short_window).sum()
    stock_df['avg_BP_medium'] = stock_df['BP'].rolling(window=medium_window).sum() / stock_df['TR'].rolling(window=medium_window).sum()
    stock_df['avg_BP_long'] = stock_df['BP'].rolling(window=long_window).sum() / stock_df['TR'].rolling(window=long_window).sum()
    
    # Ultimate Oscillator calculation
    stock_df['UO'] = (4 * stock_df['avg_BP_short'] + 2 * stock_df['avg_BP_medium'] + stock_df['avg_BP_long']) / 7
    
    # Generate signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['UO'] = stock_df['UO']
    signals['uo_signal'] = 'neutral'
    
    # Buy signal when UO crosses above 30
    signals.loc[(signals['UO'] < 30) & (signals['UO'].shift(1) >= 30), 'uo_signal'] = 'long'
    # Sell signal when UO crosses below 70
    signals.loc[(signals['UO'] > 70) & (signals['UO'].shift(1) <= 70), 'uo_signal'] = 'short'
    
    signals.drop(['UO'], axis=1, inplace=True)
    
    return signals
