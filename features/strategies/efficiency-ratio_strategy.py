
import pandas as pd

# Efficiency Ratio Strategy
def efficiency_ratio_signals(stock_df, window=10):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the changes in closing price
    price_change = stock_df['Close'].diff(window)
    
    # Calculate the sum of price movements
    sum_price_movement = stock_df['Close'].diff().abs().rolling(window=window).sum()
    
    # Calculate Efficiency Ratio
    efficiency_ratio = price_change / sum_price_movement
    
    signals['Efficiency_Ratio'] = efficiency_ratio
    signals['er_signal'] = 'neutral'
    
    # Generate trading signals
    signals.loc[(signals['Efficiency_Ratio'] > 0.5) & (signals['Efficiency_Ratio'].shift(1) <= 0.5), 'er_signal'] = 'long'
    signals.loc[(signals['Efficiency_Ratio'] < 0.2) & (signals['Efficiency_Ratio'].shift(1) >= 0.2), 'er_signal'] = 'short'
    
    return signals
