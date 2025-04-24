
import pandas as pd
from ta.volatility import AverageTrueRange

# Chande Kroll Stop Strategy
def chande_kroll_signals(stock_df, atr_period=14, multiplier=2):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Average True Range (ATR)
    atr = AverageTrueRange(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=atr_period)
    stock_df['ATR'] = atr.average_true_range()
    
    # Chande Kroll Stop calculation
    stock_df['Long_Stop'] = stock_df['Close'] - (stock_df['ATR'] * multiplier)
    stock_df['Short_Stop'] = stock_df['Close'] + (stock_df['ATR'] * multiplier)
    
    # Generate signals based on price action and Chande Kroll Stop
    signals['chande_kroll_signal'] = 'neutral'
    signals.loc[stock_df['Close'] > stock_df['Long_Stop'], 'chande_kroll_signal'] = 'long'
    signals.loc[stock_df['Close'] < stock_df['Short_Stop'], 'chande_kroll_signal'] = 'short'
    
    return signals
