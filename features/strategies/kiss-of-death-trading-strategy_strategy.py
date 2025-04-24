
import pandas as pd
from ta import trend

# Kiss of Death Trading Strategy
def kiss_of_death_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the 21-period EMA
    stock_df['EMA_21'] = stock_df['Close'].ewm(span=21, adjust=False).mean()
    
    # Identify all-time high
    stock_df['All_Time_High'] = stock_df['Close'].rolling(window=len(stock_df), min_periods=1).max()
    stock_df['Below_ATR'] = stock_df['Close'] < stock_df['All_Time_High']
    
    # Identify when the price drops below the 21 EMA and then bounces back above it
    stock_df['Above_EMA'] = stock_df['Close'] > stock_df['EMA_21']
    stock_df['Recent_Low'] = stock_df['Close'].rolling(window=20).min()
    
    # Identify Kiss of Death conditions
    stock_df['KOD_Signal'] = (
        stock_df['Below_ATR'].shift(1) & 
        ~stock_df['Below_ATR'] & 
        (stock_df['Close'] < stock_df['Recent_Low'])
    )

    # Generate signals
    signals['kod_signal'] = 'neutral'
    signals.loc[stock_df['KOD_Signal'], 'kod_signal'] = 'short'

    return signals
