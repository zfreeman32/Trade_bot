import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from gandalfprojectresearchsystem.py
def gandalf_signals(df, exit_length=10, gain_exit_length=20):  
    signals = pd.DataFrame(index=df.index)

    # Calculating ohlc4, median price, and mid-body price
    df['ohlc4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['median_price'] = (df['High'] + df['Low']) / 2
    df['mid_body'] = (df['Open'] + df['Close']) / 2

    # Initialize buy/sell signals as integer 0
    signals['gandalf_buy_signal'] = 0  
    signals['gandalf_sell_signal'] = 0  

    # Buy signal logic
    buy_condition = (
        ((df['ohlc4'].shift(1) < df['median_price'].shift(1)) &
         (df['median_price'].shift(2) <= df['ohlc4'].shift(1)) &
         (df['median_price'].shift(2) <= df['ohlc4'].shift(3))) |
        ((df['ohlc4'].shift(1) < df['median_price'].shift(3)) &
         (df['mid_body'] < df['median_price'].shift(2)) &
         (df['mid_body'].shift(1) < df['mid_body'].shift(2)))
    )
    signals.loc[buy_condition, 'gandalf_buy_signal'] = 1

    # Identify buy timestamps
    buy_indices = signals.index[signals['gandalf_buy_signal'] == 1]

    for buy_index in buy_indices:
        buy_timestamp = signals.index[signals.index == buy_index]
        
        if not buy_timestamp.empty:
            buy_timestamp = buy_timestamp[0]  # Get the timestamp
            
            if buy_timestamp in df.index:
                buy_open_price = df.at[buy_timestamp, 'Open']  # Scalar value access
                
                sell_condition = (
                    (signals.index >= buy_timestamp + pd.Timedelta(days=exit_length)) |
                    ((signals.index >= buy_timestamp + pd.Timedelta(days=gain_exit_length)) & 
                     (df['Close'] > buy_open_price)) |
                    ((df['Close'] < buy_open_price) &
                     (((df['ohlc4'].shift(-1) < df['mid_body'].shift(-1)) &
                       (df['median_price'].shift(-2) == df['mid_body'].shift(-3)) &
                       (df['mid_body'].shift(-1) <= df['mid_body'].shift(-4))) |
                      ((df['ohlc4'].shift(-2) < df['mid_body']) &
                       (df['median_price'].shift(-4) < df['ohlc4'].shift(-3)) &
                       (df['mid_body'].shift(-1) < df['ohlc4'].shift(-1)))))
                )
                
                signals.loc[sell_condition, 'gandalf_sell_signal'] = 1

    # Ensure integer output to avoid float conversion
    signals['gandalf_buy_signal'] = signals['gandalf_buy_signal'].astype(int)
    signals['gandalf_sell_signal'] = signals['gandalf_sell_signal'].astype(int)

    return signals
