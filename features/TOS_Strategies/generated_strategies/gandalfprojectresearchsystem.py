import pandas as pd
import numpy as np

# Error:

# GandalfProjectResearchSystem Strategy
def gandalf_signals(df, exit_length=10, gain_exit_length=20):
    df.index = pd.to_datetime(df.index)  # Ensure index is in datetime format
    signals = pd.DataFrame(index=df.index)
    
    # Calculating ohlc4
    df['ohlc4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    # Calculating median price
    df['median_price'] = (df['High'] + df['Low']) / 2
    # Calculating mid-body price
    df['mid_body'] = (df['Open'] + df['Close']) / 2
    
    # Buy signal
    signals['buy_signal'] = 0
    signals.loc[((df['ohlc4'].shift(1) < df['median_price'].shift(1)) &
                 (df['median_price'].shift(2) <= df['ohlc4'].shift(1)) &
                 (df['median_price'].shift(2) <= df['ohlc4'].shift(3))) |
                ((df['ohlc4'].shift(1) < df['median_price'].shift(3)) &
                 (df['mid_body'] < df['median_price'].shift(2)) &
                 (df['mid_body'].shift(1) < df['mid_body'].shift(2))), 'buy_signal'] = 1

    # Sell signal
    signals['sell_signal'] = 0
    buy_indices = signals.index[signals['buy_signal'] == 1]
    
    for buy_index in buy_indices:
        buy_timestamp = signals.index[signals.index == buy_index]
        
        if not buy_timestamp.empty:
            buy_timestamp = buy_timestamp[0]  # Get the timestamp
            
            # Ensure buy_timestamp exists in df before accessing values
            if buy_timestamp in df.index:
                buy_open_price = df.at[buy_timestamp, 'Open']  # Scalar value access
                
                sell_condition = ((signals.index >= buy_timestamp + pd.Timedelta(days=exit_length)) |
                                  ((signals.index >= buy_timestamp + pd.Timedelta(days=gain_exit_length)) & 
                                   (df['Close'] > buy_open_price)) |
                                  ((df['Close'] < buy_open_price) &
                                   (((df['ohlc4'].shift(-1) < df['mid_body'].shift(-1)) &
                                     (df['median_price'].shift(-2) == df['mid_body'].shift(-3)) &
                                     (df['mid_body'].shift(-1) <= df['mid_body'].shift(-4))) |
                                    ((df['ohlc4'].shift(-2) < df['mid_body']) &
                                     (df['median_price'].shift(-4) < df['ohlc4'].shift(-3)) &
                                     (df['mid_body'].shift(-1) < df['ohlc4'].shift(-1))))))
                
                signals.loc[sell_condition, 'sell_signal'] = 1
    
    signals.fillna(0, inplace=True)
    return signals

#%%
