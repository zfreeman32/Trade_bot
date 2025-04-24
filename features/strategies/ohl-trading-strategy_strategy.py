
import pandas as pd

# OHL Trading Strategy
def ohl_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Open'] = stock_df['Open']
    signals['High'] = stock_df['High']
    signals['Low'] = stock_df['Low']

    # Initialize 'ohl_signal' column
    signals['ohl_signal'] = 'neutral'

    # Buy signal: Open == Low
    signals.loc[signals['Open'] == signals['Low'], 'ohl_signal'] = 'long'

    # Sell signal: Open == High
    signals.loc[signals['Open'] == signals['High'], 'ohl_signal'] = 'short'

    return signals[['ohl_signal']]
