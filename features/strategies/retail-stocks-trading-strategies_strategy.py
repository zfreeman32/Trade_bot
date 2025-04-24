
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Seasonal Strategy: Trading at the End of the Month
def end_of_month_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['date'] = stock_df.index
    signals['day'] = signals['date'].dt.day
    signals['signal'] = 'neutral'

    # Generate long signal on last trading day of the month
    last_trading_days = stock_df.index[stock_df.index.is_month_end]
    signals.loc[signals['day'].isin(last_trading_days.day), 'signal'] = 'long'
    
    # Generate short signal on first trading day of the new month
    first_trading_days = stock_df.index[stock_df.index.is_month_start]
    signals.loc[signals['day'].isin(first_trading_days.day), 'signal'] = 'short'

    return signals['signal']
