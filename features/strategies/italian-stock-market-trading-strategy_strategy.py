
import pandas as pd

# Italian Stock Market Seasonal Trading Strategy
def italian_stock_market_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['day_of_week'] = stock_df.index.dayofweek  # Monday=0, Sunday=6

    # Initialize signals
    signals['trade_signal'] = 'neutral'

    # Long on Tuesdays (1) and Wednesdays (2)
    signals.loc[signals['day_of_week'] == 1, 'trade_signal'] = 'long'     # Tuesday
    signals.loc[signals['day_of_week'] == 2, 'trade_signal'] = 'long'     # Wednesday

    # Avoid trades on Mondays (0) and Fridays (4)
    signals.loc[signals['day_of_week'] == 0, 'trade_signal'] = 'no_trade'  # Monday
    signals.loc[signals['day_of_week'] == 4, 'trade_signal'] = 'no_trade'  # Friday

    # Keeping the rest as neutral
    signals['trade_signal'].replace('neutral', 'neutral', inplace=True)
    
    return signals[['trade_signal']]
