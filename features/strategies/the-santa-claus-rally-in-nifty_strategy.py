
import pandas as pd

# Santa Claus Rally Strategy
def santa_claus_rally_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Ensure the DataFrame has a datetime index to work with
    if not pd.api.types.is_datetime64_any_dtype(stock_df.index):
        raise ValueError("Index must be a datetime type")
    
    # Creating a 'signal' column for the strategy
    signals['signal'] = 'neutral'

    # Determine the date for the close of the second Friday in December
    last_friday_december = stock_df.index[(stock_df.index.month == 12) & (stock_df.index.weekday == 4)]
    close_second_friday_december = last_friday_december[last_friday_december.day >= 8].min()  # Second Friday
    
    # Determine the close of the first trading day of the new year
    first_trading_day_january = stock_df.index[(stock_df.index.month == 1) & (stock_df.index.day <= 7) & (stock_df.index.weekday < 5)].min()

    # Generating signals
    if close_second_friday_december in stock_df.index:
        signals.loc[signals.index >= close_second_friday_december, 'signal'] = 'long'
    
    if first_trading_day_january in stock_df.index:
        signals.loc[signals.index >= first_trading_day_january, 'signal'] = 'neutral'

    return signals
