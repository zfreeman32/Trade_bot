
import pandas as pd

# End of Year Stock Market Rally Strategy
def end_of_year_rally_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['end_of_year_signal'] = 'neutral'

    # Dates for end of year trading strategy (last 10 trading days of December)
    stock_df['Date'] = stock_df.index
    stock_df['Month'] = stock_df['Date'].dt.month
    stock_df['Day'] = stock_df['Date'].dt.day

    for i in range(len(stock_df)):
        date = stock_df.iloc[i]['Date']
        if date.month == 12:
            last_trading_days = stock_df[(stock_df['Date'] <= date) & (stock_df['Month'] == 12)]
            if len(last_trading_days) >= 6 and len(last_trading_days) <= 10:  # Between 6 and 10 trading days
                signals.at[date, 'end_of_year_signal'] = 'long'
                
    return signals
