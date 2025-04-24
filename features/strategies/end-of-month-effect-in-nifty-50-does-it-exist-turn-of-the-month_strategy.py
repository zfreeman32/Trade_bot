
import pandas as pd

# End Of Month Effect Trading Strategy for NIFTY 50
def end_of_month_effect_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['position'] = 'neutral'

    # Calculate the trading days of the month
    stock_df['Date'] = pd.to_datetime(stock_df.index)
    stock_df['TradingDay'] = stock_df['Date'].dt.day

    # Determine the last trading day and 10 last trading days
    last_trading_day = stock_df['TradingDay'].max()
    entry_days = list(range(last_trading_day - 9, last_trading_day + 1))

    for day in entry_days:
        entry_condition = (stock_df['TradingDay'] == day)
        exit_condition = (stock_df['TradingDay'] == 2)  # second trading day of the next month

        # Generate long signals on the entry days
        signals.loc[entry_condition, 'position'] = 'long'
        
        # Set to neutral on exit days
        signals.loc[exit_condition, 'position'] = 'neutral'
    
    # Clean up the signals DataFrame
    signals['position'] = signals['position'].shift(1)  # Shift positions to avoid lookahead bias
    signals.dropna(inplace=True)
    return signals
