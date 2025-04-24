
import pandas as pd

# Worst Week of the Year for Stocks Strategy
def worst_week_september_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Identify the OPEX Friday (third Friday of September)
    stock_df['Date'] = stock_df.index
    stock_df['Month'] = stock_df['Date'].dt.month
    stock_df['Weekday'] = stock_df['Date'].dt.weekday  # 0=Monday, ..., 4=Friday
    opex_dates = stock_df[(stock_df['Month'] == 9) & (stock_df['Weekday'] == 4) & 
                           (stock_df['Date'].dt.day >= 15) & (stock_df['Date'].dt.day <= 21)]
    
    # Close prices on OPEX Friday
    opex_close = opex_dates['Close']
    
    # Create signals DataFrame
    signals['signal'] = 'neutral'
    
    for opex_date in opex_close.index:
        # Buy signal at close of OPEX Friday
        buy_price = opex_close[opex_date]
        
        # Hold for 5 trading days (next Friday)
        end_date = opex_date + pd.DateOffset(days=5)
        
        if end_date in stock_df.index:
            signals.loc[end_date, 'signal'] = 'short'  # Short after holding for 5 days
    
    # Clean up signals to ensure signals are given only for end_date
    signals['signal'].ffill()  # Forward fill the signals to mark the days before the short

    return signals[['signal']]
