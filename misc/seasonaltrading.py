import pandas as pd
import numpy as np

def get_positive_return_freq(stock_df, year_period):
    # Calculate the return
    stock_df['Return'] = stock_df['Close'].pct_change()
    
    # Calculate frequencies
    freq_dict = {month:[] for month in range(1,13)}
    for year in range(year_period):
        for month in range(1, 13):
            month_returns = stock_df.loc[(stock_df.index.year == year) & (stock_df.index.month == month), 'Return']
            positive_returns = month_returns[month_returns > 0]
            freq_dict[month].append(len(positive_returns)/len(month_returns))
    return {month:np.mean(freq_dict[month]) for month in range(1,13)}

def seasonal_trading_signals(stock_df, years=4, high_freq=0.75, low_freq=0.25):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Add a column for months
    signals['Month'] = signals.index.month

    # Get frequencies of the positive returns
    freq_dict = get_positive_return_freq(stock_df, years)
    
    # Add simulated orders
    signals['Order'] = 'neutral'
    signals.loc[signals['Month'].map(freq_dict) >= high_freq, 'Order'] = 'sell'
    signals.loc[signals['Month'].map(freq_dict) <= low_freq, 'Order'] = 'buy'
    
    # Clean up
    signals.drop(['Month'], axis=1, inplace=True)
    
    return signals
