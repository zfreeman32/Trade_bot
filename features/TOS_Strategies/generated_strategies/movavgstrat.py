import pandas as pd

def moving_average_strategy(df, window=15, average_type='simple', mode='trend Following'):

    # compute moving average
    if average_type == 'simple':
        df['moving_avg'] = df['Close'].rolling(window=window).mean()
    elif average_type == 'exponential':
        df['moving_avg'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # create signals based on the mode
    df['moving_average_strategy_siganls'] = None
    if mode == 'trend Following':
        df.loc[df['Close'] > df['moving_avg'], 'moving_average_strategy_siganls'] = 'Buy' 
        df.loc[df['Close'] < df['moving_avg'], 'moving_average_strategy_siganls'] = 'Sell' 
    elif mode =='reversal':
        df.loc[df['Close'] > df['moving_avg'], 'moving_average_strategy_siganls'] = 'Sell'
        df.loc[df['Close'] < df['moving_avg'], 'moving_average_strategy_siganls'] = 'Buy' 
    
    return df['moving_average_strategy_siganls']
