import numpy as np
import pandas as pd

def reverse_ema(price_series, period):
    # Z-transform is a signal processing technique to normalize data. 
    # It's similar to using percentage change in this case.
    z_price = price_series.pct_change()
    weights = np.array([a for a in range(1, period+1)])
    return z_price.rolling(period).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

def reverse_ema_strat(df, trend_length=39, cycle_length=6):
    signals = pd.DataFrame(index=df.index)
    signals['trend_ema'] = reverse_ema(df['Close'], period=trend_length)
    signals['cycle_ema'] = reverse_ema(df['Close'], period=cycle_length)
    
    signals['buy_to_open'] = np.where((signals['cycle_ema'] > 0) & (signals['trend_ema'] > 0), 1, 0)
    signals['sell_to_close'] = np.where((signals['cycle_ema'] < 0) | (signals['trend_ema'] < 0), -1, 0)
    signals['sell_to_open'] = np.where((signals['cycle_ema'] < 0) & (signals['trend_ema'] < 0), -1, 0)
    signals['buy_to_close'] = np.where((signals['cycle_ema'] > 0) | (signals['trend_ema'] > 0), 1, 0)

    # If there's signal for both actions 'to open' and 'to close',
    # priority is given to 'close' actions
    signals['reverse_ema_strat_signals'] = np.where((signals['buy_to_open']+signals['sell_to_close'])==0,
                                 signals['sell_to_open'], signals['buy_to_close'])

    return signals['reverse_ema_strat_signals']