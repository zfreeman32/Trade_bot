import pandas as pd
from ta import volatility,trend

def atr_trailing_stop_le_signals(stock_df, atr_period=14, atr_factor=3, average_type='simple'):
    close = stock_df['Close']
    high = stock_df['High']
    low = stock_df['Low']
    
    atr = volatility.AverageTrueRange(high, low, close, window=atr_period).average_true_range()
    atr_trailing_stop = close - atr_factor * atr

    if average_type == 'exponential':
        atr_trailing_stop = trend.EMAIndicator(atr_trailing_stop, atr_period).ema_indicator()
    
    signals = pd.DataFrame(index=stock_df.index)
    signals['ATR Trailing Stop'] = atr_trailing_stop
    signals['atr_trailing_stop_le_signal'] = 0
    signals.loc[(close > atr_trailing_stop), 'atr_trailing_stop_le_signal'] = 1
    signals.loc[(close <= atr_trailing_stop), 'atr_trailing_stop_le_signal'] = -1
    signals.drop(['ATR Trailing Stop'], axis=1, inplace=True)

    return signals
