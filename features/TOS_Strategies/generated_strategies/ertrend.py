import pandas as pd
import numpy as np
from ta.trend import SMAIndicator

# Compute Efficiency Ratio
def compute_ER(data, window = 14):
    change = data.diff()
    volatility = change.rolling(window).sum()
    ER = change.abs().rolling(window).sum()
    return volatility / ER

# ER Trend Signal Strategy
def ERTrend_signals(stock_df, ER_window=14, ER_avg_length=14, lag=7, avg_length=14, trend_level=0.5, max_level=1.0, crit_level=0.2, mult=1.5, average_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Use selected moving average type
    if average_type == 'simple':
        MA = SMAIndicator(stock_df['Close'], avg_length).sma_indicator()
    else:
        MA = stock_df['Close'].ewm(span=avg_length).mean()
    
    ER = compute_ER(stock_df['Close'], ER_window)
    lowest_ER = ER.rolling(lag).min()
    highest_ER = ER.rolling(lag).max()

    buy_flag = (ER > crit_level) & (ER > lowest_ER * mult) & (stock_df['Close'] > MA)
    strong_trend = (ER > trend_level) & (ER < max_level)

    # Buy signals
    signals.loc[buy_flag & strong_trend, 'ERTrend_signals'] = 'buy-to-open'
    signals.loc[(stock_df['Close'].shift(-1) < MA) & (stock_df['Close'].shift(-2) > MA), 'ERTrend_signals'] = 'sell-to-close'
    
    # Sell signals
    signals.loc[~buy_flag & strong_trend, 'ERTrend_signals'] = 'sell-to-open'
    signals.loc[(stock_df['Close'].shift(-1) > MA) & (stock_df['Close'].shift(-2) < MA), 'ERTrend_signals'] = 'buy-to-close'

    # Ensure 'hold' values remain properly assigned without inplace modification
    signals['ERTrend_signals'] = signals['ERTrend_signals'].fillna('hold')
        
    return signals

