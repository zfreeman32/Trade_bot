
import pandas as pd
from ta import trend

# 9 EMA Trading Strategy
def ema9_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    ema9 = trend.EMAIndicator(stock_df['Close'], window=9).ema_indicator()
    
    signals['EMA9'] = ema9
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['EMA9']) & (stock_df['Close'].shift(1) <= signals['EMA9'].shift(1)), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['EMA9']) & (stock_df['Close'].shift(1) >= signals['EMA9'].shift(1)), 'signal'] = 'short'
    
    return signals[['signal']]
