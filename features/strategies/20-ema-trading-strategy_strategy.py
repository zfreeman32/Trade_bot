
import pandas as pd
from ta import trend

# 20 EMA Trading Strategy
def ema_20_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    # Calculate the 20-day EMA
    signals['EMA_20'] = trend.EMAIndicator(stock_df['Close'], window=20).ema_indicator()
    
    signals['ema_signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['EMA_20']) & (stock_df['Close'].shift(1) <= signals['EMA_20'].shift(1)), 'ema_signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['EMA_20']) & (stock_df['Close'].shift(1) >= signals['EMA_20'].shift(1)), 'ema_signal'] = 'short'
    
    return signals[['ema_signal']]
