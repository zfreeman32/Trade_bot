
import pandas as pd
from ta import trend

# Trend Following Strategy using a 20-day Exponential Moving Average
def trend_following_btc_signals(btc_df, short_window=20):
    signals = pd.DataFrame(index=btc_df.index)
    signals['EWM'] = trend.EMAIndicator(btc_df['Close'], window=short_window).ema_indicator()
    
    # Initialize the signal column
    signals['signal'] = 'neutral'
    
    # Create long and short signals
    signals.loc[signals['Close'] > signals['EWM'], 'signal'] = 'long'
    signals.loc[signals['Close'] < signals['EWM'], 'signal'] = 'short'
    
    return signals[['signal']]
