
import pandas as pd
from ta import momentum

# GBP/CHF Forex Strategy
def gbpchf_signals(forex_df, window=14):
    signals = pd.DataFrame(index=forex_df.index)
    
    # Calculate RSI (Relative Strength Index)
    u = forex_df['Close'].diff().where(lambda x: x > 0, 0)
    d = -forex_df['Close'].diff().where(lambda x: x < 0, 0)
    
    avg_u = u.rolling(window=window).mean()
    avg_d = d.rolling(window=window).mean()

    rs = avg_u / avg_d
    rsi = 100 - (100 / (1 + rs))
    
    signals['RSI'] = rsi
    signals['gbpchf_signal'] = 'neutral'
    
    # Define thresholds for long and short signals
    overbought = 70
    oversold = 30
    
    signals.loc[(signals['RSI'] < oversold), 'gbpchf_signal'] = 'long'
    signals.loc[(signals['RSI'] > overbought), 'gbpchf_signal'] = 'short'
    
    # Return the DataFrame with signals
    return signals[['gbpchf_signal']]
