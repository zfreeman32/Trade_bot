
import pandas as pd
from ta import trend, momentum

# GBP/AUD Trend Following Strategy
def gbp_aud_trend_signals(gbp_aud_df, short_window=20, long_window=50, rsi_window=14, rsi_overbought=70, rsi_oversold=30):
    signals = pd.DataFrame(index=gbp_aud_df.index)
    
    # Calculate Moving Averages
    signals['Short_MA'] = gbp_aud_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = gbp_aud_df['Close'].rolling(window=long_window).mean()
    
    # Calculate RSI
    rsi = momentum.RSIIndicator(gbp_aud_df['Close'], rsi_window)
    signals['RSI'] = rsi.rsi()
    
    # Initialize signals
    signals['signal'] = 'neutral'
    
    # Generate long signals
    signals.loc[(signals['Short_MA'] > signals['Long_MA']) & (signals['RSI'] < rsi_oversold), 'signal'] = 'long'
    # Generate short signals
    signals.loc[(signals['Short_MA'] < signals['Long_MA']) & (signals['RSI'] > rsi_overbought), 'signal'] = 'short'

    return signals[['signal', 'RSI', 'Short_MA', 'Long_MA']]
