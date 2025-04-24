
import pandas as pd
from ta import momentum

# Momentum Reversal Strategy
def momentum_reversal_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    rsi = momentum.RSIIndicator(stock_df['Close'], window=window).rsi()
    signals['RSI'] = rsi
    signals['momentum_signal'] = 'neutral'
    
    # Long signal: when RSI crosses above 30 (oversold)
    signals.loc[(signals['RSI'] > 30) & (signals['RSI'].shift(1) <= 30), 'momentum_signal'] = 'long'
    
    # Short signal: when RSI crosses below 70 (overbought)
    signals.loc[(signals['RSI'] < 70) & (signals['RSI'].shift(1) >= 70), 'momentum_signal'] = 'short'
    
    signals.drop(['RSI'], axis=1, inplace=True)
    return signals
