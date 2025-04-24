
import pandas as pd

# Reversal Day Trading Strategy
def reversal_day_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate new highs and lows
    signals['new_low'] = stock_df['Low'] < stock_df['Low'].shift(1)
    signals['new_high'] = stock_df['High'] > stock_df['High'].shift(1)
    
    # Check for bullish reversal condition
    signals['bullish_reversal'] = (signals['new_low'] & 
                                   (stock_df['Close'] > stock_df['High'].shift(1)) & 
                                   (stock_df['Close'] > stock_df['Open']))
    
    # Generate signals
    signals['reversal_signal'] = 'neutral'
    signals.loc[signals['bullish_reversal'], 'reversal_signal'] = 'long'
    
    return signals[['reversal_signal']]
