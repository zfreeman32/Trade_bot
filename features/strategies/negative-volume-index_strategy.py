
import pandas as pd

# Negative Volume Index (NVI) Strategy
def nvi_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    # Calculate daily volume change and NVI
    stock_df['Volume Change'] = stock_df['Volume'].diff()
    stock_df['Price Change'] = stock_df['Close'].diff()
    
    signals['NVI'] = 0  # Initialize NVI
    # Calculate NVI based on volume changes
    for i in range(1, len(stock_df)):
        if stock_df['Volume Change'].iloc[i] < 0:  # Negative volume day
            signals['NVI'].iloc[i] = signals['NVI'].iloc[i - 1]
        else:
            if stock_df['Volume'].iloc[i] > stock_df['Volume'].iloc[i - 1]:
                signals['NVI'].iloc[i] = signals['NVI'].iloc[i - 1] + (signals['NVI'].iloc[i - 1] + stock_df['Price Change'].iloc[i]) / signals['NVI'].iloc[i - 1]
            else:
                signals['NVI'].iloc[i] = signals['NVI'].iloc[i - 1]
    
    # Generate trading signals based on NVI
    signals['nvi_signal'] = 'neutral'
    signals.loc[(signals['NVI'] > signals['NVI'].shift(1)), 'nvi_signal'] = 'long'
    signals.loc[(signals['NVI'] < signals['NVI'].shift(1)), 'nvi_signal'] = 'short'
    
    return signals[['nvi_signal']]
