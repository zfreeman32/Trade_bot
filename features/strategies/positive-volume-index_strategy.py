
import pandas as pd

# Positive Volume Index (PVI) Strategy
def pvi_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the PVI
    pvi = pd.Series(index=stock_df.index, data=0.0)
    for i in range(1, len(stock_df)):
        if stock_df['Volume'].iloc[i] > stock_df['Volume'].iloc[i-1]:  # Positive volume
            if stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i-1]:  # Price up
                pvi.iloc[i] = pvi.iloc[i-1] + stock_df['Volume'].iloc[i]
            elif stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i-1]:  # Price down
                pvi.iloc[i] = pvi.iloc[i-1]
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        else:  # Volume down or unchanged
            pvi.iloc[i] = pvi.iloc[i-1]

    signals['PVI'] = pvi
    signals['pvi_signal'] = 'neutral'
    
    # Generate trading signals based on PVI trends
    signals.loc[(signals['PVI'] > signals['PVI'].shift(1)), 'pvi_signal'] = 'long'  # Bullish signal
    signals.loc[(signals['PVI'] < signals['PVI'].shift(1)), 'pvi_signal'] = 'short'  # Bearish signal
    
    return signals
