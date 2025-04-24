
import pandas as pd

# Internal Bar Strength (IBS) Strategy
def ibs_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['IBS'] = (stock_df['Close'] - stock_df['Low']) / (stock_df['High'] - stock_df['Low'])
    signals['ibs_signal'] = 'neutral'

    # Buy signal (long) when IBS is less than 0.2 (indicating close near the low)
    signals.loc[signals['IBS'] < 0.2, 'ibs_signal'] = 'long'
    # Sell signal (short) when IBS is greater than 0.8 (indicating close near the high)
    signals.loc[signals['IBS'] > 0.8, 'ibs_signal'] = 'short'
    
    signals.drop(['IBS'], axis=1, inplace=True)
    return signals
