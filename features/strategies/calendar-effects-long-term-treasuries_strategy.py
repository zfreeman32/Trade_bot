
import pandas as pd

# TLT Seasonal Trading Strategy
def tlt_seasonal_signals(tlt_df):
    signals = pd.DataFrame(index=tlt_df.index)
    signals['signal'] = 'neutral'

    # Extract the day of the month
    signals['day'] = tlt_df.index.day

    # First 7 days of the month: signal to go short
    signals.loc[signals['day'] <= 7, 'signal'] = 'short'

    # Days 8 to end of month: signal to go long
    signals.loc[signals['day'] > 7, 'signal'] = 'long'

    # Clean up
    signals.drop('day', axis=1, inplace=True)
    return signals
