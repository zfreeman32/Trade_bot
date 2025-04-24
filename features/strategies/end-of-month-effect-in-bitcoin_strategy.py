
import pandas as pd

# End of Month Effect Bitcoin Trading Strategy
def end_of_month_effect_signals(bitcoin_df, entry_days_before_month_end=10, exit_days_after_month_start=9):
    signals = pd.DataFrame(index=bitcoin_df.index)
    signals['signal'] = 'neutral'

    # Ensure the DataFrame is sorted by date
    bitcoin_df = bitcoin_df.sort_index()

    # Iterate over the DataFrame to implement the entry and exit logic
    for i in range(len(bitcoin_df)):
        if i >= entry_days_before_month_end:
            # Check for entry signal on the last trading day of the month
            if bitcoin_df.index[i].day >= (bitcoin_df.index[i] + pd.DateOffset(days=1)).days:
                signals.iloc[i - entry_days_before_month_end, signals.columns.get_loc('signal')] = 'long'

        # Check for exit signal on the given days of the new month
        if signals['signal'].iloc[i - exit_days_after_month_start] == 'long':
            signals.iloc[i, signals.columns.get_loc('signal')] = 'neutral'

    return signals
