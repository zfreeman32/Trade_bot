
import pandas as pd

# London Breakout Strategy
def london_breakout_signals(fx_df, asian_start='00:00', asian_end='06:00', london_start='08:00', london_end='11:00'):
    signals = pd.DataFrame(index=fx_df.index)
    signals['signal'] = 'neutral'

    # Convert index to datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(fx_df.index):
        fx_df.index = pd.to_datetime(fx_df.index)

    # Define Asian session range
    asian_session = fx_df.between_time(asian_start, asian_end)
    asian_high = asian_session['High'].max()
    asian_low = asian_session['Low'].min()

    # Define conditions for the London session
    london_session = fx_df.between_time(london_start, london_end)

    for time, row in london_session.iterrows():
        if row['High'] > asian_high:
            signals.at[time, 'signal'] = 'long'
        elif row['Low'] < asian_low:
            signals.at[time, 'signal'] = 'short'

    return signals
