import pandas as pd
import pytz

# First Hour Breakout Strategy
def FirstHourBreakout(data):
    # Ensure data is indexed by datetime
    data.index = pd.to_datetime(data.index)
    if data.index.tz is None:
        data.index = data.index.tz_localize(pytz.utc)
    data.index = data.index.tz_convert(pytz.timezone('US/Eastern'))
    
    signals = pd.DataFrame(index=data.index)
    signals['FirstHourBreakout_signals'] = 0
    
    # Define times
    market_open = pd.Timestamp('09:30', tz='US/Eastern').time()
    first_hour_end = pd.Timestamp('10:30', tz='US/Eastern').time()
    market_close = pd.Timestamp('16:15', tz='US/Eastern').time()
    
    # Iterate over each day
    for day in data.index.normalize().unique():
        day_data = data.loc[data.index.normalize() == day]
        
        if day_data.empty:
            continue
        
        # Calculate high and low of first trading hour
        first_hour_data = day_data.between_time(market_open, first_hour_end)
        if first_hour_data.empty:
            continue
        
        first_hour_high = first_hour_data['High'].max()
        first_hour_low = first_hour_data['Low'].min()
        
        # Issue buy signal
        signals.loc[(day_data.index > first_hour_data.index[-1]) & (day_data['High'] > first_hour_high), 'FirstHourBreakout_signals'] = 1
        
        # Issue sell signal
        signals.loc[(day_data.index > first_hour_data.index[-1]) & (day_data['Low'] < first_hour_low), 'FirstHourBreakout_signals'] = -1
        
        # Close positions at end of day
        signals.loc[data.index.normalize() == day, 'FirstHourBreakout_signals'] = 0
        signals.drop(['Date'], axis=1, inplace=True)
    return signals
