
import pandas as pd
import numpy as np
import ephem  # You might need to install this library to calculate moon phases

# Full Moon/Moon Phases Trading Strategy
def moon_phase_signals(stock_df):
    # Ensure that the index is in datetime format
    stock_df.index = pd.to_datetime(stock_df.index)

    # Create a new DataFrame for signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['moon_phase'] = None
    signals['moon_signal'] = 'neutral'

    # Determine the full moon dates for the trading period
    full_moon_dates = []
    new_moon_dates = []

    for date in stock_df.index:
        moon = ephem.Moon(date)
        phase = (moon.phase / 100) * 29.53  # Moon phases range from 0 to 29.53 days
        if phase < 1:  # New Moon
            new_moon_dates.append(date)
        elif phase > 14:  # Full Moon
            full_moon_dates.append(date)

    # Generate signals based on moon phases
    for date in stock_df.index:
        if date in full_moon_dates:
            signals.at[date, 'moon_phase'] = 'full_moon'
            signals.at[date, 'moon_signal'] = 'sell'  # Sell on full moon
        elif date in new_moon_dates:
            signals.at[date, 'moon_phase'] = 'new_moon'
            signals.at[date, 'moon_signal'] = 'buy'  # Buy on new moon

    return signals
