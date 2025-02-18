import pandas as pd
import numpy as np

def calculate_z_score(series):
    return (series - series.mean()) / np.std(series)

def simple_moving_average(series, window):
    return series.rolling(window).mean()

def simple_mean_reversion_signals(stock_df, length=30, fast_length_factor=1, slow_length_factor=10,
                                  z_score_entry_level=1.0, z_score_exit_level=0.5):

    close_prices = stock_df['Close']
    signals = pd.DataFrame(index=stock_df.index)
    
    # Compute SMA and Z-Score
    signals['FasterSMA'] = simple_moving_average(close_prices, length * fast_length_factor)
    signals['SlowerSMA'] = simple_moving_average(close_prices, length * slow_length_factor)
    signals['zScore'] = calculate_z_score(close_prices)

    # Initialize signal column
    signals['simple_mean_reversion_signals'] = 'neutral'

    # Buy Signal: When Z-Score < -Entry Level and Faster SMA > Slower SMA
    signals.loc[(signals['zScore'] < -z_score_entry_level) & (signals['FasterSMA'] > signals['SlowerSMA']), 'simple_mean_reversion_signals'] = 'buy'

    # Sell Signal: When Z-Score > Entry Level and Faster SMA < Slower SMA
    signals.loc[(signals['zScore'] > z_score_entry_level) & (signals['FasterSMA'] < signals['SlowerSMA']), 'simple_mean_reversion_signals'] = 'sell'

    # Drop unnecessary columns
    signals = signals[['simple_mean_reversion_signals']]

    return signals
