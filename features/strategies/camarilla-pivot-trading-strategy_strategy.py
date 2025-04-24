
import pandas as pd

# Camarilla Pivot Trading Strategy
def camarilla_pivot_signals(stock_df):
    # Calculate the previous day's high, low, and close
    prev_high = stock_df['High'].shift(1)
    prev_low = stock_df['Low'].shift(1)
    prev_close = stock_df['Close'].shift(1)
    
    # Calculate the Camarilla pivots
    h = prev_high
    l = prev_low
    c = prev_close
    
    resistance_levels = {
        'R4': c + (h - l) * 1.1 / 2,
        'R3': c + (h - l) * 1.1 / 4,
        'R2': c + (h - l) * 1.1 / 6,
        'R1': c + (h - l) * 1.1 / 12
    }
    
    support_levels = {
        'S1': c - (h - l) * 1.1 / 12,
        'S2': c - (h - l) * 1.1 / 6,
        'S3': c - (h - l) * 1.1 / 4,
        'S4': c - (h - l) * 1.1 / 2
    }
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'

    # Generate long and short signals
    signals.loc[(stock_df['Close'] > resistance_levels['R3']), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < support_levels['S3']), 'signal'] = 'short'
    
    return signals
