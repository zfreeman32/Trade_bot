
import pandas as pd
import numpy as np

# Accumulative Swing Index (ASI) Strategy
def asi_signals(stock_df, period=14):
    def calculate_swing_index(highs, lows, closes):
        swing_index = []
        for i in range(1, len(closes)):
            current_close = closes[i]
            previous_close = closes[i - 1]
            high = highs[i]
            low = lows[i]
            previous_high = highs[i - 1]
            previous_low = lows[i - 1]
            
            if current_close > previous_close:
                swing = ((high - previous_high) / (previous_high - previous_low)) * 100
            else:
                swing = -((previous_low - low) / (previous_high - previous_low)) * 100
            
            swing_index.append(swing)
        return np.insert(swing_index, 0, 0)  # prepend a 0 for the first index

    swing_index_values = calculate_swing_index(stock_df['High'], stock_df['Low'], stock_df['Close'])
    asi = pd.Series(swing_index_values).rolling(window=period).sum()

    signals = pd.DataFrame(index=stock_df.index)
    signals['ASI'] = asi
    signals['asi_signal'] = 'neutral'
    signals.loc[(signals['ASI'] > 0) & (signals['ASI'].shift(1) <= 0), 'asi_signal'] = 'long'
    signals.loc[(signals['ASI'] < 0) & (signals['ASI'].shift(1) >= 0), 'asi_signal'] = 'short'
    signals.drop(['ASI'], axis=1, inplace=True)
    return signals
