
import pandas as pd

# Even vs. Odd Days Trading Strategy
def even_odd_days_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['day'] = stock_df.index.day

    # Initialize signals
    signals['strategy_signal'] = 'neutral'

    # Buy on even days and hold until the close of the next odd day
    for i in range(1, len(signals)):
        if signals['day'].iloc[i] % 2 == 0:  # Even day
            signals['strategy_signal'].iloc[i] = 'long'  # Buy at close of this day
        elif signals['day'].iloc[i] % 2 != 0 and signals['strategy_signal'].iloc[i-1] == 'long':
            signals['strategy_signal'].iloc[i] = 'short'  # Sell at close of this odd day
        else:
            signals['strategy_signal'].iloc[i] = signals['strategy_signal'].iloc[i-1]  # Hold
            
    return signals.drop('day', axis=1)
