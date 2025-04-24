
import pandas as pd

# Double Top Chart Pattern Trading Strategy
def double_top_signals(stock_df, peak_distance=5, confirmation_distance=3):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'

    # Identify peaks and troughs
    stock_df['peak'] = stock_df['Close'].rolling(window=peak_distance).max()
    stock_df['trough'] = stock_df['Close'].rolling(window=peak_distance).min()

    for i in range(peak_distance, len(stock_df) - peak_distance):
        # Check for double top pattern
        if (stock_df['Close'][i] == stock_df['peak'][i]) and (stock_df['Close'][i - peak_distance] == stock_df['peak'][i - peak_distance]):
            # Check if neckline is broken
            neckline = stock_df['Close'][i - (peak_distance // 2):i].min()
            if stock_df['Close'][i] < neckline:
                signals['signal'].iloc[i] = 'short'

    # Look for confirmation of trend reversal
    for i in range(len(signals)):
        if signals['signal'].iloc[i] == 'short':
            for j in range(1, confirmation_distance + 1):
                if i + j < len(signals) and stock_df['Close'].iloc[i + j] < neckline:
                    signals['signal'].iloc[i + j] = 'short'
                else:
                    break

    return signals
