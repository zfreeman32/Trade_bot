
import pandas as pd

# 3 Days Down Overnight Trading Strategy
def three_days_down_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']

    # Identify the condition for three consecutive down days
    signals['Down_Day'] = signals['Close'].diff() < 0  # True if the current day is a down day
    signals['Three_Days_Down'] = signals['Down_Day'].rolling(window=3).sum() == 3  # Sum of down days over the last 3 days

    # Generate signals based on the conditions
    signals['signal'] = 'neutral'  # Default state
    signals.loc[signals['Three_Days_Down'], 'signal'] = 'long'  # Enter long position at close of third down day

    # Assuming the next day is the exit point, we set the exit signal for the next day
    signals['exit_signal'] = signals['signal'].shift(-1)
    
    return signals[['signal', 'exit_signal']]
