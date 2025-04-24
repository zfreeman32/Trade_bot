
import pandas as pd

# Turnaround Tuesday Trading Strategy
def turnaround_tuesday_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the condition for Monday's close
    stock_df['Prev_Close'] = stock_df['Close'].shift(1)
    
    # Define if today is Monday
    stock_df['Is_Monday'] = stock_df.index.weekday == 0  # Monday is 0
    
    # Condition: Monday close must be at least 1% lower than the previous Friday's close
    stock_df['Condition'] = stock_df['Prev_Close'] < stock_df['Prev_Close'].shift(2) * 0.99
    
    # Generate signals
    signals['turnaround_signal'] = 'neutral'
    signals.loc[stock_df['Is_Monday'] & stock_df['Condition'], 'turnaround_signal'] = 'long'
    
    return signals
