
import pandas as pd
from ta import trend

# Percentage Price Oscillator (PPO) Strategy
def ppo_signals(stock_df, fast_length=12, slow_length=26, signal_length=9):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate PPO
    ppo_indicator = trend.PPOIndicator(stock_df['Close'], window_slow=slow_length, window_fast=fast_length, window_sign=signal_length)
    signals['PPO'] = ppo_indicator.ppo()
    signals['Signal_Line'] = ppo_indicator.ppo_signal()

    # Generate signals based on PPO crossover with Signal Line
    signals['ppo_signal'] = 'neutral'
    signals.loc[(signals['PPO'] > signals['Signal_Line']) & (signals['PPO'].shift(1) <= signals['Signal_Line'].shift(1)), 'ppo_signal'] = 'long'
    signals.loc[(signals['PPO'] < signals['Signal_Line']) & (signals['PPO'].shift(1) >= signals['Signal_Line'].shift(1)), 'ppo_signal'] = 'short'

    signals.drop(['PPO', 'Signal_Line'], axis=1, inplace=True)
    return signals
