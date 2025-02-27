import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# PPO (Percentage Price Oscillator)
def ppo_signals(stock_data, fast_window=12, slow_window=26, signal_window=9):
    """
    Computes PPO crossover signals.

    Returns:
    A DataFrame with 'PPO_signal'.
    """
    signals = pd.DataFrame(index=stock_data.index)

    # Calculate PPO and PPO signal
    ppo = momentum.PercentagePriceOscillator(stock_data['Close'], fast_window, slow_window, signal_window)
    signals['PPO'] = ppo.ppo()
    signals['PPO_Signal'] = ppo.ppo_signal()

    # Generate buy/sell signals on PPO crossover
    signals['PPO_buy_signal'] = 0
    signals['PPO_sell_signal'] = 0
    signals.loc[(signals['PPO'] > signals['PPO_Signal']) & 
                (signals['PPO'].shift(1) <= signals['PPO_Signal'].shift(1)), 'PPO_buy_signal'] = 1
    
    signals.loc[(signals['PPO'] < signals['PPO_Signal']) & 
                (signals['PPO'].shift(1) >= signals['PPO_Signal'].shift(1)), 'PPO_sell_signal'] = 1

    return signals
