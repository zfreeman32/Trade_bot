
import pandas as pd

# Monthly Trading Strategy Membership Signals
def monthly_trading_strategy_signals(stock_df, last_signal=None):
    signals = pd.DataFrame(index=stock_df.index)
    signals['monthly_return'] = stock_df['Close'].pct_change(periods=30)  # Monthly returns

    # Create a neutral position initially
    signals['strategy_signal'] = 'neutral'
    
    # Generate signals based on monthly return
    signals.loc[(signals['monthly_return'] > 0), 'strategy_signal'] = 'long'
    signals.loc[(signals['monthly_return'] < 0), 'strategy_signal'] = 'short'
    
    # Implement the recurring nature of signals
    if last_signal is not None:
        # Carry forward the last signal if there is one to avoid abrupt changes
        signals['strategy_signal'] = signals['strategy_signal'].where(signals['strategy_signal'] == 'neutral', last_signal)

    return signals
