
import pandas as pd

# Membership Strategy Signals
def membership_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Example strategy using moving average for signal generation
    signals['Short_MA'] = stock_df['Close'].rolling(window=10).mean()  # Short moving average
    signals['Long_MA'] = stock_df['Close'].rolling(window=30).mean()   # Long moving average
    signals['Signal'] = 0  # Default signal is neutral (0)

    # Generate buy (1) and sell (-1) signals based on moving average crossover
    signals['Signal'][10:] = np.where(signals['Short_MA'][10:] > signals['Long_MA'][10:], 1, 0)  # Buy signal
    signals['Signal'][10:] = np.where(signals['Short_MA'][10:] < signals['Long_MA'][10:], -1, signals['Signal'][10:])  # Sell signal
    
    # Map numerical signals to 'long', 'short', 'neutral'
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['Signal'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['Signal'] == -1, 'trade_signal'] = 'short'
    
    return signals[['trade_signal']]
