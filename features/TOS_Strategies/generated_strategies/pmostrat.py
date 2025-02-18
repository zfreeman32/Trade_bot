import pandas as pd
import numpy as np

# Price Momentum Oscillator (PMO) Strategy
def pmo_signals(stock_df, length1=20, length2=10, signal_length=10):
    # Calculate the one-bar rate of change
    roc = stock_df['Close'].pct_change()
    
    # Smooth the rate of change using two exponential moving averages
    pmo_line = roc.ewm(span=length1, adjust=False).mean().ewm(span=length2, adjust=False).mean()
    
    # Create the signal line, which is an EMA of the PMO line
    pmo_signal = pmo_line.ewm(span=signal_length, adjust=False).mean()
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['pmo_line'] = pmo_line
    signals['pmo_signals'] = pmo_signal

    # Generate trading signals based on crossover of PMO line and PMO signal
    signals['buy'] = np.where((signals['pmo_line'] > signals['pmo_signals']) & (signals['pmo_line'].shift(1) <= signals['pmo_signals'].shift(1)), 1, 0)
    signals['sell'] = np.where((signals['pmo_line'] < signals['pmo_signals']) & (signals['pmo_line'].shift(1) >= signals['pmo_signals'].shift(1)), -1, 0)
    
    # 'pmo_signals' column contains buy(1)/sell(-1)/neutral(0) signals
    signals['pmo_signals'] = signals['buy'] + signals['sell']
    
    signals.drop(['pmo_line', 'buy', 'sell'], axis=1, inplace=True)

    return signals
