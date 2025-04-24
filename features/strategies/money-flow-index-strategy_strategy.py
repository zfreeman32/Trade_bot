
import pandas as pd
from ta.volume import MFIIndicator

# Money Flow Index (MFI) Strategy
def mfi_signals(stock_df, window=14, overbought=80, oversold=20):
    signals = pd.DataFrame(index=stock_df.index)
    mfi = MFIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)
    signals['MFI'] = mfi.mfi()
    signals['mfi_signal'] = 'neutral'
    
    # Generate buy signals
    signals.loc[(signals['MFI'] < oversold), 'mfi_signal'] = 'long'
    
    # Generate sell signals
    signals.loc[(signals['MFI'] > overbought), 'mfi_signal'] = 'short'
    
    return signals[['mfi_signal']]
