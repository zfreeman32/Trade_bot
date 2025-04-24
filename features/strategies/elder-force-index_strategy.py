
import pandas as pd
from ta import volume as ta_volume

# Elder Force Index Strategy
def elder_force_index_signals(stock_df, period=13):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Elder Force Index
    stock_df['Force_Index'] = (stock_df['Close'].diff(period) * stock_df['Volume']).rolling(window=period).sum()
    signals['Force_Index'] = stock_df['Force_Index']
    
    signals['efi_signal'] = 'neutral'
    signals.loc[(signals['Force_Index'] > 0) & (signals['Force_Index'].shift(1) <= 0), 'efi_signal'] = 'long'
    signals.loc[(signals['Force_Index'] < 0) & (signals['Force_Index'].shift(1) >= 0), 'efi_signal'] = 'short'
    
    signals.drop(['Force_Index'], axis=1, inplace=True)
    return signals
