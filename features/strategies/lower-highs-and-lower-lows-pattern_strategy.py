
import pandas as pd

# Lower Highs And Lower Lows Pattern Trading Strategy
def lower_highs_lows_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['lower_high'] = (stock_df['High'] < stock_df['High'].shift(1))
    signals['lower_low'] = (stock_df['Low'] < stock_df['Low'].shift(1))
    
    signals['pattern'] = signals['lower_high'] & signals['lower_low']
    signals['signal'] = 'neutral'
    signals.loc[signals['pattern'], 'signal'] = 'short'
    
    # Optionally, you can include exit strategy logic if needed
    # Example: You can track the holding period or set conditions for exiting 'short' positions.
    
    signals.drop(['lower_high', 'lower_low', 'pattern'], axis=1, inplace=True)
    return signals
