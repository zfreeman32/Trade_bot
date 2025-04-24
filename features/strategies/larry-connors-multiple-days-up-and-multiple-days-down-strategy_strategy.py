
import pandas as pd

# Multiple Days Up and Multiple Days Down Strategy
def connors_mdu_mdd_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the number of down days in the last 5 days
    signals['down_days'] = (stock_df['Close'].shift(1) > stock_df['Close']).rolling(window=5).sum()
    
    # Generate trading signals based on the strategy
    signals['mdu_mdd_signal'] = 'neutral'
    signals.loc[signals['down_days'] >= 4, 'mdu_mdd_signal'] = 'long'
    
    # Clean up DataFrame to retain necessary info
    signals.drop(['down_days'], axis=1, inplace=True)
    
    return signals
