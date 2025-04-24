
import pandas as pd

# Macro Economy Trading Strategy
def macro_economy_signals(stock_df, interest_rate_col, inflation_col, geopolitical_risk_col):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Simple heuristic for the signals based on macro variables
    signals['signal'] = 'neutral'
    
    # Logic for generating signals
    signals.loc[(stock_df['Close'].shift(1) < stock_df['Close']) & 
                 (stock_df[interest_rate_col].shift(1) > stock_df[interest_rate_col]) & 
                 (stock_df[inflation_col] < 3), 'signal'] = 'long'
    
    signals.loc[(stock_df['Close'].shift(1) > stock_df['Close']) & 
                 (stock_df[geopolitical_risk_col] > 5), 'signal'] = 'short'
    
    return signals
