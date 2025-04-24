
import pandas as pd
import numpy as np

# Do Stocks Outperform Treasury Bills Strategy
def stock_vs_treasury_signals(stock_df, treasury_bill_rate=0.0033):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Returns'] = stock_df['Close'].pct_change()
    signals['Cumulative_Returns'] = (1 + signals['Returns']).cumprod()
    
    # Compare with Treasury Bill Rate
    signals['Beat_Treasury'] = signals['Cumulative_Returns'] > (1 + treasury_bill_rate)**(signals.index.month / 12)  # Approximation for monthly comparison
    
    signals['signal'] = 'neutral'
    # Generating signals based on performance
    signals.loc[signals['Beat_Treasury'] & (signals['Beat_Treasury'].shift(1) == False), 'signal'] = 'long'
    signals.loc[~signals['Beat_Treasury'] & (signals['Beat_Treasury'].shift(1) == True), 'signal'] = 'short'
    
    signals.drop(['Returns', 'Cumulative_Returns', 'Beat_Treasury'], axis=1, inplace=True)
    return signals
