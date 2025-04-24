
import pandas as pd

# Lumber/Gold Ratio Trading Strategy
def lumber_gold_ratio_signals(lumber_df, gold_df, threshold=0.5):
    signals = pd.DataFrame(index=lumber_df.index)
    
    # Calculate Lumber/Gold ratio
    signals['Lumber'] = lumber_df['Close']
    signals['Gold'] = gold_df['Close']
    signals['Lumber_Gold_Ratio'] = signals['Lumber'] / signals['Gold']
    
    # Generate trading signals based on the Lumber/Gold ratio
    signals['signal'] = 'neutral'
    signals.loc[signals['Lumber_Gold_Ratio'] > threshold, 'signal'] = 'long'
    signals.loc[signals['Lumber_Gold_Ratio'] < threshold, 'signal'] = 'short'
    
    return signals[['Lumber_Gold_Ratio', 'signal']]
