import pandas as pd
import numpy as np

def three_bar_inside_bar_se(df):
    # Create a signal column initialized to 'neutral'
    signals = pd.DataFrame(index=df.index)
    signals['three_bar_inside_bar_se_signal'] = 'neutral'
    
    for i in range(3, len(df)):
        # The first bar's Close price is higher than that of the second bar 
        condition1 = df['Close'].iloc[i-3] > df['Close'].iloc[i-2]
        
        # The third bar's High price is less than that of the previous bar
        condition2 = df['High'].iloc[i-1] < df['High'].iloc[i-2]
        
        # The third bar's Low price is greater than that of the previous bar
        condition3 = df['Low'].iloc[i-1] > df['Low'].iloc[i-2]
        
        # The fourth bar's Close price is lower than that of the previous bar
        condition4 = df['Close'].iloc[i] < df['Close'].iloc[i-1]
        
        # If all conditions are met, generate a 'short' signal
        if condition1 and condition2 and condition3 and condition4:
            signals.loc[df.index[i], 'three_bar_inside_bar_se_signal'] = 'short'
    
    return signals


