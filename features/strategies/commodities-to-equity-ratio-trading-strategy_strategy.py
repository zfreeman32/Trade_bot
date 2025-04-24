
import pandas as pd
import numpy as np

# Trading Strategy based on specified membership plan 
def membership_plan_signals(stock_df, strategy_list, strategy_bonus, months=12):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    
    # Assuming strategy_list is a list of booleans indicating if the plan is active
    active_strategies = strategy_list.count(True)
    
    # Generating signals based on chosen strategies and bonuses
    if active_strategies >= 15:
        signals['membership_signal'] = 'long'
    elif active_strategies >= 10:
        signals['membership_signal'] = 'short'
    
    # Monthly strategy bonus
    if months > 0:
        signals['bonus_signal'] = 'strategies available'
    
    # Combine the membership signal with monthly strategy bonus
    signals['final_signal'] = np.where(signals['membership_signal'] == 'neutral', 
                                        'neutral', 
                                        signals['membership_signal'])
    
    return signals[['membership_signal', 'bonus_signal', 'final_signal']]
