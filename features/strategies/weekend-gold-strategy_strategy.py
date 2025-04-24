
import pandas as pd
import numpy as np

# Weekend Trading Strategy
def weekend_trading_signals(price_df, stop_loss_pct=0.02, target_profit_pct=0.05):
    signals = pd.DataFrame(index=price_df.index)
    
    # Calculate returns
    price_df['Returns'] = price_df['Close'].pct_change()
    
    # Identify gaps - compare Friday's close (before the weekend) with Saturday's open
    price_df['Friday_Close'] = price_df['Close'].shift(1)
    price_df['Gap'] = price_df['Open'] - price_df['Friday_Close']
    
    # Generating signals based on gap
    signals['signal'] = 'neutral'
    
    # Condition to go long if there's a positive gap, and short if there's a negative gap
    signals.loc[(price_df['Gap'] > 0), 'signal'] = 'long'
    signals.loc[(price_df['Gap'] < 0), 'signal'] = 'short'
    
    # Calculate stop loss and target levels
    signals['Stop_Loss'] = np.where(signals['signal'] == 'long',
                                     price_df['Open'] * (1 - stop_loss_pct),
                                     price_df['Open'] * (1 + stop_loss_pct))
    
    signals['Target_Profit'] = np.where(signals['signal'] == 'long',
                                         price_df['Open'] * (1 + target_profit_pct),
                                         price_df['Open'] * (1 - target_profit_pct))
    
    return signals[['signal', 'Stop_Loss', 'Target_Profit']]
