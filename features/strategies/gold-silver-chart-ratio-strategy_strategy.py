
import pandas as pd
from ta import momentum

# Gold Silver Ratio Trading Strategy
def gold_silver_ratio_signals(gold_df, silver_df, rsi_window=5, rsi_overbought=75, rsi_oversold=50):
    # Calculate RSI for Gold and Silver
    gold_rsi = momentum.RSIIndicator(gold_df['Close'], window=rsi_window).rsi()
    silver_rsi = momentum.RSIIndicator(silver_df['Close'], window=rsi_window).rsi()
    
    # Create a signals DataFrame
    signals = pd.DataFrame(index=gold_df.index)
    signals['gold_rsi'] = gold_rsi
    signals['silver_rsi'] = silver_rsi
    signals['signal'] = 'neutral'
    
    # Buy signal: Gold when RSI > 75, Sell short silver when RSI > 75
    signals.loc[(signals['gold_rsi'] > rsi_overbought), 'signal'] = 'long_gold_short_silver'
    
    # Exit signal: when RSI falls below 50
    signals.loc[(gold_rsi < rsi_oversold) | (silver_rsi < rsi_oversold), 'signal'] = 'exit'
    
    return signals[['gold_rsi', 'silver_rsi', 'signal']]
