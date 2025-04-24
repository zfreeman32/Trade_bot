
import pandas as pd

# Exhaustion Gap Trading Strategy
def exhaustion_gap_signals(stock_df, volume_multiplier=1.5, price_gap_ratio=0.02):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the price gaps
    stock_df['Price Gap'] = stock_df['Open'].shift(1) - stock_df['Close'].shift(1)
    
    # Identify gaps and their direction
    signals['Exhaustion Gap'] = stock_df['Price Gap'] > (stock_df['Close'].shift(1) * price_gap_ratio)
    
    # Calculate volume to determine if gap is accompanied by high volume
    avg_volume = stock_df['Volume'].rolling(window=20).mean()
    signals['High Volume'] = stock_df['Volume'] > (avg_volume * volume_multiplier)
    
    # Generate signals based on gap presence and volume
    signals['signal'] = 'neutral'
    signals.loc[signals['Exhaustion Gap'] & signals['High Volume'], 'signal'] = 'short'
    signals.loc[signals['Exhaustion Gap'].shift(1) & signals['High Volume'], 'signal'] = 'long'
    
    return signals[['signal']]
