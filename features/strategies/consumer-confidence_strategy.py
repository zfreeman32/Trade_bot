
import pandas as pd

# Consumer Confidence Index Trading Strategy
def consumer_confidence_signals(stock_df, consumer_confidence_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['consumer_confidence'] = consumer_confidence_df['Confidence Index']
    
    # Generate signals based on consumer confidence levels
    signals['cc_signal'] = 'neutral'
    signals.loc[signals['consumer_confidence'] > signals['consumer_confidence'].rolling(window=3).mean(), 'cc_signal'] = 'long'
    signals.loc[signals['consumer_confidence'] < signals['consumer_confidence'].rolling(window=3).mean(), 'cc_signal'] = 'short'

    return signals
