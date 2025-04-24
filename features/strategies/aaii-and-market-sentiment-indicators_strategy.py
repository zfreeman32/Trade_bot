
import pandas as pd

# AAII Market Sentiment Strategy
def aaii_sentiment_signals(aaii_sentiment_df, bullish_threshold=60, bearish_threshold=30):
    """
    Generates trading signals based on the AAII Sentiment Index.
    
    Parameters:
    aaii_sentiment_df (DataFrame): DataFrame containing the AAII sentiment data.
    bullish_threshold (int): Threshold for bullish sentiment.
    bearish_threshold (int): Threshold for bearish sentiment.
    
    Returns:
    DataFrame: A DataFrame containing the sentiment signals.
    """
    signals = pd.DataFrame(index=aaii_sentiment_df.index)
    signals['AAII_Sentiment'] = aaii_sentiment_df['Bullish'] - aaii_sentiment_df['Bearish']
    signals['sentiment_signal'] = 'neutral'
    
    # Assigning signals based on sentiment thresholds
    signals.loc[signals['AAII_Sentiment > bullish_threshold', 'sentiment_signal'] = 'short'
    signals.loc[signals['AAII_Sentiment < -bearish_threshold', 'sentiment_signal'] = 'long'
    
    return signals[['sentiment_signal']]
