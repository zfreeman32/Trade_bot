
import pandas as pd
import numpy as np
from ta import momentum

# Market Sentiment Analysis Strategy
def market_sentiment_strategy(stock_df, sentiment_score_column='SentimentScore', threshold=0.5):
    """
    Generate trading signals based on market sentiment analysis.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock prices and sentiment scores.
    sentiment_score_column (str): Column name for sentiment scores in the DataFrame.
    threshold (float): Threshold for sentiment score to generate trade signals.

    Returns:
    pd.DataFrame: DataFrame with signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['sentiment'] = stock_df[sentiment_score_column]

    signals['trade_signal'] = 'neutral'
    signals.loc[signals['sentiment'] > threshold, 'trade_signal'] = 'long'
    signals.loc[signals['sentiment'] < -threshold, 'trade_signal'] = 'short'
    
    return signals[['trade_signal']]
