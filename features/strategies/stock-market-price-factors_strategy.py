
import pandas as pd
import numpy as np

# Market Sentiment and Economic Indicators Strategy
def market_sentiment_signals(stock_df, news_sentiment_df, economic_indicators_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['price_change'] = stock_df['Close'].pct_change()
    
    # Simple sentiment score calculation
    sentiment_score = news_sentiment_df['Sentiment'].rolling(window=5).mean()
    economic_indicator_score = economic_indicators_df['Indicator'].rolling(window=5).mean()
    
    # Create signal logic based on sentiment and economic indicators
    signals['signal'] = 'neutral'
    signals.loc[(sentiment_score > 0) & (economic_indicator_score > 0), 'signal'] = 'long'
    signals.loc[(sentiment_score < 0) & (economic_indicator_score < 0), 'signal'] = 'short'
    
    # Clean dataframe by dropping 'price_change' column
    return signals[['signal']]
