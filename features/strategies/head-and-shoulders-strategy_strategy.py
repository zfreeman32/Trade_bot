
import pandas as pd
from ta import momentum

# Market Sentiment Analysis Strategy
def market_sentiment_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['RSI'] = momentum.RSIIndicator(stock_df['Close'], window).rsi()
    signals['Market_Sentiment'] = 'neutral'
    
    # Define long and short conditions based on RSI
    signals.loc[signals['RSI'] < 30, 'Market_Sentiment'] = 'long'  # Oversold condition
    signals.loc[signals['RSI'] > 70, 'Market_Sentiment'] = 'short'  # Overbought condition
    
    return signals[['Market_Sentiment']]
