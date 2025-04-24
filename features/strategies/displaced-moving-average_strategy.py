
import pandas as pd
from ta import trend

# Displaced Moving Average (DMA) Strategy
def dma_signals(stock_df, window=20, displacement=5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Displaced Moving Average
    dma = trend.SMAIndicator(stock_df['Close'], window=window).sma_indicator().shift(-displacement)
    signals['DMA'] = dma
    
    # Generate signals
    signals['dma_signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['DMA']) & (stock_df['Close'].shift(1) <= signals['DMA'].shift(1)), 'dma_signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['DMA']) & (stock_df['Close'].shift(1) >= signals['DMA'].shift(1)), 'dma_signal'] = 'short'
    
    # Drop the DMA column for clarity
    signals.drop(['DMA'], axis=1, inplace=True)
    
    return signals
