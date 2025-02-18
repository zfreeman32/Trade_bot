import pandas as pd
from ta import trend

def halloween_strategy(data: pd.DataFrame, sma_length: int = 30):
    signals = pd.DataFrame(index=data.index)
    
    
    # Create SMA Indicator
    sma = trend.SMAIndicator(data["Close"], sma_length)
    signals['SMA'] = sma.sma_indicator()
    
    # Create a signal column and initialize it to Hold.
    signals['signal'] = 'Hold'
    
    # Generate Long Entry signal
    signals.loc[(signals.index.month == 10) & (signals.index.day == 1) & (data['Close'] > signals['SMA']), 'signal'] = 'Buy'
    
    # Generate Long Exit Signal
    signals.loc[(signals.index.month == 5) & (signals.index.day == 1), 'signal'] = 'Sell'
    signals.drop(['SMA'], axis=1, inplace=True)
    return signals
