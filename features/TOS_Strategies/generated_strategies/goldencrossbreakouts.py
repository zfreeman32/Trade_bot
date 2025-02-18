import pandas as pd
import numpy as np
from ta import trend

# GoldenCrossBreakouts strategy
def golden_cross_signals(stock_df, fast_length=50, slow_length=200, average_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)

    if average_type == 'simple':
        fast_ma = trend.SMAIndicator(stock_df['Close'], window=fast_length).sma_indicator()
        slow_ma = trend.SMAIndicator(stock_df['Close'], window=slow_length).sma_indicator()
    elif average_type == 'exponential':
        fast_ma = trend.EMAIndicator(stock_df['Close'], window=fast_length).ema_indicator()
        slow_ma = trend.EMAIndicator(stock_df['Close'], window=slow_length).ema_indicator()
    else:
        raise ValueError("Invalid moving average type. Choose 'simple' or 'exponential'")

    signals['FastMA'] = fast_ma
    signals['SlowMA'] = slow_ma
    signals['golden_cross_signal'] = 'neutral'
    signals.loc[signals['FastMA'] > signals['SlowMA'], 'golden_cross_signal'] = 'long'
    signals.loc[signals['FastMA'] < signals['SlowMA'], 'golden_cross_signal'] = 'short'
    signals.drop(columns=['FastMA', 'SlowMA'], inplace=True)

    return signals
