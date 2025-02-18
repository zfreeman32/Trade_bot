import pandas as pd
import numpy as np
from ta import momentum

def rocwb_signals(stock_df, roc_length=14, average_length=9, ema_length=12, num_rmss=2, average_type='simple'):

    # Create a DataFrame to hold signals
    signals = pd.DataFrame(index=stock_df.index)

    # Compute ROC
    roc = momentum.roc(stock_df['Close'], window=roc_length)
    signals['ROC'] = roc

    # Compute average ROC
    mov_avgs = {
        'simple': pd.Series.rolling,
        'exponential': pd.Series.ewm,
        # The following two would need the library `ta`
        # 'weighted': ta.volatility.bollinger_mavg
        # 'Wilder's': ta.trend.ema_indicator,
        # 'Hull': ta.trend.hma_indicator,
    }
    signals['AvgROC'] = mov_avgs[average_type](signals['ROC'], window=average_length).mean()

    # Compute RMS of ROC
    signals['RMS'] = np.sqrt(np.mean(np.square(signals['ROC'].diff().dropna())))

    # Compute bands
    signals['LowerBand'] = signals['AvgROC'] - num_rmss * signals['RMS']
    signals['UpperBand'] = signals['AvgROC'] + num_rmss * signals['RMS']

    # Compute EMA
    signals['EMA'] = stock_df['Close'].ewm(span=ema_length, adjust=False).mean()

    # Initialize the signal column as neutral
    signals['rocwb_signal'] = 'neutral'

    # Generate Buy signals (when Close is above EMA and ROC is above LowerBand)
    signals.loc[(stock_df['Close'] > signals['EMA']) & (signals['ROC'] > signals['LowerBand']), 'rocwb_signal'] = 'buy'

    # Generate Sell signals (when Close is below EMA and ROC is below UpperBand)
    signals.loc[(stock_df['Close'] < signals['EMA']) & (signals['ROC'] < signals['UpperBand']), 'rocwb_signal'] = 'sell'

    # Remove all auxiliary columns
    signals.drop(['ROC', 'AvgROC', 'RMS', 'LowerBand', 'UpperBand', 'EMA'], axis=1, inplace=True)

    # Return signals DataFrame
    return signals

