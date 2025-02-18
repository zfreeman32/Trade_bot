import pandas as pd
import numpy as np
from ta import volume, trend, momentum

def TechnicalStockRatingStrat(stock_df, market_df, vfi_length=14, sma_length=14, trend_qual_length=10, exit_length=63, max_stiffness=2, score_crit=5.0, weight_mf=1.0, weight_ta=1.0, weight_ut=1.0, weight_tq=1.0, weight_md=2.0):
    signals = pd.DataFrame(index=stock_df.index)
    vfi = volume.VolumeFlowIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], vfi_length)
    sma = trend.SMAIndicator(stock_df['Close'], sma_length)
    ema_market = trend.EMAIndicator(market_df['Close'], sma_length)
    
    signals['score'] = 0
    signals.loc[vfi.volume_flow_indicator() > 0, 'score'] += weight_mf     # Money flow
    signals.loc[stock_df['Close'] > sma.sma_indicator(), 'score'] += weight_ta     # Trading above average
    signals.loc[sma.sma_indicator().diff().rolling(window=4).sum() > 0, 'score'] += weight_ut     # In uptrend
    signals.loc[abs((stock_df['Close'] - sma.sma_indicator()).diff()) <= max_stiffness, 'score'] += weight_tq     # Trend quality
    signals.loc[ema_market.ema_indicator().diff().rolling(window=2).sum() > 0, 'score'] += weight_md     # Overall market direction

    signals['order'] = 'neutral'
    signals.loc[signals['score'] >= score_crit, 'order'] = 'buy'
    for i in range(exit_length, len(signals)):
        if signals.loc[signals.index[i - exit_length], 'order'] == 'buy':
            signals.loc[signals.index[i], 'order'] = 'sell'
            
    signals.drop(['score'], axis=1, inplace=True)
    return signals
#%% Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is in datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = TechnicalStockRatingStrat(stock_df)

# Display output
print(signals_df.head())
