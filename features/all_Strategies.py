#%%
import pandas as pd
from ta import momentum, trend, volatility, volume

#%% 
# PPO (Percentage Price Oscillator)
def ppo_signals(stock_data, fast_window=12, slow_window=26, signal_window=9):
    """
    Computes PPO crossover signals.

    Returns:
    A DataFrame with 'PPO_signal'.
    """
    signals = pd.DataFrame(index=stock_data.index)

    # Calculate PPO and PPO signal
    ppo = momentum.PercentagePriceOscillator(stock_data['Close'], fast_window, slow_window, signal_window)
    signals['PPO'] = ppo.ppo()
    signals['PPO_Signal'] = ppo.ppo_signal()

    # Generate buy/sell signals on PPO crossover
    signals['PPO_signal'] = 'neutral'
    signals.loc[(signals['PPO'] > signals['PPO_Signal']) & 
                (signals['PPO'].shift(1) <= signals['PPO_Signal'].shift(1)), 'PPO_signal'] = 'long'
    
    signals.loc[(signals['PPO'] < signals['PPO_Signal']) & 
                (signals['PPO'].shift(1) >= signals['PPO_Signal'].shift(1)), 'PPO_signal'] = 'short'

    return signals

#%% 
# Awesome Oscillator Zero Cross Strategy
def Awesome_Oscillator_signals(stock_df):
    """
    Computes Awesome Oscillator zero-crossing signals.

    Returns:
    A DataFrame with 'Ao_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Awesome Oscillator
    ao_indicator = momentum.AwesomeOscillatorIndicator(high=stock_df['High'], low=stock_df['Low'])
    signals['AO'] = ao_indicator.awesome_oscillator()

    # Generate signals on zero-crossing
    signals['Ao_signal'] = 'neutral'
    signals.loc[(signals['AO'].shift(1) < 0) & (signals['AO'] >= 0), 'Ao_signal'] = 'long'
    signals.loc[(signals['AO'].shift(1) >= 0) & (signals['AO'] < 0), 'Ao_signal'] = 'short'

    return signals

#%% 
# KAMA Cross Strategy
def kama_cross_signals(stock_df, fast_period=10, slow_period=20):
    """
    Computes KAMA crossover and price cross signals.

    Returns:
    A DataFrame with 'kama_cross_signal' and 'kama_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute Fast and Slow KAMA
    fast_kama = momentum.kama(stock_df['Close'], window=fast_period)
    slow_kama = momentum.kama(stock_df['Close'], window=slow_period)

    # Generate crossover signals
    signals['kama_cross_signal'] = 'neutral'
    signals.loc[(fast_kama > slow_kama) & (fast_kama.shift(1) <= slow_kama.shift(1)) & (stock_df['Close'] > fast_kama), 
                'kama_cross_signal'] = 'long'
    
    signals.loc[(fast_kama < slow_kama) & (fast_kama.shift(1) >= slow_kama.shift(1)) & (stock_df['Close'] < fast_kama), 
                'kama_cross_signal'] = 'short'

    # Generate price cross KAMA signals
    signals['kama_signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > fast_kama) & (stock_df['Close'].shift(1) <= fast_kama.shift(1)), 'kama_signal'] = 'long'
    signals.loc[(stock_df['Close'] < fast_kama) & (stock_df['Close'].shift(1) >= fast_kama.shift(1)), 'kama_signal'] = 'short'

    return signals

#%% 
# Stochastic Oscillator Strategy
def stoch_signals(stock_df, fast_period=10, slow_period=20):
    """
    Computes Stochastic Oscillator crossover signals.

    Returns:
    A DataFrame with 'stoch_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Stochastic Oscillator
    stoch = momentum.StochasticOscillator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=14, smooth_window=3)
    signals['%K'] = stoch.stoch()
    signals['%D'] = stoch.stoch_signal()

    # Generate crossover signals
    signals['stoch_signal'] = 'neutral'
    signals.loc[
        (signals['%K'] > signals['%D']) & (signals['%K'].shift(1) <= signals['%D'].shift(1)), 'stoch_signal'
    ] = 'long'
    
    signals.loc[
        (signals['%K'] < signals['%D']) & (signals['%K'].shift(1) >= signals['%D'].shift(1)), 'stoch_signal'
    ] = 'short'

    # Drop temporary columns
    signals.drop(['%K', '%D'], axis=1, inplace=True)

    return signals

#%% 
# True Strength Index (TSI) Strategy
def tsi_signals(stock_df, window_slow=25, window_fast=13):
    """
    Computes True Strength Index (TSI) crossover signals.

    Returns:
    A DataFrame with 'tsi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate TSI
    tsi = momentum.TSIIndicator(stock_df['Close'], window_slow, window_fast)
    signals['TSI'] = tsi.tsi()

    # Generate signals based on TSI zero-crossing
    signals['tsi_signal'] = 'neutral'
    signals.loc[(signals['TSI'] > 0) & (signals['TSI'].shift(1) <= 0), 'tsi_signal'] = 'long'
    signals.loc[(signals['TSI'] < 0) & (signals['TSI'].shift(1) >= 0), 'tsi_signal'] = 'short'

    # Drop temporary column
    signals.drop(['TSI'], axis=1, inplace=True)

    return signals

#%% 
# Williams %R Overbought/Oversold Strategy
def williams_signals(stock_df, lbp=14):
    """
    Computes Williams %R overbought and oversold signals.

    Returns:
    A DataFrame with 'williams_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Williams %R
    williams_r = momentum.WilliamsRIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], lbp)
    signals['WilliamsR'] = williams_r.williams_r()

    # Generate overbought/oversold signals
    signals['williams_signal'] = 'neutral'
    signals.loc[signals['WilliamsR'] <= -80, 'williams_signal'] = 'overbought'
    signals.loc[signals['WilliamsR'] >= -20, 'williams_signal'] = 'oversold'

    # Drop temporary column
    signals.drop(['WilliamsR'], axis=1, inplace=True)

    return signals

#%% 
# Rate of Change (ROC) Overbought/Oversold Strategy
def roc_signals(stock_df, window=12):
    """
    Computes ROC overbought and oversold signals.

    Returns:
    A DataFrame with 'roc_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Rate of Change (ROC)
    roc = momentum.ROCIndicator(stock_df['Close'], window)
    signals['ROC'] = roc.roc()

    # Generate overbought/oversold signals
    signals['roc_signal'] = 'neutral'
    signals.loc[signals['ROC'] >= 10, 'roc_signal'] = 'overbought'
    signals.loc[signals['ROC'] <= -10, 'roc_signal'] = 'oversold'

    # Drop temporary column
    signals.drop(['ROC'], axis=1, inplace=True)

    return signals

#%% 
# Relative Strength Index (RSI) Overbought/Oversold Strategy
def rsi_signals(stock_df, window=14):
    """
    Computes RSI overbought and oversold signals.

    Returns:
    A DataFrame with 'rsi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window)
    signals['RSI'] = rsi.rsi()

    # Generate overbought/oversold signals
    signals['rsi_signal'] = 'neutral'
    signals.loc[signals['RSI'] >= 70, 'rsi_signal'] = 'overbought'
    signals.loc[signals['RSI'] <= 30, 'rsi_signal'] = 'oversold'

    # Drop temporary column
    signals.drop(['RSI'], axis=1, inplace=True)

    return signals

#%% 
# Stochastic RSI Overbought/Oversold Strategy
def stochrsi_signals(stock_df, window=14, smooth1=3, smooth2=3):
    """
    Computes StochRSI overbought and oversold signals.

    Returns:
    A DataFrame with 'stochrsi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate StochRSI
    stoch_rsi = momentum.StochRSIIndicator(stock_df['Close'], window, smooth1, smooth2)
    signals['StochRSI'] = stoch_rsi.stochrsi()

    # Generate overbought/oversold signals
    signals['stochrsi_signal'] = 'neutral'
    signals.loc[signals['StochRSI'] >= 0.8, 'stochrsi_signal'] = 'overbought'
    signals.loc[signals['StochRSI'] <= 0.2, 'stochrsi_signal'] = 'oversold'

    # Drop temporary column
    signals.drop(['StochRSI'], axis=1, inplace=True)

    return signals

#%% 
# Commodity Channel Index (CCI) Strategy
def cci_signals(stock_df, window=20, constant=0.015, overbought=100, oversold=-100):
    """
    Computes CCI trend direction and trading signals.

    Returns:
    A DataFrame with 'cci_direction' and 'cci_Signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create CCI Indicator
    cci = trend.CCIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window, constant)
    signals['CCI'] = cci.cci()

    # Determine market direction
    signals['cci_direction'] = 'neutral'
    signals.loc[signals['CCI'] > oversold, 'cci_direction'] = 'bullish'
    signals.loc[signals['CCI'] < overbought, 'cci_direction'] = 'bearish'

    # Generate buy/sell signals based on overbought/oversold conditions
    signals['cci_Signal'] = 'neutral'
    signals.loc[(signals['CCI'] > overbought) & (signals['CCI'].shift(1) <= overbought), 'cci_Signal'] = 'long'
    signals.loc[(signals['CCI'] < oversold) & (signals['CCI'].shift(1) >= oversold), 'cci_Signal'] = 'short'

    # Drop temporary column
    signals.drop(['CCI'], axis=1, inplace=True)

    return signals

#%% 
# Detrended Price Oscillator (DPO) Strategy
def dpo_signals(stock_df, window=20):
    """
    Computes DPO trend direction and crossover signals.

    Returns:
    A DataFrame with 'dpo_direction_Signal' and 'dpo_Signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create DPO Indicator
    dpo = trend.DPOIndicator(stock_df['Close'], window)
    signals['DPO'] = dpo.dpo()

    # Determine market direction
    signals['dpo_direction_Signal'] = 'neutral'
    signals.loc[signals['DPO'] > 0, 'dpo_direction_Signal'] = 'bullish'
    signals.loc[signals['DPO'] < 0, 'dpo_direction_Signal'] = 'bearish'

    # Generate buy/sell signals based on zero-crossing
    signals['dpo_Signal'] = 'neutral'
    signals.loc[(signals['DPO'] > 0) & (signals['DPO'].shift(1) <= 0), 'dpo_Signal'] = 'long'
    signals.loc[(signals['DPO'] < 0) & (signals['DPO'].shift(1) >= 0), 'dpo_Signal'] = 'short'

    # Drop temporary column
    signals.drop(['DPO'], axis=1, inplace=True)

    return signals

#%% 
# Exponential Moving Average (EMA) Crossover Strategy
def ema_signals(stock_df, short_window=12, long_window=26):
    """
    Computes EMA crossover trend and trading signals.

    Returns:
    A DataFrame with 'EMA_Direction_Signal' and 'EMA_Signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate short-term and long-term EMAs
    ema_short = stock_df['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = stock_df['Close'].ewm(span=long_window, adjust=False).mean()

    # Determine market direction
    signals['EMA_Direction_Signal'] = 'neutral'
    signals.loc[ema_short > ema_long, 'EMA_Direction_Signal'] = 'bullish'
    signals.loc[ema_short < ema_long, 'EMA_Direction_Signal'] = 'bearish'

    # Generate buy/sell signals based on EMA crossovers
    signals['EMA_Signal'] = 'neutral'
    signals.loc[(ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1)), 'EMA_Signal'] = 'long'
    signals.loc[(ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1)), 'EMA_Signal'] = 'short'

    return signals

#%% 
# Ichimoku Cloud Strategy
def ichimoku_signals(stock_df, window1=9, window2=26):
    """
    Computes Ichimoku trend direction and crossover signals.

    Returns:
    A DataFrame with 'ichi_signal' and 'ichi_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create Ichimoku Indicator
    ichimoku = trend.IchimokuIndicator(stock_df['High'], stock_df['Low'], window1, window2)
    signals['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
    signals['kijun_sen'] = ichimoku.ichimoku_base_line()
    signals['senkou_span_a'] = ichimoku.ichimoku_a()
    signals['senkou_span_b'] = ichimoku.ichimoku_b()

    # Determine cloud color
    signals['cloud_color'] = 'neutral'
    signals.loc[signals['senkou_span_a'] > signals['senkou_span_b'], 'cloud_color'] = 'green'
    signals.loc[signals['senkou_span_a'] < signals['senkou_span_b'], 'cloud_color'] = 'red'

    # Generate crossover signals
    signals['ichi_signal'] = 'neutral'
    signals.loc[(signals['tenkan_sen'] > signals['kijun_sen']) & 
                (signals['tenkan_sen'].shift(1) <= signals['kijun_sen'].shift(1)), 'ichi_signal'] = 'long'
    
    signals.loc[(signals['tenkan_sen'] < signals['kijun_sen']) & 
                (signals['tenkan_sen'].shift(1) >= signals['kijun_sen'].shift(1)), 'ichi_signal'] = 'short'

    # Determine trend direction
    signals['ichi_direction'] = 'neutral'
    signals.loc[signals['cloud_color'] == 'green', 'ichi_direction'] = 'bullish'
    signals.loc[signals['cloud_color'] == 'red', 'ichi_direction'] = 'bearish'

    # Drop temporary columns
    signals.drop(['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'cloud_color'], axis=1, inplace=True)

    return signals

#%% 
# Know Sure Thing (KST) Strategy
def kst_signals(stock_df, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15, nsig=9):
    """
    Computes KST trend direction and crossover signals.

    Returns:
    A DataFrame with 'kst_signal' and 'kst_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create KST Indicator
    kst = trend.KSTIndicator(stock_df['Close'], roc1, roc2, roc3, roc4, window1, window2, window3, window4, nsig)
    signals['KST'] = kst.kst()
    signals['KST_Signal'] = kst.kst_sig()

    # Generate crossover signals
    signals['kst_signal'] = 'neutral'
    signals.loc[(signals['KST'] > signals['KST_Signal']) & 
                (signals['KST'].shift(1) <= signals['KST_Signal'].shift(1)), 'kst_signal'] = 'long'
    
    signals.loc[(signals['KST'] < signals['KST_Signal']) & 
                (signals['KST'].shift(1) >= signals['KST_Signal'].shift(1)), 'kst_signal'] = 'short'

    # Determine trend direction
    signals['kst_direction'] = 'neutral'
    signals.loc[signals['KST'] > signals['KST_Signal'], 'kst_direction'] = 'bullish'
    signals.loc[signals['KST'] < signals['KST_Signal'], 'kst_direction'] = 'bearish'

    # Drop temporary columns
    signals.drop(['KST', 'KST_Signal'], axis=1, inplace=True)

    return signals

#%% 
# Moving Average Convergence Divergence (MACD) Strategy
def macd_signals(stock_df, window_slow=26, window_fast=12, window_sign=9):
    """
    Computes MACD trend direction and crossover signals.

    Returns:
    A DataFrame with 'macd_signal' and 'macd_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create MACD Indicator
    macd = trend.MACD(stock_df['Close'], window_slow, window_fast, window_sign)
    signals['MACD'] = macd.macd()
    signals['MACD_Signal'] = macd.macd_signal()

    # Generate crossover signals
    signals['macd_signal'] = 'neutral'
    signals.loc[(signals['MACD'] > signals['MACD_Signal']) & 
                (signals['MACD'].shift(1) <= signals['MACD_Signal'].shift(1)), 'macd_signal'] = 'long'
    
    signals.loc[(signals['MACD'] < signals['MACD_Signal']) & 
                (signals['MACD'].shift(1) >= signals['MACD_Signal'].shift(1)), 'macd_signal'] = 'short'

    # Determine trend direction
    signals['macd_direction'] = 'neutral'
    signals.loc[signals['MACD'] > signals['MACD_Signal'], 'macd_direction'] = 'bullish'
    signals.loc[signals['MACD'] < signals['MACD_Signal'], 'macd_direction'] = 'bearish'

    # Drop temporary columns
    signals.drop(['MACD', 'MACD_Signal'], axis=1, inplace=True)

    return signals

#%% 
# Golden Cross SMA Strategy
def golden_ma_signals(stock_df, short_period=50, long_period=200):
    """
    Computes SMA crossover trend and trading signals.

    Returns:
    A DataFrame with 'ma_direction' and 'ma_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short and long SMAs
    short_sma = trend.SMAIndicator(stock_df['Close'], short_period).sma_indicator()
    long_sma = trend.SMAIndicator(stock_df['Close'], long_period).sma_indicator()

    # Determine trend direction
    signals['ma_direction'] = 'neutral'
    signals.loc[short_sma > long_sma, 'ma_direction'] = 'bullish'
    signals.loc[short_sma <= long_sma, 'ma_direction'] = 'bearish'

    # Generate crossover signals
    signals['ma_signal'] = 'neutral'
    signals.loc[(short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1)), 'ma_signal'] = 'long'
    signals.loc[(short_sma <= long_sma) & (short_sma.shift(1) > long_sma.shift(1)), 'ma_signal'] = 'short'

    return signals

#%% 
# 13-26 SMA Strategy
def short_ma_signals(stock_df, short_period=13, long_period=26):
    """
    Computes SMA crossover trend and trading signals.

    Returns:
    A DataFrame with 'ma_direction' and 'ma_signal'.
    """
    return golden_ma_signals(stock_df, short_period, long_period)

#%% 
# 5-8-13 SMA Strategy
def strategy_5_8_13(stock_df):
    """
    Computes 5-8-13 SMA crossover trend direction.

    Returns:
    A DataFrame with '5_8_13_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short SMAs
    sma5 = trend.SMAIndicator(stock_df['Close'], 5).sma_indicator()
    sma8 = trend.SMAIndicator(stock_df['Close'], 8).sma_indicator()
    sma13 = trend.SMAIndicator(stock_df['Close'], 13).sma_indicator()

    # Determine market direction
    signals['5_8_13_signal'] = 'neutral'
    signals.loc[(sma5 > sma8) & (sma8 > sma13), '5_8_13_signal'] = 'bullish'
    signals.loc[(sma5 < sma8) & (sma8 < sma13), '5_8_13_signal'] = 'bearish'

    return signals

#%% 
# 5-8-13 WMA Strategy
def strategy_w5_8_13(stock_df):
    """
    Computes 5-8-13 WMA crossover trend direction.

    Returns:
    A DataFrame with 'w5_8_13_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute weighted moving averages
    wma5 = trend.WMAIndicator(stock_df['Close'], 5).wma()
    wma8 = trend.WMAIndicator(stock_df['Close'], 8).wma()
    wma13 = trend.WMAIndicator(stock_df['Close'], 13).wma()

    # Determine market direction
    signals['w5_8_13_signal'] = 'neutral'
    signals.loc[(wma5 > wma8) & (wma8 > wma13), 'w5_8_13_signal'] = 'bullish'
    signals.loc[(wma5 < wma8) & (wma8 < wma13), 'w5_8_13_signal'] = 'bearish'

    return signals

#%% 
# Keltner Channel Strategy
def keltner_channel_strategy(stock_df, window=20, window_atr=10, multiplier=2):
    """
    Computes Keltner Channel breakout signals.

    Returns:
    A DataFrame with 'kc_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute Keltner Channel bands
    keltner_channel = volatility.KeltnerChannel(stock_df['High'], stock_df['Low'], stock_df['Close'], window, window_atr, multiplier)
    signals['kc_upper'] = keltner_channel.keltner_channel_hband()
    signals['kc_lower'] = keltner_channel.keltner_channel_lband()

    # Generate breakout signals
    signals['kc_signal'] = 'neutral'
    signals.loc[stock_df['Close'] > signals['kc_upper'], 'kc_signal'] = 'bearish'
    signals.loc[stock_df['Close'] < signals['kc_lower'], 'kc_signal'] = 'bullish'

    # Drop temporary columns
    signals.drop(['kc_upper', 'kc_lower'], axis=1, inplace=True)

    return signals

#%% 
# Chaikin Money Flow (CMF) Strategy
def cmf_signals(stock_df, window=20, threshold=0.1):
    """
    Computes CMF trend signals.

    Returns:
    A DataFrame with 'cmf_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute CMF
    cmf = volume.ChaikinMoneyFlowIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)
    signals['CMF'] = cmf.chaikin_money_flow()

    # Generate signals
    signals['cmf_signal'] = 'neutral'
    signals.loc[signals['CMF'] > threshold, 'cmf_signal'] = 'bearish'
    signals.loc[signals['CMF'] < -threshold, 'cmf_signal'] = 'bullish'

    # Drop temporary column
    signals.drop(['CMF'], axis=1, inplace=True)

    return signals

#%% 
# Money Flow Index (MFI) Strategy
def mfi_signals(stock_df, window=14):
    """
    Computes MFI overbought/oversold signals.

    Returns:
    A DataFrame with 'mfi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute MFI
    mfi = volume.MFIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)
    signals['MFI'] = mfi.money_flow_index()

    # Generate signals
    signals['mfi_signal'] = 'neutral'
    signals.loc[signals['MFI'] > 80, 'mfi_signal'] = 'overbought'
    signals.loc[signals['MFI'] < 20, 'mfi_signal'] = 'oversold'

    # Drop temporary column
    signals.drop(['MFI'], axis=1, inplace=True)

    return signals

#%% 
# Ease of Movement (EOM) Strategy
def eom_signals(stock_df, window=14):
    """
    Computes EOM trend signals.

    Returns:
    A DataFrame with 'eom_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute EOM
    eom = volume.EaseOfMovementIndicator(stock_df['High'], stock_df['Low'], stock_df['Volume'], window)
    signals['EOM'] = eom.ease_of_movement()

    # Generate signals
    signals['eom_signal'] = 'neutral'
    signals.loc[signals['EOM'] > 0, 'eom_signal'] = 'bullish'
    signals.loc[signals['EOM'] < 0, 'eom_signal'] = 'bearish'

    # Drop temporary column
    signals.drop(['EOM'], axis=1, inplace=True)

    return signals

#%% 
# Compute New Features Strategy
def compute_new_features(df):
    """
    Computes additional technical indicators as new features.

    Returns:
    A DataFrame with computed features.
    """
    df['rvi'] = volatility.bollinger_pband(df['Close'], window=14, fillna=True)
    df['cmo'] = momentum.roc(df['Close'], window=14, fillna=True)
    df['williams_r'] = momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14, fillna=True)
    df['donchian_high'] = df['High'].rolling(window=20).max()
    df['donchian_low'] = df['Low'].rolling(window=20).min()
    df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
    vortex = trend.VortexIndicator(df['High'], df['Low'], df['Close'], window=14, fillna=True)
    df['vortex_vi+'] = vortex.vortex_indicator_pos()
    df['vortex_vi-'] = vortex.vortex_indicator_neg()

    return df

#%%
def aroon_strategy(stock_df, window=25):
    """
    Computes Aroon trend strength and direction.

    Returns:
    A DataFrame with 'aroon_Trend_Strength', 'aroon_direction_signal', and 'aroon_signal'.
    """
    # Create Aroon Indicator
    aroon = trend.AroonIndicator(stock_df['Close'], stock_df['Low'], window=window)
    
    # Create a DataFrame to store signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['Aroon_Up'] = aroon.aroon_up()
    signals['Aroon_Down'] = aroon.aroon_down()

    # Determine trend strength
    signals['aroon_Trend_Strength'] = 'weak'
    signals.loc[(signals['Aroon_Up'] >= 70) | (signals['Aroon_Down'] >= 70), 'aroon_Trend_Strength'] = 'strong'

    # Determine direction signal
    signals['aroon_direction_signal'] = 'bearish'
    signals.loc[signals['Aroon_Up'] > signals['Aroon_Down'], 'aroon_direction_signal'] = 'bullish'

    # Generate trading signals
    signals['aroon_signal'] = 'neutral'
    signals.loc[
        (signals['aroon_direction_signal'] == 'bullish') & 
        (signals['aroon_direction_signal'].shift(1) == 'bearish'), 'aroon_signal'
    ] = 'long'
    
    signals.loc[
        (signals['aroon_direction_signal'] == 'bearish') & 
        (signals['aroon_direction_signal'].shift(1) == 'bullish'), 'aroon_signal'
    ] = 'short'

    # Drop temporary columns
    signals.drop(['Aroon_Up', 'Aroon_Down'], axis=1, inplace=True)

    return signals
#%%
def atr_signals(stock_df, atr_window=14, ema_window=20):
    """
    Computes ATR trend strength signals.

    Returns:
    A DataFrame with 'atr_trend_strength' column.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate ATR
    atr = volatility.AverageTrueRange(stock_df['High'], stock_df['Low'], stock_df['Close'], window=atr_window).average_true_range()

    # Calculate EMA of ATR
    atr_ema = trend.EMAIndicator(atr, window=ema_window).ema_indicator()

    # Determine trend strength
    signals['atr_trend_strength'] = 'weak'
    signals.loc[atr > atr_ema, 'atr_trend_strength'] = 'strong'

    return signals
#%%
#RSI Divergence strategy

def rsi_signals_with_divergence(stock_df, window=14, long_threshold=30, short_threshold=70, width=10):
    """
    Computes RSI signals with divergence detection.

    Returns:
    A DataFrame with 'RSI_signal' and 'RSI' values.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate RSI
    rsi_indicator = momentum.RSIIndicator(stock_df['Close'], window=window)
    signals['RSI'] = rsi_indicator.rsi()

    # Generate RSI overbought/oversold signals
    signals['RSI_signal'] = 'neutral'
    signals.loc[signals['RSI'] < long_threshold, 'RSI_signal'] = 'long'
    signals.loc[signals['RSI'] > short_threshold, 'RSI_signal'] = 'short'

    # **Divergence Detection**
    signals['Divergence'] = 'none'

    # Find local peaks (potential bearish divergence) & valleys (potential bullish divergence)
    local_max = signals['RSI'].rolling(window=width, center=True).max()
    local_min = signals['RSI'].rolling(window=width, center=True).min()

    # Identify divergence (only if RSI is at extremes)
    signals.loc[(signals['RSI'] == local_max) & (signals['RSI'] > short_threshold), 'Divergence'] = 'bearish'
    signals.loc[(signals['RSI'] == local_min) & (signals['RSI'] < long_threshold), 'Divergence'] = 'bullish'

    return signals

#%%
# ADX 

def adx_strength_direction(stock_df, window=14):
    """
    Calculates ADX trend strength and direction.

    Returns:
    A DataFrame with 'adx_Trend_Strength' and 'adx_Direction' columns.
    """
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=window)

    # Store in a signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['ADX'] = adx.adx()
    signals['adx_Trend_Strength'] = 'neutral'
    signals['adx_Direction'] = 'neutral'

    # Determine trend strength
    signals.loc[signals['ADX'] > 25, 'adx_Trend_Strength'] = 'strong'
    signals.loc[signals['ADX'] < 20, 'adx_Trend_Strength'] = 'weak'

    # Determine trend direction
    signals.loc[adx.adx_pos() > adx.adx_neg(), 'adx_Direction'] = 'bullish'
    signals.loc[adx.adx_pos() < adx.adx_neg(), 'adx_Direction'] = 'bearish'

    return signals

# %%
# Mass Index

def mass_index_signals(stock_df, fast_window=9, slow_window=25):
    """
    Computes Mass Index reversal signals.

    Returns:
    A DataFrame with 'Mass_Index' and 'mass_signal' columns.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Mass Index
    mass_index = trend.MassIndex(stock_df['High'], stock_df['Low'], window_fast=fast_window, window_slow=slow_window)
    signals['Mass_Index'] = mass_index.mass_index()

    # Calculate Short & Long EMAs for trend direction
    short_ema = trend.EMAIndicator(stock_df['Close'], window=fast_window).ema_indicator()
    long_ema = trend.EMAIndicator(stock_df['Close'], window=slow_window).ema_indicator()

    # Set thresholds for Mass Index reversals
    reversal_bulge_threshold = 27
    reversal_bulge_exit_threshold = 26.5

    # Determine if the market is in a downtrend
    in_downtrend = short_ema < long_ema

    # Generate signals for reversals
    signals['mass_signal'] = 'neutral'
    signals.loc[(in_downtrend) & (signals['Mass_Index'] > reversal_bulge_threshold) & 
                (signals['Mass_Index'].shift(1) <= reversal_bulge_exit_threshold), 'mass_signal'] = 'long'
    
    signals.loc[(~in_downtrend) & (signals['Mass_Index'] > reversal_bulge_threshold) & 
                (signals['Mass_Index'].shift(1) <= reversal_bulge_exit_threshold), 'mass_signal'] = 'short'

    return signals
#%%
# PSAR 

def psar_signals(stock_df, step=0.02, max_step=0.2):
    """
    Computes Parabolic SAR trend direction and signals.

    Returns:
    A DataFrame with 'psar_direction' and 'psar_signal' columns.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute PSAR values
    psar_indicator = trend.PSARIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], step=step, max_step=max_step)
    psar_values = psar_indicator.psar()

    # Determine PSAR trend direction
    signals['psar_direction'] = 'neutral'
    signals.loc[stock_df['Close'] > psar_values, 'psar_direction'] = 'bullish'
    signals.loc[stock_df['Close'] < psar_values, 'psar_direction'] = 'bearish'

    # Generate buy/sell signals based on crossovers
    signals['psar_signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > psar_values) & (stock_df['Close'].shift(1) <= psar_values.shift(1)), 'psar_signal'] = 'long'
    signals.loc[(stock_df['Close'] < psar_values) & (stock_df['Close'].shift(1) >= psar_values.shift(1)), 'psar_signal'] = 'short'

    return signals

#%%
# STC

def stc_signals(stock_df, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3):
    """
    Computes Schaff Trend Cycle (STC) signals.

    Returns:
    A DataFrame with 'stc_signal' and 'stc_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate STC
    stc_indicator = trend.STCIndicator(stock_df['Close'], window_slow, window_fast, cycle, smooth1, smooth2)
    signals['STC'] = stc_indicator.stc()

    # Determine overbought/oversold conditions
    signals['stc_signal'] = 'neutral'
    signals.loc[signals['STC'] > 75, 'stc_signal'] = 'overbought'
    signals.loc[signals['STC'] < 25, 'stc_signal'] = 'oversold'

    # Determine trend direction
    signals['stc_direction'] = 'neutral'
    signals.loc[signals['STC'] > 50, 'stc_direction'] = 'bullish'
    signals.loc[signals['STC'] < 50, 'stc_direction'] = 'bearish'

    return signals

#%%
# Vortex
def vortex_signals(stock_df, window=14):
    """
    Computes Vortex Indicator trend direction and signals.

    Returns:
    A DataFrame with 'vortex_signal' and 'vortex_direction_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute Vortex Indicator values
    vortex = trend.VortexIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=window)
    signals['Positive'] = vortex.vortex_indicator_pos()
    signals['Negative'] = vortex.vortex_indicator_neg()

    # Determine bullish/bearish trend direction
    signals['vortex_direction_signal'] = 'neutral'
    signals.loc[signals['Positive'] > signals['Negative'], 'vortex_direction_signal'] = 'bullish'
    signals.loc[signals['Positive'] < signals['Negative'], 'vortex_direction_signal'] = 'bearish'

    # Generate trading signals based on crossovers
    signals['vortex_signal'] = 'neutral'
    signals.loc[
        (signals['Positive'] > signals['Negative']) & 
        (signals['Positive'].shift(1) <= signals['Negative'].shift(1)), 'vortex_signal'
    ] = 'long'
    
    signals.loc[
        (signals['Positive'] < signals['Negative']) & 
        (signals['Positive'].shift(1) >= signals['Negative'].shift(1)), 'vortex_signal'
    ] = 'short'

    # Drop temporary columns
    signals.drop(['Positive', 'Negative'], axis=1, inplace=True)

    return signals
#%%
# Golden Cross WMA

def golden_wma_signals(stock_df, short_period=50, long_period=200):
    """
    Computes Weighted Moving Average (WMA) crossover signals.

    Returns:
    A DataFrame with 'wma_direction' and 'wma_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short and long WMAs
    short_wma = trend.WMAIndicator(stock_df['Close'], short_period).wma()
    long_wma = trend.WMAIndicator(stock_df['Close'], long_period).wma()

    # Determine trend direction
    signals['wma_direction'] = 'neutral'
    signals.loc[short_wma > long_wma, 'wma_direction'] = 'bullish'
    signals.loc[short_wma <= long_wma, 'wma_direction'] = 'bearish'

    # Generate crossover signals
    signals['wma_signal'] = 'neutral'
    signals.loc[
        (short_wma > long_wma) & 
        (short_wma.shift(1) <= long_wma.shift(1)), 'wma_signal'
    ] = 'long'
    
    signals.loc[
        (short_wma <= long_wma) & 
        (short_wma.shift(1) > long_wma.shift(1)), 'wma_signal'
    ] = 'short'

    return signals
#%%
# 13-26 MA STrategy

def short_wma_signals(stock_df, short_period=13, long_period=26):
    """
    Computes 13-26 Weighted Moving Average (WMA) crossover signals.

    Returns:
    A DataFrame with 'wma_direction' and 'wma_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short and long WMAs
    short_wma = trend.WMAIndicator(stock_df['Close'], short_period).wma()
    long_wma = trend.WMAIndicator(stock_df['Close'], long_period).wma()

    # Determine trend direction
    signals['wma_direction'] = 'neutral'
    signals.loc[short_wma > long_wma, 'wma_direction'] = 'bullish'
    signals.loc[short_wma <= long_wma, 'wma_direction'] = 'bearish'

    # Generate crossover signals
    signals['wma_signal'] = 'neutral'
    signals.loc[
        (short_wma > long_wma) & (short_wma.shift(1) <= long_wma.shift(1)), 'wma_signal'
    ] = 'long'
    
    signals.loc[
        (short_wma <= long_wma) & (short_wma.shift(1) > long_wma.shift(1)), 'wma_signal'
    ] = 'short'

    return signals

# %%
# Donchain Channel

def donchian_channel_strategy(stock_df, window=20):
    """
    Computes Donchian Channel breakout signals.

    Returns:
    A DataFrame with 'dc_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Donchian Channel
    donchian = volatility.DonchianChannel(stock_df['High'], stock_df['Low'], stock_df['Close'], window=window)
    signals['Upper_Channel'] = donchian.donchian_channel_hband()
    signals['Lower_Channel'] = donchian.donchian_channel_lband()

    # Determine long and short signals (breakout strategy)
    signals['dc_signal'] = 'neutral'
    signals.loc[stock_df['Close'] > signals['Upper_Channel'], 'dc_signal'] = 'long'
    signals.loc[stock_df['Close'] < signals['Lower_Channel'], 'dc_signal'] = 'short'

    # Drop temporary columns
    signals.drop(['Upper_Channel', 'Lower_Channel'], axis=1, inplace=True)

    return signals

def turnaround_tuesday_strategy(stock_df):
    """
    Implements the Turnaround Tuesday strategy.
    
    Returns:
    A DataFrame with 'Date', 'Signal', 'EntryPrice', and 'ExitPrice'.
    """
    # Ensure necessary columns exist
    if 'Date' not in stock_df.columns or 'Open' not in stock_df.columns or 'Close' not in stock_df.columns:
        raise ValueError("Input DataFrame must have 'Date', 'Open', and 'Close' columns.")

    # Convert Date to datetime format
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # Calculate Internal Bar Strength (IBS)
    stock_df['IBS'] = (stock_df['Close'] - stock_df['Open']) / (stock_df['High'] - stock_df['Low'])

    # Filter for Mondays
    monday_data = stock_df[stock_df['Date'].dt.day_name() == 'Monday']

    # Initialize signals DataFrame
    signals = pd.DataFrame(columns=['Date', 'Signal', 'EntryPrice', 'ExitPrice'])

    # Iterate through Mondays to find eligible signals
    for _, row in monday_data.iterrows():
        if row['Close'] < row['Open'] and row['IBS'] < 0.2:
            buy_date = row['Date']
            entry_price = row['Close']

            # Find corresponding Tuesday's closing price
            tuesday_data = stock_df[stock_df['Date'] == buy_date + pd.DateOffset(days=1)]
            exit_price = tuesday_data['Close'].values[0] if not tuesday_data.empty else None

            # Append signal
            signals = pd.concat([signals, pd.DataFrame([{
                'Date': buy_date, 'Signal': 'Long', 'EntryPrice': entry_price, 'ExitPrice': exit_price
            }])], ignore_index=True)

    return signals