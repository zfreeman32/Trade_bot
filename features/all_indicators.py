#%%
import numpy as np
import pandas as pd
import talib

def generate_all_indicators(df):
    """
    Generate a comprehensive set of technical indicators using TA-Lib.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with OHLCV data. Must have columns: 'Open', 'High', 'Low', 'Close', 'Volume'
    
    Returns:
    pandas.DataFrame: Original dataframe with all technical indicators added

    147 Indicators from TA-Lib and Manually Calculated
    """
    
    open_price = df['Open'].values.astype(np.float64)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    close = df['Close'].values.astype(np.float64)
    volume = df['Volume'].values.astype(np.float64)
    
    try:
        # Overlap Studies
        df['UPPERBAND'], df['MIDDLEBAND'], df['LOWERBAND'] = talib.BBANDS(close, timeperiod=20)
        df['DEMA'] = talib.DEMA(close, timeperiod=30)
        df['EMA_10'] = talib.EMA(close, timeperiod=10)
        df['EMA_20'] = talib.EMA(close, timeperiod=20)
        df['EMA_50'] = talib.EMA(close, timeperiod=50)
        df['EMA_200'] = talib.EMA(close, timeperiod=200)
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close)
        df['KAMA'] = talib.KAMA(close, timeperiod=30)
        df['MA_10'] = talib.MA(close, timeperiod=10)
        df['MA_20'] = talib.MA(close, timeperiod=20)
        df['MA_50'] = talib.MA(close, timeperiod=50)
        df['MA_200'] = talib.MA(close, timeperiod=200)
        df['MAMA'], df['FAMA'] = talib.MAMA(close)
        df['SAR'] = talib.SAR(high, low)
        df['SAREXT'] = talib.SAREXT(high, low)
        df['SMA'] = talib.SMA(close, timeperiod=30)
        df['T3'] = talib.T3(close, timeperiod=5)
        df['TEMA'] = talib.TEMA(close, timeperiod=30)
        df['TRIMA'] = talib.TRIMA(close, timeperiod=30)
        df['WMA'] = talib.WMA(close, timeperiod=30)
        
        # Momentum Indicators
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
        df['APO'] = talib.APO(close)
        df['AROON_DOWN'], df['AROON_UP'] = talib.AROON(high, low, timeperiod=14)
        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
        df['BOP'] = talib.BOP(open_price, high, low, close)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        df['CMO'] = talib.CMO(close, timeperiod=14)
        df['DX'] = talib.DX(high, low, close, timeperiod=14)
        df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(close)
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
        df['MOM'] = talib.MOM(close, timeperiod=10)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
        df['PPO'] = talib.PPO(close)
        df['ROC'] = talib.ROC(close, timeperiod=10)
        df['ROCP'] = talib.ROCP(close, timeperiod=10)
        df['ROCR'] = talib.ROCR(close, timeperiod=10)
        df['ROCR100'] = talib.ROCR100(close, timeperiod=10)
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close)
        df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(high, low, close)
        df['STOCHRSI_K'], df['STOCHRSI_D'] = talib.STOCHRSI(close, timeperiod=14)
        df['TRIX'] = talib.TRIX(close, timeperiod=30)
        df['ULTOSC'] = talib.ULTOSC(high, low, close)
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Volume Indicators
        df['AD'] = talib.AD(high, low, close, volume)
        df['ADOSC'] = talib.ADOSC(high, low, close, volume)
        df['OBV'] = talib.OBV(close, volume)
        
        # Volatility Indicators
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
        df['TRANGE'] = talib.TRANGE(high, low, close)
        
        # Price Transform
        df['AVGPRICE'] = talib.AVGPRICE(open_price, high, low, close)
        df['MEDPRICE'] = talib.MEDPRICE(high, low)
        df['TYPPRICE'] = talib.TYPPRICE(high, low, close)
        df['WCLPRICE'] = talib.WCLPRICE(high, low, close)
        
        # Cycle Indicators
        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
        df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
        df['HT_PHASOR_INPHASE'], df['HT_PHASOR_QUADRATURE'] = talib.HT_PHASOR(close)
        df['HT_SINE'], df['HT_LEADSINE'] = talib.HT_SINE(close)
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

        # Statistical Indicators
        df['rolling_mean'] = df['Close'].rolling(window=14).mean()
        df['rolling_std'] = df['Close'].rolling(window=14).std()
        df['z_score'] = (df['Close'] - df['rolling_mean']) / df['rolling_std']

        # Additional Indicators
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['HMA'] = talib.WMA(2 * talib.WMA(close, timeperiod=10 // 2) - talib.WMA(close, timeperiod=10), timeperiod=int(np.sqrt(10)))
        df['KELTNER_UPPER'] = talib.SMA(close, timeperiod=20) + (2 * talib.ATR(high, low, close, timeperiod=10))
        df['KELTNER_LOWER'] = talib.SMA(close, timeperiod=20) - (2 * talib.ATR(high, low, close, timeperiod=10))
        df['DONCHIAN_HIGH'] = df['High'].rolling(window=20).max()
        df['DONCHIAN_LOW'] = df['Low'].rolling(window=20).min()
        df['EFI'] = talib.EMA((df['Close'] - df['Close'].shift(1)) * df['Volume'], timeperiod=13)
        df['SUPER_TREND'] = df['Close'] - df['ATR'] * 2  # Basic supertrend calculation

        # Candlestick Patterns
        patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS',
            'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR',
            'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE',
            'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER',
            'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
            'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
            'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING',
            'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
        ]
        for pattern in patterns:
            df[pattern] = getattr(talib, pattern)(open_price, high, low, close)
        
    except Exception as e:
        print(f"Error generating indicators: {str(e)}")
        return None

    return df