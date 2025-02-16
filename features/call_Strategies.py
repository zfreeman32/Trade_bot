#%%
import sys
sys.path.append('./Strategies/all_Strategies.py')
import pandas as pd
from . import all_Strategies

def generate_all_signals(stock_df):

    # Call your functions and store their results in separate DataFrames
    ppo_signals_df = all_Strategies.ppo_signals(stock_df)
    awesome_oscillator_signals_df = all_Strategies.Awesome_Oscillator_signals(stock_df)
    kama_cross_signals_df = all_Strategies.kama_cross_signals(stock_df)
    williams_signals_df = all_Strategies.williams_signals(stock_df)
    tsi_signals_df = all_Strategies.tsi_signals(stock_df)
    stoch_signals_df = all_Strategies.stoch_signals(stock_df)
    roc_signals_df = all_Strategies.roc_signals(stock_df)
    rsi_signals_df = all_Strategies.rsi_signals(stock_df)
    stochrsi_signals_df = all_Strategies.stochrsi_signals(stock_df)
    # aroon_signals_df = all_Strategies.aroon_strategy(stock_df)
    cci_signals_df = all_Strategies.cci_signals(stock_df)
    dpo_signals_df = all_Strategies.dpo_signals(stock_df)
    ema_signals_df = all_Strategies.ema_signals(stock_df)
    ichimoku_signals_df = all_Strategies.ichimoku_signals(stock_df)
    kst_signals_df = all_Strategies.kst_signals(stock_df)
    macd_signals_df = all_Strategies.macd_signals(stock_df)
    golden_ma_signals_df = all_Strategies.golden_ma_signals(stock_df)
    short_ma_signals_df = all_Strategies.short_ma_signals(stock_df)
    strategy_5_8_13_df = all_Strategies.strategy_5_8_13(stock_df)
    # atr_signals_df = all_Strategies.atr_signals(stock_df)
    keltner_channel_strategy_df = all_Strategies.keltner_channel_strategy(stock_df)
    cmf_signals_df = all_Strategies.cmf_signals(stock_df)
    eom_signals_df = all_Strategies.eom_signals(stock_df)
    mfi_signals_df = all_Strategies.mfi_signals(stock_df)
    strategy_w5_8_13_df = all_Strategies.strategy_w5_8_13(stock_df)

    # Concatenate the results into one large DataFrame
    all_signals_df = pd.concat([stock_df, strategy_w5_8_13_df, mfi_signals_df, eom_signals_df, cmf_signals_df, keltner_channel_strategy_df, cmf_signals_df, strategy_5_8_13_df, short_ma_signals_df, golden_ma_signals_df, macd_signals_df, kst_signals_df, ichimoku_signals_df, ema_signals_df, dpo_signals_df, cci_signals_df, roc_signals_df, rsi_signals_df, stochrsi_signals_df, stoch_signals_df, tsi_signals_df, williams_signals_df, kama_cross_signals_df, ppo_signals_df, awesome_oscillator_signals_df], axis=1)
    
    return all_signals_df

# signals_df = generate_all_signals(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\VIX.csv', r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\VIX.csv')
# %%
