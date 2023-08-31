#%%
import pandas as pd
from ta import volume

#%%
# Chaikin Money Flow
def cmf_signals(stock_df, window=20, threshold=0.1):
    signals = pd.DataFrame(index=stock_df.index)
    signals['cmf_signal'] = 0

    # Calculate Chaikin Money Flow (CMF)
    cmf = volume.ChaikinMoneyFlowIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)

    signals['CMF'] = cmf.chaikin_money_flow()

    # Generate buy (1) and sell (-1) signals based on CMF
    for i in range(1, len(signals)):
        if signals['CMF'][i] > threshold:  # You can adjust this threshold as needed
            signals.loc[i, 'cmf_signal'] = ['buy']  # Buy signal (CMF is positive)
        elif signals['CMF'][i] < -threshold:  # You can adjust this threshold as needed
            signals.loc[i, 'cmf_signal'] = ['sell']  # Sell signal (CMF is negative)

    return signals

#%%
# MFI
def mfi_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['mfi_signal'] = 0

    # Calculate Money Flow Index (MFI)
    mfi = volume.MFIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)

    signals['MFI'] = mfi.money_flow_index()

    # Generate buy (1) and sell (-1) signals based on MFI
    for i in range(1, len(signals)):
        if signals['MFI'][i] > 90:  # You can adjust this overbought threshold as needed
            signals.loc[i, 'mfi_signal'] = ['overbought']  # Sell signal (MFI is overbought)
        elif signals['MFI'][i] < 10:  # You can adjust this oversold threshold as needed
            signals.loc[i, 'mfi_signal'] = ['oversold']  # Buy signal (MFI is oversold)

    return signals

#%%
# 

# %%

def eom_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['eom_signal'] = 0

    # Calculate Ease of Movement (EOM) Indicator
    eom = volume.EaseOfMovementIndicator(stock_df['High'], stock_df['Low'], stock_df['Volume'], window)

    signals['EOM'] = eom.ease_of_movement()

    # Generate bullish (1) and bearish (-1) signals based on EOM
    for i in range(1, len(signals)):
        if signals['EOM'][i] > 0:  # You can adjust this threshold as needed
            signals.loc[i, 'eom_signal'] = ['bullish']  # Bullish signal (EOM is positive)
        elif signals['EOM'][i] < 0:  # You can adjust this threshold as needed
            signals.loc[i, 'eom_signal'] = ['bearish']   # Bearish signal (EOM is negative)

    return signals