#%%
import pandas as pd

# Turnaround Tuesday
# Today is Monday.
# The close must be lower than the open.
# The IBS must be below 0.2.
# If 1-3 are true, then enter at the close.
# Sell at Tuesday’s close.

#%%

def turnaround_tuesday_strategy(data):
    if 'Date' not in data.columns or 'Open' not in data.columns or 'Close' not in data.columns:
        raise ValueError("Input DataFrame must have 'Date', 'Open', and 'Close' columns.")

    data['High'] = data['High'].shift(1)
    data['Low'] = data['Low'].shift(1)
    data['IBS'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'])
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')  # Corrected to 'data'

    signals = []

    monday_data = data[data['Date'].dt.day_name() == 'Monday']

    for index, row in monday_data.iterrows():  # Corrected to unpack 'index' and 'row'
        if row['Close'] < row['Open'] and row['IBS'] < 0.2:
            buy_date = row['Date']
            entry_price = row['Close']

            # Find Tuesday's close price for the buy signal
            tuesday_data = data[data['Date'].dt.day_name() == 'Tuesday']
            tuesday_close = tuesday_data[tuesday_data['Date'] == buy_date + pd.DateOffset(days=1)]['Close'].values

            if len(tuesday_close) > 0:
                exit_price = tuesday_close[0]
            else:
                exit_price = None

            signals.append({'Date': buy_date, 'Signal': 'Long', 'EntryPrice': entry_price, 'ExitPrice': exit_price})

    return pd.DataFrame(signals)

#%%
csv_file = r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv'
stock_df = pd.read_csv(csv_file)
print(stock_df)
turnaround_tuesday = pd.DataFrame(turnaround_tuesday_strategy(stock_df))
print(turnaround_tuesday)
# %%

