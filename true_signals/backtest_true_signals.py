import pandas as pd
from backtesting import Backtest, Strategy

data = pd.read_csv("../Trading_Bot/SPY.csv")
data.drop(columns=['Date', 'Adj Close'], inplace=True)
signal = pd.read_csv("./true_signals/SPY_true_signals.csv")
signal.drop(columns=['Close'], inplace=True)
# Encode the 'signals' column
signal['signals'] = signal['signals'].map({'long': 1, 'short': -1, 0: 0})
df = pd.concat([data, signal], axis=1)
df = df.fillna(0)
df = df.replace('nan', 0)

closepos_column = df.columns[-1]
closepos_values = df[closepos_column]

signal_column = df.columns[-2]
signal_values = df[signal_column]

#%%
def SIGNAL():
    return signal_values

def CLOSE_POS():
    return closepos_values

# Define your strategy class
class MyStrategy(Strategy):
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)
        self.close_position = self.I(CLOSE_POS)

    def next(self):
        super().next()

        # Check if closepos_values == 1, if true, close any current open position
        if self.close_position == 1:
            if self.position:
                self.position.close()  # Close the current open position

        # Check the 'signals' column to open a new position
        if self.signal1 == 1:
            if not self.position:
                self.buy()  # Open a long position
            elif self.position.size < 0:
                self.position.close()
                self.buy()

        elif self.signal1 == -1:
            if not self.position:
                self.sell()  # Open a short position
            elif self.position.size > 0:
                self.position.close()
                self.sell()

#%%
# Backtest your strategy
bt = Backtest(data, MyStrategy, cash=100000)
stats = bt.run()
print(stats)
