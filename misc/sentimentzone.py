
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator

# SentimentZone Oscillator (SZO) Strategy
def sentiment_zone_signals(stock_df, length=30, long_length=60, oversold=-7):
    signals = pd.DataFrame(index=stock_df.index)
    # Assuming SZO Indicator Function is available.
    szo = SZO(stock_df['Close'], length)
    sma_szo = SMAIndicator(szo, length)
    ema_price = EMAIndicator(stock_df['Close'], long_length)

    # Buy conditions
    signals['buy_signal'] = ((sma_szo.sma_indicator().crosses_above(0) & (stock_df['Close'] > ema_price.ema_indicator()))
                   | ((szo < oversold) & sma_szo.sma_indicator().rising() & (stock_df['Close'] > ema_price.ema_indicator()))
                   | ((szo.crosses_above(oversold)) & (sma_szo.sma_indicator() > 0) & ema_price.ema_indicator().rising()))

    # Sell conditions
    signals['sell_signal'] = ((sma_szo.sma_indicator().crosses_below(0))
                   | (szo.crosses_below(7) & sma_szo.sma_indicator().declining()))

    return signals
