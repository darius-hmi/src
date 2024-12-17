import numpy as np
import pandas as pd
import matplotlib as plt
import yfinance as yf

def test_yfinance(stock):
    df = yf.download(stock, start='2021-01-01', end='2024-01-01')
    return df

df = test_yfinance('MSFT')
df.sort_values(by='Date', inplace=True)

dailyRet = df.loc[:, 'Adj Close'].pct_change()
riskFreeRate = 0.04
ExcessDailyReturn = riskFreeRate/253
excessRet = dailyRet - ExcessDailyReturn
SharpeRatio = np.sqrt(252) * np.mean(excessRet)/np.std(excessRet)
print(SharpeRatio)
