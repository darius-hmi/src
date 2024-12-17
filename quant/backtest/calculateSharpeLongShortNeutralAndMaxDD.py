import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def test_yfinance(stock):
    df = yf.download(stock, start='2021-01-01', end='2024-01-01')
    df.reset_index(inplace=True)
    return df

stockBuy = 'IGE'
stockShort = 'SPY'

df1 = test_yfinance(stockBuy)
df2 = test_yfinance(stockShort)
df = pd.merge(df1, df2, on='Date', suffixes=('_'+stockBuy, '_'+stockShort))
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

adjColBuy = 'Adj Close_'+stockBuy
adjColShort = 'Adj Close_'+stockShort

dailyRet = df[[adjColBuy, adjColShort]].pct_change()
netRet = (dailyRet[adjColBuy]-dailyRet[adjColShort])/2
SharpeRatio = np.sqrt(252) * np.mean(netRet)/np.std(netRet)
print(SharpeRatio)


cumRet = np.cumprod(1+netRet)-1
print(cumRet)
plt.plot(cumRet)
plt.show()


def calculateMaxDD(cumRet):
# =============================================================================
# calculation of maximum drawdown and maximum drawdown duration based on
# cumulative COMPOUNDED returns. cumret must be a compounded cumulative return.
# i is the index of the day with maxDD.
# =============================================================================
    highwatermark=np.zeros(cumRet.shape)
    drawdown=np.zeros(cumRet.shape)
    drawdownduration=np.zeros(cumRet.shape)
    
    for t in np.arange(1, cumRet.shape[0]):
        highwatermark[t]=np.maximum(highwatermark[t-1], cumRet[t])
        drawdown[t]=(1+cumRet[t])/(1+highwatermark[t])-1
        if drawdown[t]==0:
            drawdownduration[t]=0
        else:
            drawdownduration[t]=drawdownduration[t-1]+1
             
    maxDD, i=np.min(drawdown), np.argmin(drawdown) # drawdown < 0 always
    maxDDD=np.max(drawdownduration)
    return maxDD, maxDDD, i

maxDrawdown, maxDrawdownDuration, startDrawdownDay=calculateMaxDD(cumRet.values)

print('max Drawdown: ', maxDrawdown)
print('Max Drawdown Duration:', maxDrawdownDuration)
print('Max Drawdown Day: ', startDrawdownDay)