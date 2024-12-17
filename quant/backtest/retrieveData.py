import yfinance as yf
import pandas

def test_yfinance():
    for symbol in ['IGE']:
        print(">>", symbol, end=' ... ')
        data = yf.download(symbol, start='2001-11-26', end='2007-11-26')
        data.to_csv('IGE.csv')

if __name__ == "__main__":
    test_yfinance()