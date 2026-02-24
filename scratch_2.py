import numpy as np
import yfinance as yf
import pandas as pd
import library.settings  as settings  # constants/paths

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


tickers_dict = settings.TICKERS_DICT

datapoints_list = ['sector', 'trailingPE', 'forwardPE', 'priceToBook', 'trailingEps', 'epsForward', 'epsCurrentYear', 'targetMedianPrice', 'targetLowPrice', 'targetHighPrice', 'lastDividendValue' ]

ticker_static_info = pd.DataFrame(columns = datapoints_list)
ticker_static_info.insert(0, "ticker", list(tickers_dict.keys()))


for ticker in tickers_dict:

    for datapoint in datapoints_list:
        try:
            ticker_static_info.loc[ticker_static_info['ticker'] == ticker, datapoint] = yf.Ticker(tickers_dict[ticker][0]).info[datapoint]
        except:
            ticker_static_info.loc[ticker_static_info['ticker'] == ticker, datapoint]  = np.nan




tickers_dict.keys()
tickers_dict['NVDA.US'][0]

tickers_dict['NVDA.US']

try:
    yf.Ticker("IUSA.AS").get_analyst_price_targets()
except Exception  as e:
    print(f"symbol not available (HTTP error): {e}")


x.info['sector']