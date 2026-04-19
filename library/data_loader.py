"""
data_loader.py

This module contains functions to load various data (CSV, yfinance) and to clean them

TODO:
    generalize the splits, so that it works not only for nvidia, but also chargepoint etc...
    create tools to export data or to even store in a DB
"""

#-- Import of packages
import glob
import os
import pandas as pd
import re
import yfinance as yf
import logging
import sys
import numpy as np
from yfinance import tickers

import library.settings  as settings

logger = logging.getLogger(__name__)

source = settings.XTB_INPUT_FILE_PATH


#--- Checks ---
def check_yfinance_connection():
    """
    Checks whether connection to yahoo finance can be established
    """
    try:
        x = yf.Ticker("AAPL").history(period = '1d')
        if x is not None: logger.info("Connection to Yahoo Finance was successful")
               
    except Exception as e:
        logger.warning(f"Error: {e} \nCould not connect to Yahoo Finance")

#--- CSV file loader helper functions
def get_most_recent_excel_file(folder_path):
    """
    Selects the most recently modified file in particular folder
    """
    excel_files = glob.glob(os.path.join(folder_path, '*.xls*'))
    
    if not excel_files:
        return None  # No Excel files found
        
    most_recent_file = max(excel_files, key=os.path.getmtime)
    return most_recent_file

#--- CSV File loader
def load_prtf_file(file_name = None, folder_path = None):
    """
    Loads the selected file from XTB, either explicitly specified or
    the most recent file in a specific folder, for which it uses the get_most_recent_excel_file function
    """

    if file_name != None:
        if os.path.exists(file_name) == False:
            logger.error("specified filename does not exist: " + file_name)
            sys.exit(1)
        else:
            prtf_file = pd.read_excel(file_name, sheet_name = 'CASH OPERATION HISTORY', header = 10, index_col = 1 )
            logger.info('file loaded: ' + file_name)
            
    if folder_path != None:
        if os.path.exists(folder_path) == False:
            logger.error("specified folder does not exist: " + folder_path)  
            sys.exit(1)
        else:
            most_recent_file = get_most_recent_excel_file(folder_path)
            prtf_file = pd.read_excel(most_recent_file, sheet_name = 'CASH OPERATION HISTORY', header = 10, index_col = 1 )
            logger.info('most recent file was loaded: ' + most_recent_file)
    
    return prtf_file


#--- Data Cleaning functions
def trim_xtb_xls(prtf_file):
    """
    trims the loaded file of unnecessary stuff - empty rows etc...
    """
    prtf_file = prtf_file.drop(['Unnamed: 0', 'Unnamed: 7'], axis = 1)
    prtf_file = prtf_file.drop([prtf_file.index[-1]], axis = 0)
    return prtf_file

def get_position_from_comment(string):
    """
    Separates volume from comment
    """
    match = re.search(r'(\d+\.?\d*)\s*/?', string)
    if match:
        return match.group(1) 
    return None

def get_splits(yticker, xtb_ticker = None, inception = '2022-01-01' ):
    """
    gets data on stock splits
    """

    ticker = yf.Ticker(yticker)
    ticker_orders_df = pd.DataFrame(ticker.splits)

    if xtb_ticker == None:
        ticker_orders_df['symbol'] = yticker
    else: 
        ticker_orders_df['symbol'] = xtb_ticker

    ticker_orders_df = ticker_orders_df.reset_index()

    ticker_orders_df['Date'] = ticker_orders_df['Date'].dt.date
    ticker_orders_df = ticker_orders_df.loc[ticker_orders_df['Date'] >= inception, :]  # filters only those splits that we need

    ticker_orders_df.rename(columns = {'Date':'date'}, inplace = True)
    ticker_orders_df.rename(columns = {'Stock Splits' : 'split'}, inplace = True)
    return ticker_orders_df



#-- Main function cleaning the orders DF...
def read_orders_df(orders_df):
    """
        Reads the orders_df:
            - changes datatypes
            - extracts info from comment
            - adds additional columns for further processing
            - adjusts stock splits
            - returns adjusted DF

        Depends on:
                get_splits()
                get_position_from_comment()
                trim_xtb_xls()
    """
    try:
        # orders_df = prtf_file_loaded

        orders_df = trim_xtb_xls(orders_df) # drops irrelevant rows
        orders_df = orders_df.rename(columns = {'Type': 'type', 'Comment':'comment', 'Symbol':'symbol', 'Time':'time', 'Amount':'amount'})

        # Change dtypes for faster processing
        orders_df[['type', 'comment', 'symbol']] = orders_df[['type', 'comment', 'symbol']].astype('string')
        orders_df['time'] = pd.to_datetime(orders_df['time'], dayfirst = True)
        orders_df['date'] = orders_df['time'].dt.date
        
        # Extract comment - trade price
        orders_df['trade_price'] = orders_df['comment'].str.split('@', expand = True)[1]
        
        # Determine sell/buy operations
        orders_df.loc[orders_df['comment'].str.contains('BUY'), 'position_type'] = 'buy'
        orders_df.loc[orders_df['comment'].str.contains('CLOSE BUY'), 'position_type'] = 'sell'
        
        # Get volume
        orders_df.loc[orders_df['comment'].str.contains('@'), 'volume'] = orders_df['comment'].apply(get_position_from_comment)
        orders_df['volume'] = pd.to_numeric(orders_df['volume'])

        # Stock splits adjustments
        symbols_inception = orders_df.groupby('symbol', dropna=True)['date'].min()
        yticker = pd.DataFrame.from_dict(settings.TICKERS_DICT, orient='index', columns=['yfinance_ticker', 'currency']).reset_index(names='symbol')
        yticker.drop(columns='currency', inplace=True)
        symbols_inception_yticker = pd.merge(symbols_inception, yticker, on='symbol', how='left')

#---------------------------------------------------------------------------------------------------------------------------------------
# TODO: check the following, for some reason the ticker is not working.
        symbols_inception_yticker = symbols_inception_yticker.loc[symbols_inception_yticker['yfinance_ticker'] != 'OD7F.DE']
        symbols_inception_yticker = symbols_inception_yticker.loc[~symbols_inception_yticker['yfinance_ticker'].isna()]

        # ---------------------------------------------------------------------------------------------------

        splits_history = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns, UTC]'),
                                       'split': pd.Series(dtype='float'),
                                       'symbol': pd.Series(dtype='string')
                                       })

        for row in symbols_inception_yticker.itertuples():
            split_per_ticker = get_splits(row.yfinance_ticker, row.symbol, row.date)
            if not split_per_ticker.empty:
                splits_history = pd.concat([splits_history, split_per_ticker])

        orders_df = pd.merge(orders_df, splits_history, on=['date', 'symbol'], how='outer')
        orders_df.sort_values(by = ['symbol', 'date'], inplace = True)
        orders_df['split'] = (orders_df.groupby('symbol')['split'].bfill())
        orders_df.loc[orders_df['split'].isna(), 'split'] = 1
        orders_df['split'] = orders_df['split'].astype(float)
        orders_df = orders_df.loc[~orders_df['type'].isna(), :] #drops the rows made from splits, that dont have any other data other then the split
        orders_df.loc[:, 'volume'] = orders_df['volume'] * orders_df['split']

        # Fill in NaN values
        orders_df.loc[orders_df['volume'].isna(), 'volume'] = 0
        orders_df.loc[orders_df['volume'] == '', 'volume'] = 0

        # Direction
        orders_df['direction'] = orders_df['volume']
        orders_df.loc[orders_df['type'].str.contains('Stock sale'), 'direction'] = orders_df['direction'] * (-1)
        orders_df.loc[orders_df['direction'].isna(), 'direction'] = 0
        orders_df.loc[orders_df['direction'] == '', 'direction'] = 0

        # Change dtypes for faster processing
        orders_df[['symbol', 'position_type']] = orders_df[['symbol', 'position_type']].astype('string')
        orders_df[['volume', 'direction']] = orders_df[['volume', 'direction']].astype(float)

        logger.info("XTB file was processed successfully")
    
    except Exception as e:
        logger.error(f"XTB file was NOT processed: {e}")

    return orders_df



#--- Prices DF
def download_tickers_prices(tickers_df, history_start, history_end):
    """
        downloads prices for all tickers in the input DF and outputs them in a desired format
    """
    try:
        tickers_to_download = list(tickers_df['yf_ticker'].unique())

        #removes EUREUR just because it temporarily does not work and is not neccessary
        tickers_to_download.remove('EUREUR=X')

        dates_df = pd.DataFrame(index = pd.date_range(history_start, history_end))

        downloaded_prices = yf.download(tickers_to_download, start = history_start, end = history_end, auto_adjust=False)['Adj Close']
        downloaded_prices.index.name = 'date'

        price_series_df = pd.merge(dates_df, downloaded_prices, left_index = True, right_on = 'date', how = 'left')
        price_series_df = price_series_df.set_index('date')
        price_series_df = price_series_df.ffill()

        price_series_df = pd.melt(price_series_df.reset_index(), id_vars = ['date'], value_vars = list(price_series_df.columns), var_name = 'yf_ticker', value_name= 'price' )
        price_series_df = pd.merge(price_series_df, tickers_df, left_on = 'yf_ticker', right_on = 'yf_ticker', how = 'outer' )

        price_series_df = price_series_df.set_index('date')

        logger.info("prices were downloaded successfully")

    except Exception as e:
        logger.error(f"prices were not downloaded: {e}")

    return price_series_df



#--- Ticker Level Data Loader
def create_tickers_df(tickers_dict, fx_dict):
    """
    checks whether the input data has the right format, and then outputs a DF with the static data with tickers
    """

    if not isinstance(tickers_dict, dict):
        logger.warning("first input must be tickers dictionary")
    elif not isinstance(fx_dict, dict):
        logger.warning("second input must be fx dictionary")
        
    # Create the df
    tickers_df = pd.DataFrame(data = tickers_dict).T.reset_index()
    tickers_df = tickers_df.rename(columns = {0 : 'yf_ticker', 1 : 'currency', 'index' : 'symbol'})
    tickers_df.insert(0, 'ticker_type', 'stocks')

    fx_df = pd.DataFrame(data = fx_dict.values(), index = fx_dict.keys()).reset_index()
    fx_df.columns = ['symbol', 'yf_ticker']
    fx_df['currency'] = ''
    fx_df.insert(0, 'ticker_type', 'fx')
    tickers_df = pd.concat([tickers_df, fx_df], axis = 0)

    logger.info("Tickers/FX constants processed successfully")
    
    return tickers_df


def check_constants_exist(tickers_df, orders_df):
    """
    checks that constants exist - if not, they need to be set up in settings.py
    """
    orders_set = set(orders_df['symbol'].dropna())
    constants_set = set(tickers_df['symbol'])

    unassigned_tickers = orders_set.difference(constants_set)

    if len(unassigned_tickers) != 0:
        logger.warning(f'Constants should be updated in setting.py, process will continue but {unassigned_tickers} will not be included')
    pass



def load_financials(datapoints: list, tickers: dict):
    ticker_static_info = pd.DataFrame(columns=datapoints)
    ticker_static_info.insert(0, "ticker", list(tickers.keys()))

    for ticker in tickers:
        for datapoint in datapoints:
            try:
                ticker_static_info.loc[ticker_static_info['ticker'] == ticker, datapoint] = \
                yf.Ticker(tickers[ticker][0]).info[datapoint]
            except:
                ticker_static_info.loc[ticker_static_info['ticker'] == ticker, datapoint] = np.nan
    return ticker_static_info