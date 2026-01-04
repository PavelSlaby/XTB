'''
data_loader.py

This module contains function to load data (CSV, yfinance) and to clean them

TODO:
    generalize the splits, so that it works not only for nvidia, but also chargepoint etc...
    create tools to export data or to even store in a DB
'''

#-- import of packages
import glob
import os
import pandas as pd
import re
import yfinance as yf
import logging
import sys
import library.settings  as settings

logger = logging.getLogger(__name__)

source = settings.XTB_INPUT_FILEPATH


#--- Checks ---
def check_yfinance_connection():
    '''
    Checks whether connection to yahoo finance can be established
    '''
    try:
        x = yf.Ticker("AAPL").history(period = '1d')
        if x is not None: logger.info("connection to Yahoo Finance is successfull") 
               
    except Exception as e:
        logger.warning(f"Error: {e} \nCould not connect to Yahoo Finance")
        #sys.exit(1) # raises the error in Spyder

#--- CSV file loader helper functions
def get_most_recent_excel_file(folder_path):
    ''' 
    Selects the most recently modified file in particular folder
    '''
    excel_files = glob.glob(os.path.join(folder_path, '*.xls*'))
    
    if not excel_files:
        return None  # No Excel files found
        
    most_recent_file = max(excel_files, key=os.path.getmtime)
    return most_recent_file

#--- CSV File loader
def load_prtf_file(file_name = None, folder_path = None):
    ''' 
    Loads the selected file from XTB, either explicitly specified or 
    the most recent file in a specific folder, for which it uses the get_most_recent_excel_file function
    '''

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
    '''
    trims the loaded file of unnecessary stuff - empty rows etc...
    '''
    prtf_file = prtf_file.drop(['Unnamed: 0', 'Unnamed: 7'], axis = 1)
    prtf_file = prtf_file.drop([prtf_file.index[-1]], axis = 0)
    return prtf_file

def get_position_from_comment(string):
    '''
    Separates volume from comment
    '''
    match = re.search(r'(\d+\.?\d*)\s*/?', string)
    if match:
        return match.group(1) 
    return None

def get_splits(yticker, xtb_ticker = None, hist= '5y'):
    '''
    gets data on stock splits
    '''
    ticker = yf.Ticker(yticker)
    ticker.history(period = hist)
    ticker_orders_df = pd.DataFrame(ticker.splits)
    if xtb_ticker == None:
        ticker_orders_df['Symbol'] = yticker
    else: 
        ticker_orders_df['Symbol'] = xtb_ticker
    ticker_orders_df = ticker_orders_df.reset_index()
    ticker_orders_df['Date'] = ticker_orders_df['Date'].dt.date
    ticker_orders_df.rename(columns = {'Stock Splits' : 'Split'}, inplace = True)
    return ticker_orders_df

#-- Main function cleaning the orders DF...
def read_orders_df(orders_df):
    '''
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
    '''
    try: 
        orders_df = trim_xtb_xls(orders_df)
    
        # Change dtypes for faster processing
        orders_df[['Type', 'Comment', 'Symbol']] = orders_df[['Type', 'Comment', 'Symbol']].astype('string')
        orders_df['Time'] = pd.to_datetime(orders_df['Time'], dayfirst = True)
        orders_df['Date'] = orders_df['Time'].dt.date
        
        # Extract comment - trade price
        orders_df['trade_price'] = orders_df['Comment'].str.split('@', expand = True)[1]
        
        # Determine sell/buy operations
        orders_df.loc[orders_df['Comment'].str.contains('BUY'), 'position_type'] = 'buy'
        orders_df.loc[orders_df['Comment'].str.contains('CLOSE BUY'), 'position_type'] = 'sell'
        
        #Get volume
        orders_df.loc[orders_df['Comment'].str.contains('@'), 'volume'] = orders_df['Comment'].apply(get_position_from_comment)
        orders_df['volume'] = pd.to_numeric(orders_df['volume'])
        
        # Get Direction
        orders_df['direction'] = orders_df['volume']
        orders_df.loc[orders_df['Type'].str.contains('Stock sale') , 'direction'] = orders_df['direction'] * (-1)
        orders_df.loc[orders_df['direction'] == '', 'direction'] = 0
        orders_df['direction'] = orders_df['direction'].astype(float)
        orders_df['volume'] = orders_df['volume'].astype(float)
            
        # Stock splits adjustments 
        list_shares_w_splits = [['NVDA', 'NVDA.US']]
        split_orders_df = pd.DataFrame()
        
        for i in list_shares_w_splits:
            split_orders_df = pd.concat( [split_orders_df,  get_splits(i[0], i[1]) ], ignore_index =True )
           
        orders_df = pd.merge(orders_df, split_orders_df, on=['Date', 'Symbol'], how='outer')
        orders_df.sort_values(by = ['Symbol', 'Date'], inplace = True)
        
        for i in list_shares_w_splits:
            orders_df.loc[orders_df['Symbol'] == i[1], ['Split']] = orders_df['Split'].bfill()
        
        orders_df.loc[orders_df['Split'].isna(), 'Split'] = 1
        
        orders_df['Split'] = orders_df['Split'].astype(float)
        
        orders_df.loc[:, 'volume'] = orders_df['volume'] * orders_df['Split']
        orders_df.loc[:, 'direction']  = orders_df['direction'] * orders_df['Split']
        
        # fill in NaN values
        orders_df.loc[orders_df['volume'].isna(), 'volume'] = 0
        orders_df.loc[orders_df['volume'] == '', 'volume'] = 0
        orders_df.loc[orders_df['direction'].isna(), 'direction'] = 0
        orders_df.loc[orders_df['direction'] == '', 'direction'] = 0
        orders_df.loc[orders_df['Type'].isna(), 'Type'] = 'Split'
        
        # drop the actuall split dates, which do not matter anymore
        orders_df.drop(orders_df.loc[orders_df['Type'] == 'Split' ].index, inplace = True)
        
        orders_df.loc[orders_df['Type'].str.contains('Stock sale'), 'direction' ] = orders_df['volume' ].astype(float) * (-1)
        
        #change dtypes for faster processing
        orders_df[['Symbol', 'position_type']] = orders_df[['Symbol', 'position_type']].astype('string')
    
        logger.info("XTB file was processed successfully")
    
    except Exception as e:
        logger.error(f"XTB file was NOT processed: {e}")

    return orders_df



#--- Prices DF
def download_tickers_prices(tickers_df, history_start, history_end):
    '''
        downloads prices for all tickers in the input DF and outputs them in a desired format
    '''
    try:
        tickers_to_download = list(tickers_df['yf_ticker'].unique())

        #removes EUREUR just because it temporarily does not work
        tickers_to_download.pop(len(tickers_to_download) - 1)


        dates_df = pd.DataFrame(index = pd.date_range(history_start, history_end))

        downloaded_prices = yf.download(tickers_to_download, start = history_start, end = history_end, auto_adjust=False)['Adj Close']

        price_series_df = pd.merge(dates_df, downloaded_prices, left_index = True, right_on = 'Date', how = 'left')
        price_series_df = price_series_df.set_index('Date')
        price_series_df = price_series_df.ffill()

        price_series_df = pd.melt(price_series_df.reset_index(), id_vars = ['Date'], value_vars = list(price_series_df.columns), var_name = 'yf_ticker', value_name= 'Price' )
        price_series_df = pd.merge(price_series_df, tickers_df, left_on = 'yf_ticker', right_on = 'yf_ticker', how = 'outer' )

        price_series_df = price_series_df.set_index('Date')

        logger.info("prices were downloaded successfully")

    except Exception as e:
        logger.error(f"prices were not downloaded: {e}")

    return price_series_df



#--- Ticker Level Data Loader
def create_tickers_df(tickers_dict, fx_dict):
    '''
    checks whether the input data has the right format, and then outputs a DF with the static data with tickers
    '''

    if not isinstance(tickers_dict, dict):
        logger.warning("first input must be tickers dictionary")
    elif not isinstance(fx_dict, dict):
        logger.warning("second input must be fx dictionary")
        
    # lets put in more checks here

    # create the df
    tickers_df = pd.DataFrame(data = tickers_dict).T.reset_index()
    tickers_df = tickers_df.rename(columns = {0 : 'yf_ticker', 1 : 'crncy', 'index' : 'Symbol'})
    
    fx_df = pd.DataFrame(data = fx_dict.values(), index = fx_dict.keys()).reset_index()
    fx_df.columns = ['Symbol', 'yf_ticker']
    fx_df['crncy'] = ''
    
    tickers_df = pd.concat([tickers_df, fx_df], axis = 0)
    
    logger.info("Tickers/FX constants processed successfully")
    
    return tickers_df



def check_constants_exist(tickers_df, orders_df):
    '''
    checks that constants exist - if not, they need to be set up in settings.py
    '''
    orders_set = set(orders_df['Symbol'].dropna())
    constants_set = set(tickers_df['Symbol'])

    unassigned_tickers = orders_set.difference(constants_set)

    if len(unassigned_tickers) != 0:
        logger.warning(f'Constants should be updated in setting.py, process will continue but {unassigned_tickers} will not be included')
    pass
