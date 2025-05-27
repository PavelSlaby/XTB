# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:52:39 2024

@author: pavel
"""

'''
TO DO
Function:
    -- calculate how much PnL would I have made if  invested it all in SP500
    -- what if analysis on price - how much pnl/rtrn if stock reaches certain level

DataSets:
        1] orders_df: -- each cash operation. DF is extended/processes.   
        2] tickers_df: stores static data about each symbol
        3] price_series_df: prices series for each ticker
        4] daily_positions_df: position for each day
        5] pnl_items_df: other items other than the position
        6] pnl_daily_df: daily pnl changes for each ticker
        7] aggregated_stats: most recent stats
    
    
    
    
     
        
Analytics
-- different kinds of PnL - tutal return
-- concentration
-- VaR

'''

#%% Import packages and prepare the environment

import pandas as pd
import yfinance as yf
from datetime import datetime 
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import sys

#graph params
plt.rcParams['figure.figsize'] = [10,5]
plt.rcParams['figure.dpi'] = 100
plt.style.use('seaborn') # seaborn style
mpl.rcParams['font.family'] = 'serif'

#pandas params
pd.set_option('display.max_columns', None) #to display all columns
pd.set_option('display.width', 500)  #use the entire width to display the columns
pd.set_option('display.max_rows', 1000)


# set the folder path
folder_path = r"C:\Users\pslaby\......................\Desktop\osobni\investicni_vypisy"
folder_path2 = r"F:\Investing\XTB"

if  os.path.exists(folder_path): 
    os.chdir(folder_path) 
else: 
    os.chdir(folder_path2)

os.getcwd()


# manual mapping, xtb_ticker to yfinance ticker and crncy
tickers_dict = {
            'CRWD.US' : ['CRWD', 'USD'] , 
            'DTLE.UK' : ['DTLE.L', 'EUR'],
            'ECAR.UK' : ['ECAR.L', 'USD'],
            'GOOGC.US' : ['GOOG', 'USD'],
            'NVDA.US' : ['NVDA', 'USD'],
            'ORSTED.DK' : ['ORSTED.CO', 'DKK'],
            'VWS.DK' :  ['VWS.CO' , 'DKK'],
            'TSLA.US': ['TSLA', 'USD'],
            'IUSA.DE' : ['IUSA.DE', 'EUR'], # TODO: maybe I should change the ticker here
            'XAIX.DE':  ['XAIX.DE', 'EUR'], # TODO: could also change it for (IE00BGV5VN51.SG
            'ENR.DE':  ['ENR.DE', 'EUR'],
            'AMEM.DE':  ['AEEM.PA', 'EUR']
            }

# crncy and respective ticker
fx_dict = {
           'DKK' : 'DKKEUR=X',
           'USD' : 'USDEUR=X',
           'EUR' : 'EUREUR=X'
           }


def check_yfinance_connection():
    try:
        x = yf.Ticker("EURUSD").history(period = '1d')
        if x is not None: print('connection ok') 
        
    except Exception as e:
        print(f"Error: {e} \nCould not connect to Yahoo Finance - exiting the program.")
        #sys.exit(1) # raises the error in Spyder


check_yfinance_connection()


#dat = yf.Ticker("MSFT")
#dat.info

#%% Load source data

def get_most_recent_excel_file(folder_path):
    # Selects the most recently modified file in particular folder
    excel_files = glob.glob(os.path.join(folder_path, '*.xls*'))
    
    if not excel_files:
        return None  # No Excel files found
        
    most_recent_file = max(excel_files, key=os.path.getmtime)
    return most_recent_file


def load_prtf_file(file_name = None, folder_path = None):
    # Loads the selected file from XTB, either explicitly specified or the most recent file in a specific folder
    if file_name == None and folder_path == None:
        print("either file name of folder name need to be specified")
    elif file_name != None:
        prtf_file = pd.read_excel(file_name, sheet_name = 'CASH OPERATION HISTORY', header = 10, index_col = 1 )
    else:
        most_recent_file = get_most_recent_excel_file(folder_path)
        prtf_file = pd.read_excel(most_recent_file, sheet_name = 'CASH OPERATION HISTORY', header = 10, index_col = 1 )
    return prtf_file
    
def trim_prtf_file(prtf_file):
    # trims the loaded file of unneccasry stuff - emptry rows etc...
    prtf_file = prtf_file.drop(['Unnamed: 0', 'Unnamed: 7'], axis = 1)
    prtf_file = prtf_file.drop([prtf_file.index[-1]], axis = 0)
    return prtf_file



#---------
prtf_file = load_prtf_file(None, os.getcwd())
orders_df = trim_prtf_file(prtf_file)





#%% Process the source orders table


# Get volume from comment
def get_position_from_comment(string):
    match = re.search(r'(\d+\.?\d*)\s*/?', string)   # TODO: tohle chece zlepsit at to najde jen /
    if match:
        return match.group(1) 
    return None

#Get data on stock splits
def get_splits(yticker, xtb_ticker = None, hist= '5y'):
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


def read_orders_df(orders_df):
    '''
    Reads the orders_df:
        - changes datatypes
        - extracts info from comment
        - adds additional columns for further processing
        - adjusts stock splits
    '''
    
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
    
    # orders_df.groupby('Symbol').agg({'direction' : 'sum'}).round(2) # Show the volume per ticket
    
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

    return orders_df


orders_df = read_orders_df(orders_df)


#%%  Ticker Level Data




def create_tickers_df(tickers_dict, fx_dict):
    # checks whether the input data has the right format, and then outputs a DF with the static data with tickers
    if not isinstance(tickers_dict, dict):
        print("first input must be tickers dictionary")
    elif not isinstance(fx_dict, dict):
        print("second input must be fx dictionary")
    
    # lets put in more checks here

    # create the df
    tickers_df = pd.DataFrame(data = tickers_dict).T.reset_index()
    tickers_df = tickers_df.rename(columns = {0 : 'yf_ticker', 1 : 'crncy', 'index' : 'Symbol'})
    
    fx_df = pd.DataFrame(data = fx_dict.values(), index = fx_dict.keys()).reset_index()
    fx_df.columns = ['Symbol', 'yf_ticker']
    fx_df['crncy'] = ''
    
    tickers_df = pd.concat([tickers_df, fx_df], axis = 0)
    return tickers_df



tickers_df = create_tickers_df(tickers_dict, fx_dict)



#%% Prices DF
'''
results in a DF with all the prices, downloads prices from yfinance
'''
def get_last_price(ticker):
    return yf.Ticker(ticker).history(period = '1d').Close.values[0] 

history_start = '2023-01-01'
history_end = datetime.today()


def download_tickers_prices(tickers_df):
    # downloads prices for all tickers in the input DF and outputs them in a desired format    
    tickers_to_download = list(tickers_df['yf_ticker'].unique())

    #removes EUREUR just because it temporarily does not work
    tickers_to_download.pop(len(tickers_to_download) - 1)
        
    dates_df = pd.DataFrame(index = pd.date_range(history_start, history_end))
    
    downloaded_prices = yf.download(tickers_to_download, start = history_start, end = history_end, auto_adjust=False)['Adj Close']                        

    price_series_df = pd.merge(dates_df, downloaded_prices, left_index = True, right_on = 'Date', how = 'left')
    price_series_df = price_series_df.set_index('Date')
    price_series_df = price_series_df.fillna(method = 'ffill')
    
    price_series_df = pd.melt(price_series_df.reset_index(), id_vars = ['Date'], value_vars = list(price_series_df.columns), var_name = 'yf_ticker', value_name= 'Price' )
    price_series_df = pd.merge(price_series_df, tickers_df, left_on = 'yf_ticker', right_on = 'yf_ticker', how = 'outer' )
    
    price_series_df = price_series_df.set_index('Date')
    
    return price_series_df


price_series_df = download_tickers_prices(tickers_df)


#%% 
# write some utils file like the follow:
from portfolio_utils import create_tickers_df, ....


class portfolio(orders_df):
    def __init__(self, orders_df):
        self.orders_df = orders_df
        
    def get_stat_data()
        self.tickers_df = create_tickers_df(tickers_dict, fx_dict)
        
    def get_prices()
        self.prices = download_tickers_prices(tickers_df)
        
    def create_position_df():
        positions_df = self.orders_df
        .....


#%% daily position
'''
DF with a daily position [date, symbol, direction, market value_eur]
'''
# source data
positions_df = orders_df.copy()
xtb_symbols = tickers_df.loc[tickers_df['crncy'] != '',   'Symbol']
history_start = '2023-01-01'
history_end = datetime.today()

## Create a DF with ticker per each date
dates_df = pd.date_range(history_start, history_end)
dates_df = pd.MultiIndex.from_product([dates_df, xtb_symbols], names = ['Date', 'Symbol']).to_frame(index = False)

#del positions_df['ID']
del positions_df['Time']
del positions_df['Split']

# Cost = Amount, should we rename to to "invested capital?"
positions_df.loc[positions_df['Type'].isin(['Stock purchase', 'Stock sale']), 'cost'] = positions_df['Amount']


positions_df = pd.merge(positions_df, tickers_df, left_on = 'Symbol', right_on = 'Symbol')

# filters only relevant transasctions -- # TODO: extend this to DIVIDENTs and fees
daily_positions_df = positions_df.loc[positions_df['Type'].isin(['Stock purchase', 'Stock sale']), ['Date', 'Symbol', 'direction', 'crncy', 'cost']] 

# Sum up -- important if there are more orders for the same symbol in one day
daily_positions_df =  daily_positions_df.groupby(['Symbol', 'Date', 'crncy'], as_index = False).agg({'direction': 'sum', 'cost': 'sum'})

# Forward fill for each date
daily_positions_df['Date']  = daily_positions_df['Date'].astype('datetime64[ns]') 
daily_positions_df['outstanding_position'] = daily_positions_df.groupby(['Symbol'])['direction'].cumsum()

daily_positions_df['cost']  = daily_positions_df['cost'].astype('float64')

daily_positions_df['cost_cumsum'] = daily_positions_df.groupby(['Symbol'])['cost'].cumsum()


daily_positions_df = pd.merge(dates_df, daily_positions_df, left_on = ['Date', 'Symbol'], right_on = ['Date', 'Symbol'] , how = 'outer')
daily_positions_df.loc[daily_positions_df['direction'].isna() == True,  'direction'] = 0
daily_positions_df = daily_positions_df.sort_values(by = 'Date')
daily_positions_df['outstanding_position']  =  daily_positions_df.groupby('Symbol')['outstanding_position'].ffill()
daily_positions_df['cost_cumsum']  =  daily_positions_df.groupby('Symbol')['cost_cumsum'].ffill()
daily_positions_df['crncy']  =  daily_positions_df.groupby('Symbol')['crncy'].ffill()
daily_positions_df.loc[daily_positions_df['outstanding_position'].isna() == True,  'outstanding_position'] = 0 # replace nan with 0, but maybe I should rather drop the rows...


#get the daily prices and fx
daily_positions_df = pd.merge(daily_positions_df, price_series_df[['Symbol', 'Price']].reset_index(), left_on = ['Date', 'Symbol'], right_on = ['Date', 'Symbol'] , how = 'left')
daily_positions_df = pd.merge(daily_positions_df, price_series_df[['Symbol', 'Price']].reset_index(), left_on = ['Date', 'crncy'], right_on = ['Date', 'Symbol'] , how = 'left')
del daily_positions_df['Symbol_y']
daily_positions_df.rename(columns = {'Symbol_x' : 'Symbol', 'Price_x': 'Price', 'Price_y': 'fx'}, inplace = True)
daily_positions_df.loc[daily_positions_df['crncy'] == 'EUR', 'fx'] = 1 
daily_positions_df.loc[daily_positions_df['cost'].isna() == True, 'cost'] = 0


## Calculate the metrics
#MV
daily_positions_df['MV'] = daily_positions_df['Price'] * daily_positions_df['outstanding_position'] * daily_positions_df['fx']

#PNL
daily_positions_df['pnl_ltd'] = daily_positions_df['MV'] + daily_positions_df['cost_cumsum']

for i in xtb_symbols:
    daily_positions_df.loc[daily_positions_df['Symbol'] == i, 'pnl_dtd'] = (daily_positions_df.loc[daily_positions_df['Symbol'] == i, 'pnl_ltd'] - daily_positions_df.loc[daily_positions_df['Symbol'] == i, 'pnl_ltd'].shift(1))

daily_positions_df.loc[daily_positions_df['pnl_dtd'].isna(), 'pnl_dtd' ] = daily_positions_df['pnl_ltd']

daily_positions_df = daily_positions_df.loc[(daily_positions_df['outstanding_position'] != 0) | (daily_positions_df['direction'] != 0)]


daily_positions_df.head(10)

#%% Other Orders - grouped
 
pnl_items_df = orders_df.loc[orders_df['Type'].isin(['DIVIDENT', 'Withholding tax', 'SEC fee']) , ['Date', 'Symbol', 'Type', 'Comment', 'Amount', ] ].copy()
pnl_items_df = pnl_items_df.groupby(['Date', 'Symbol', 'Type', 'Comment'])['Amount'].sum().reset_index()

pnl_items_df.head(20)


#%% PnL Table

#merge position and other pnl data
pnl_items_df_agg = pnl_items_df.groupby(['Date', 'Symbol'])['Amount'].sum().reset_index()
pnl_items_df_agg['Date'] = pd.to_datetime(pnl_items_df_agg['Date'])

pnl_daily_df = pd.merge(daily_positions_df, pnl_items_df_agg, left_on = ['Date', 'Symbol'], right_on = ['Date', 'Symbol'], how = 'left' )

pnl_daily_df = pnl_daily_df.loc[(pnl_daily_df['outstanding_position'] != 0) | (pnl_daily_df['direction'] != 0)]
pnl_daily_df.loc[pnl_daily_df['Amount'].isna(), 'Amount' ] = 0
pnl_daily_df.loc[pnl_daily_df['pnl_dtd'].isna(), 'pnl_dtd' ] = 0


pnl_daily_df = pnl_daily_df.rename(columns = {'Amount': 'other_pnl'})

#Calculate metrics 
#PNL
pnl_daily_df['pnl_tot_dtd'] = pnl_daily_df['pnl_dtd'] + pnl_daily_df['other_pnl']
pnl_daily_df['pnl_tot_ltd'] = pnl_daily_df.groupby(['Symbol'])['pnl_tot_dtd'].cumsum()
pnl_daily_df['pnl_ltd'] = pnl_daily_df.groupby(['Symbol'])['pnl_dtd'].cumsum()
pnl_daily_df['pnl_rel_ltd'] = pnl_daily_df['pnl_ltd'] / pnl_daily_df['cost_cumsum'] *-1
pnl_daily_df['pnl_rel_tot_ltd'] = pnl_daily_df['pnl_tot_ltd'] / pnl_daily_df['cost_cumsum'] *-1
pnl_daily_df['pnl_rel_dtd'] = pnl_daily_df['pnl_dtd'] / (pnl_daily_df['MV']   - pnl_daily_df['pnl_dtd'])



# Daily Total Portfolio Values
total_position = pnl_daily_df[['Date',  'Symbol', 'pnl_tot_ltd', 'pnl_ltd', 'MV', 'cost_cumsum', 'cost',  'pnl_tot_dtd', 'pnl_rel_dtd']].pivot(index='Date', columns='Symbol', values=[ 'pnl_tot_ltd', 'pnl_rel_dtd', 'MV', 'cost', 'cost_cumsum', 'pnl_ltd', 'pnl_tot_dtd'])
total_position['prtf_pnl_tot_ltd'] = total_position['pnl_tot_ltd'].sum(axis = 1)
total_position['prtf_pnl_ltd'] = total_position['pnl_ltd'].sum(axis = 1)
total_position['prtf_mv'] = total_position['MV'].sum(axis = 1)
total_position['prtf_cost_sum'] = total_position['cost_cumsum'].sum(axis = 1) * -1
total_position['prtf_tot_rtn_ltd'] = total_position['prtf_pnl_tot_ltd'] / total_position['prtf_cost_sum'] 
total_position['prtf_rtn_ltd'] = total_position['prtf_pnl_ltd'] / total_position['prtf_cost_sum'] 
total_position['prtf_cost_dtd'] = total_position['cost'].sum(axis = 1)  * -1
#total_position['unit_rtn_tot'] = 1 + total_position['prtf_tot_rtn_ltd']
#total_position['unit_rtn_tot_dtd'] = total_position['unit_rtn_tot'] / total_position['unit_rtn_tot'].shift(1) -1



bmk_price_series = price_series_df.loc[price_series_df['Symbol']== 'IUSA.DE', 'Price']

# Convert to two-level DataFrame
bmk_price_series = pd.DataFrame(bmk_price_series)
bmk_price_series.columns = pd.MultiIndex.from_tuples([('Price', 'bmk')])


bmk_price_series.head()

total_position = pd.merge(total_position, bmk_price_series, how = 'left', left_index = True, right_on = 'Date' )


total_position['Price'] = total_position['Price'] / total_position.iloc[0, total_position.columns.get_loc('Price')][0]






#NAV
total_position['Total_Units'] = 1
total_position['NAV'] = 1
    

for i in range(len(total_position['Total_Units'])):
    if i == 0: ## probably faster if I get rif of the IF and just do it beforehad, so that the if conditiona does not hvae to eb evaluated each time
        total_position.iloc[i, total_position.columns.get_loc('Total_Units')] =  total_position.iloc[i, total_position.columns.get_loc('prtf_cost_dtd')]
        total_position.iloc[i, total_position.columns.get_loc('NAV')] =  total_position.iloc[i, total_position.columns.get_loc('prtf_mv')][0] / total_position.iloc[i, total_position.columns.get_loc('Total_Units')][0]

    else:
        total_position.iloc[i, total_position.columns.get_loc('Total_Units')] = ( 
                                                                                 total_position.iloc[i - 1, total_position.columns.get_loc('Total_Units')][0]
                                                                                 + 
                                                                                 total_position.iloc[i, total_position.columns.get_loc('prtf_cost_dtd')][0] 
                                                                                                                                                                
                                                                                 /total_position.iloc[i - 1, total_position.columns.get_loc('NAV')][0]
                                                                                 )
              
        total_position.iloc[i, total_position.columns.get_loc('NAV')] = total_position.iloc[i, total_position.columns.get_loc('prtf_mv')][0] / total_position.iloc[i, total_position.columns.get_loc('Total_Units')][0]
        
        
        
        
total_position['unit_rtn_tot_dtd'] = total_position['NAV'] / total_position['NAV'].shift(1) -1

#%% Other Statistics

risk_free_rate = 0.03

#1Y Sharpe ratio
total_position['1Y_rtn'] = total_position["unit_rtn_tot_dtd"].rolling(window=365).apply(lambda x: (x +1).prod(), raw=True) - 1 
total_position['1Y_excess_rtn'] = total_position['1Y_rtn'] - risk_free_rate
total_position['1Y_excess_std_dev'] = total_position['1Y_excess_rtn'].rolling(window=365).apply(lambda x: x.std(), raw=True)
total_position['1Y_sharpe'] = total_position['1Y_excess_rtn']  / total_position['1Y_excess_std_dev'] 

      
#Biggest daily loss
biggest_daily_loss_index = total_position.index[total_position['unit_rtn_tot_dtd'] == min(total_position['unit_rtn_tot_dtd'].fillna(0))]
prev_day = biggest_daily_loss_index - pd.Timedelta(days = 1)

loss = total_position.loc[total_position.index == biggest_daily_loss_index[0], 'prtf_mv'][0]   - total_position.loc[total_position.index == prev_day[0], 'prtf_mv'][0]

print("Biggest daily loss was: " + str(round(loss, 0)) + " on: " + str(biggest_daily_loss_index[0]))


#Biggest daily gain
biggest_daily_loss_index = total_position.index[total_position['unit_rtn_tot_dtd'] == max(total_position['unit_rtn_tot_dtd'].fillna(0))]
prev_day = biggest_daily_loss_index - pd.Timedelta(days = 1)

loss = total_position.loc[total_position.index == biggest_daily_loss_index[0], 'prtf_mv'][0]   - total_position.loc[total_position.index == prev_day[0], 'prtf_mv'][0]

print("Biggest daily gain was: " + str(round(loss, 0)) + " on: " + str(biggest_daily_loss_index[0]))



#Maximum Drawdown
max_drawdown = 1
for i in total_position.index[1:]:
    crnt_value = total_position.loc[total_position.index == i, 'NAV'][0]
    prev_peak = total_position.loc[total_position.index < i, 'NAV'].max()
    prev_peak_date = total_position.loc[total_position.index < i, 'NAV'].idxmax()
    
    drawdown = crnt_value / prev_peak 
    drawdown_abs = total_position.loc[total_position.index == i, 'prtf_mv'][0] - total_position.loc[total_position.index == prev_peak_date, 'prtf_mv'][0]
    drawdown_abs_adj = sum(total_position.loc[(total_position.index >= prev_peak_date) & (total_position.index < i), 'prtf_cost_dtd'])
    
    crnt_value - prev_peak
    if drawdown <= max_drawdown:
        max_drawdown = drawdown 
        max_drawdown_abs = drawdown_abs - drawdown_abs_adj
        peak_date = prev_peak_date
        through_date = i
       

print("Max drawdown was: " + str(round(max_drawdown -1 , 2)) + "% (" + str(round(max_drawdown_abs, 0))   + " EUR ) from: " + str(peak_date) + " to " + str(through_date))


#%% VaR 



var_date = pnl_daily_df.iloc[-1]['Date']


crnt_mv =    pnl_daily_df.loc[pnl_daily_df['Date'] == var_date, 'MV'].sum()          


weights = pnl_daily_df.loc[pnl_daily_df['Date'] == var_date, ['Symbol', 'MV']]
weights['weight'] = weights['MV'] / crnt_mv

var = price_series_df.loc[price_series_df['Symbol'].isin(weights['Symbol'].tolist()), ['Price', 'Symbol']].reset_index()

var = pd.merge(var, weights[['Symbol', 'weight']], left_on = 'Symbol', right_on = 'Symbol', how = 'left' )

var['prtf_mv'] = var['weight'] = var['Price']

var = var.dropna()

returns = pd.DataFrame()
returns['nav'] =  var.groupby('Date').agg({'prtf_mv' : 'sum'})
returns['rtn'] = returns['nav'] / returns['nav'].shift(1) - 1

var = np.percentile(returns['rtn'][1:], 5) #first value is nan, thats why the slice

var_abs = crnt_mv * var

#lets do some backtesting....

prtf_rtns = total_position['unit_rtn_tot_dtd'] - 1

sum(prtf_rtns < var)
prtf_rtns.shape[0]

backtest = sum(prtf_rtns < var) / prtf_rtns.shape[0]


print(var)
print(backtest)


#%% Other under construction

total_position.to_excel('output.xlsx') 



#%% Portfolio Most Recent Overview
aggregated_stats = round(pnl_daily_df.groupby('Symbol').agg({
                                                        'pnl_tot_ltd': 'last',
                                                        'pnl_rel_tot_ltd': 'last', 
                                                        'pnl_dtd': 'last', 
                                                        'outstanding_position': 'last', 
                                                        'Price': 'last', 
                                                        'MV': 'last',  
                                                        'cost_cumsum': 'last'  
                                                        }), 3 )


aggregated_stats['MV_%'] = aggregated_stats['MV'] / sum(aggregated_stats['MV']) * 100

#Statistics:
prtf_tot_rtn_ltd =  round(total_position['prtf_tot_rtn_ltd'][-1] * 100, 4)
nav_per_share = round(total_position['NAV'].tail(1)[0], 4)
prtf_pnl_tot_ltd = round(total_position['prtf_pnl_tot_ltd'].tail(1)[0], 0)
prtf_mv =  round(total_position['prtf_mv'].tail(1)[0], 0)
invested_own_funds = round(total_position['prtf_cost_sum'].tail(1)[0], 0)
prtf_tot_rtn_1y = round(total_position['1Y_rtn'][-1] * 100, 4)
prtf_sharpe_1y = round(total_position['1Y_sharpe'].tail(1)[0], 4)



print("Total Portfolio Return LTD is: " + str(prtf_tot_rtn_ltd) + "%" )
print("Total Portfolio Return 1Y is: " + str(prtf_tot_rtn_1y) + "%" )

print("NAV per share is: " + str(nav_per_share) )
print("LTD Total PnL is: " + str(prtf_pnl_tot_ltd) )
print("Sharpe 1Y is: " + str(prtf_sharpe_1y) )


print("MV portfolio is: " + str(prtf_mv) )
print("Currently invested own funds: " + str(invested_own_funds) )



print(aggregated_stats)


#%% How much would the pnl be if I invested in X?
# assuming I would invest exactly the same amount of money I did at the same times I did, just in a different ticker...

benchmark_symbol = 'IUSA.DE'

def simulate_bmk_rtn(benchmark_symbol):  
    invested_amount = daily_positions_df[['Date', 'cost']].groupby('Date').sum('cost')
    benchmark_price_series_df = price_series_df.loc[price_series_df['Symbol'] == benchmark_symbol, ['Price']]
    
    invested_amount_price =  pd.merge(invested_amount, benchmark_price_series_df, how = 'left', left_on = 'Date', right_on = 'Date')
    invested_amount_price.loc[invested_amount_price['cost'] == 0, 'cost' ] =  0 #np.nan 
    invested_amount_price['direction'] = invested_amount_price['cost'] * -1 / invested_amount_price['Price'] 
    invested_amount_price['cost_cumsum'] =  invested_amount_price['cost'].cumsum() * -1
    invested_amount_price['direction_cumsum'] =  invested_amount_price['direction'].cumsum()
    invested_amount_price['MV'] = invested_amount_price['direction_cumsum']  * invested_amount_price['Price'] 
    invested_amount_price['Total_Rel_Rtn'] = invested_amount_price['MV'] / invested_amount_price['cost_cumsum'] -1
    
    # graph
    x_axis = total_position.index 
    y_axis1 = total_position['prtf_pnl_ltd'] 
    y_axis2 = total_position['prtf_mv'] 
    y_axis3 = total_position['prtf_cost_sum'] 
    y_axis4 = total_position['prtf_tot_rtn_ltd'] 
    y_axis5 = invested_amount_price['MV'] 
    y_axis6 = invested_amount_price['Total_Rel_Rtn']
    
    fig, ax1 = plt.subplots()
    ax1.plot(x_axis  , y_axis1, label = 'PnL' )
    ax1.plot(x_axis  , y_axis2, label = 'MV' )
    ax1.plot(x_axis  , y_axis3, label = 'Invested Capital' )
    ax1.plot(x_axis  , y_axis5, label = benchmark_symbol )
    
    ax1.set_ylabel('EUR')
    
    ax2 = ax1.twinx()
    ax2.plot(x_axis  , y_axis4, color = 'orange', label = '% Return' )
    ax2.plot(x_axis  , y_axis6, color = 'red', label = str(benchmark_symbol) + ' return'  )
    
    ax2.set_ylabel('% Rtn', color = 'orange')
    ax2.tick_params(labelcolor = 'orange')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    prtf_tot_rtn_last  = total_position['prtf_tot_rtn_ltd'].tail(1)[0] * 100
    prtf_rtn_last  = total_position['prtf_rtn_ltd'].tail(1)[0] * 100
    Total_Rel_Rtn_last = invested_amount_price['Total_Rel_Rtn'].tail(1)[0]  * 100
    
    print("Benchmark symbol is: " + benchmark_symbol)
    print("note that this simulation does not consider DIVIDENTs....")
    print("Portfolio return was: " + str(prtf_rtn_last.round(3)) + "%")
    print("Portfolio total return was: " + str(prtf_tot_rtn_last.round(3)) + "%")
    print("Simulated benchmark return was: " + str(Total_Rel_Rtn_last.round(3)) + "%")
    print("Excess non-DIVIDENT return was: " + str((prtf_rtn_last - Total_Rel_Rtn_last).round(3)) + "%")


# for i in  xtb_symbols_list:
#     print(i)
#     simulate_bmk_rtn(i)
#     print(" ")


simulate_bmk_rtn('IUSA.DE')

#%% Matplotlib

def plot_mv(xtb_symbol):
    daily_positions_df.loc[daily_positions_df['Symbol'] == xtb_symbol, ['Symbol', 'MV'] ].plot()



plot_mv('ENR.DE')


## 1]  Show MV of each symbol in the same graph
for i in list(pnl_daily_df['Symbol'].unique()): 
    x_axis = pnl_daily_df.loc[pnl_daily_df['Symbol'] == i, 'Date']
    y_axis = pnl_daily_df.loc[pnl_daily_df['Symbol'] == i, 'MV']

    plt.plot(x_axis  , y_axis, label = i )
    plt.title('MV Growth') 
    plt.legend(loc = 0)
    plt.savefig('MV growth.png')  
    
   
## 2]  Show MV of each symbol seperately
for i in list(pnl_daily_df['Symbol'].unique()): 
    x_axis = pnl_daily_df.loc[pnl_daily_df['Symbol'] == i, 'Date']
    y_axis = pnl_daily_df.loc[pnl_daily_df['Symbol'] == i, 'MV']

    plt.plot(x_axis  , y_axis, label = i )
    plt.title(i) 
    plt.legend(loc = 0)
    plt.show()


## 3 Show MV of each symbol in the same graph Stacked
pivoted_df = pnl_daily_df[['Date',  'Symbol', 'MV']].pivot(index='Date', columns='Symbol', values='MV')
pivoted_df.fillna(0, inplace=True)

# sort it by when I invested in it....
sorting_list = []
for i in pivoted_df.columns:
    sorting_list.append( [i , pivoted_df.loc[:, i].loc[pivoted_df.loc[:, i] != 0].index[0] ])

sorting_list = sorted(sorting_list , key = lambda item: item[1])
column_order = [i[0] for i in sorting_list]
    
fig, ax = plt.subplots()
ax.stackplot(pivoted_df.index, pivoted_df[column_order].T.values, labels=pivoted_df[column_order].columns )
ax.set_title('Market Value Growth')
ax.set_ylabel('EUR')
plt.legend(loc = 2)
plt.show()



   
  

#%% -- INTEREST
interest_orders_df = orders_df.loc[orders_df['Type'].isin(['Free funds interests tax', 'Free funds interests'])]
interest_orders_df[interest_orders_df.Time >= '2024-01-01']['Amount'].sum()



#%% Run daily report

print("Total Portfolio Return LTD is: " + str(portoflio_ltd_tot_rtn) + "%" )
aggregated_stats


