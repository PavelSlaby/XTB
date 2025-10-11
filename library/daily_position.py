'''
daily_position.py

This module contains function to create daily position table

functions:

todo: 
- the final table contains rows with nulls, see if it can contain only useful rows    

dependencies: XTB csv file has to be processed into orders_df dataframe    

'''
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_position_df(orders_df, tickers_df, price_series_df, history_start, history_end = None):
    '''    
        Create a DF with ticker per each date
    ''' 
    if orders_df is None or orders_df.empty: 
        print("orders_df and tickers_df dataframes were not specified or are empty")

    positions_df = orders_df.copy()

    del positions_df['Time']
    del positions_df['Split']
   
    if history_end == None: history_end = datetime.today()
    
    xtb_symbols = tickers_df.loc[tickers_df['crncy'] != '',   'Symbol']
    
    dates_df = pd.date_range(history_start, history_end)
    dates_df = pd.MultiIndex.from_product([dates_df, xtb_symbols], names = ['Date', 'Symbol']).to_frame(index = False)

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
    
    logger.info("daily position table updated")

    return daily_positions_df