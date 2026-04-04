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

    # Data check
    if orders_df is None or orders_df.empty or tickers_df is None or tickers_df.empty:
        print("orders_df and tickers_df dataframes were not specified or are empty")

    # Set the end of the timeframe for the calculations, default value = today
    if history_end == None: history_end = datetime.today()

    # Filters only stock tickers, the rest is for currency pairs
    xtb_symbols = tickers_df.loc[tickers_df['ticker_type'] == 'stocks', 'symbol']

    date_range = pd.date_range(history_start, history_end)
    dates_per_ticker = pd.MultiIndex.from_product([date_range, xtb_symbols], names=['date', 'symbol']).to_frame(index=False)

    # Modify the orders DF
    orders_modified = orders_df.copy()
    del orders_modified['Time']
    del orders_modified['Split']

    orders_modified = pd.merge(orders_modified, tickers_df, left_on = 'symbol', right_on = 'symbol')

    # filters only transactions for constructing the position
    daily_positions_df = orders_modified.loc[orders_modified['Type'].isin(['Stock purchase', 'Stock sale']), ['date', 'symbol', 'direction', 'currency', 'Amount']]

    # Sum up -- important if there are more orders for the same symbol in one day, it aggregates them per day/symbol
    daily_positions_df =  daily_positions_df.groupby(['symbol', 'date', 'currency'], as_index = False).agg({'direction': 'sum', 'Amount': 'sum'})

    daily_positions_df['date']  = daily_positions_df['date'].astype('datetime64[ns]')
    daily_positions_df['amount'] = daily_positions_df['Amount'].astype('float64')


    daily_positions_df['outstanding_position'] = daily_positions_df.groupby(['symbol'])['direction'].cumsum()
    daily_positions_df['cost_cumsum'] = daily_positions_df.groupby(['symbol'])['cost'].cumsum()


    # Forward fill for each date
    daily_positions_df = pd.merge(dates_per_ticker, daily_positions_df, left_on = ['date', 'symbol'], right_on = ['date', 'symbol'] , how = 'outer')
    daily_positions_df.loc[daily_positions_df['direction'].isna() == True,  'direction'] = 0
    daily_positions_df = daily_positions_df.sort_values(by = 'date')
    daily_positions_df['outstanding_position']  =  daily_positions_df.groupby('symbol')['outstanding_position'].ffill()
    daily_positions_df['cost_cumsum']  =  daily_positions_df.groupby('symbol')['cost_cumsum'].ffill()
    daily_positions_df['currency']  =  daily_positions_df.groupby('symbol')['currency'].ffill()

    # drop rows where there is no outstanding position
    daily_positions_df =  daily_positions_df.loc[daily_positions_df['outstanding_position'].isna() == False]


    #get the daily prices and fx
    daily_positions_df = pd.merge(daily_positions_df, price_series_df[['symbol', 'Price']].reset_index(), left_on = ['date', 'symbol'], right_on = ['date', 'symbol'] , how = 'left')
    daily_positions_df = pd.merge(daily_positions_df, price_series_df[['symbol', 'Price']].reset_index(), left_on = ['date', 'currency'], right_on = ['date', 'symbol'] , how = 'left')
    del daily_positions_df['symbol_y']
    daily_positions_df.rename(columns = {'symbol_x' : 'symbol', 'Price_x': 'Price', 'Price_y': 'fx'}, inplace = True)
    daily_positions_df.loc[daily_positions_df['currency'] == 'EUR', 'fx'] = 1
    daily_positions_df.loc[daily_positions_df['cost'].isna() == True, 'cost'] = 0
    
    logger.info("daily position table updated")

    return daily_positions_df