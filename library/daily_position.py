"""
This module contains function to create daily position table
dependencies: XTB csv file has to be processed into orders_df dataframe

"""

from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_position_df(orders_df, tickers_df, price_series_df, history_start, history_end = None):
    """
        Create a DF with ticker per each date
    """

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
    orders_modified = pd.merge(orders_modified, tickers_df, left_on = 'symbol', right_on = 'symbol', how= 'left')

    # Filter only transactions for constructing the position
    daily_positions_df = orders_modified.loc[orders_modified['type'].isin(['Stock purchase', 'Stock sale']), ['date', 'symbol', 'direction', 'currency', 'amount']]

    # Sum up order per day -- important if there are more orders for the same symbol in one day, it aggregates them per day/symbol
    daily_positions_df =  daily_positions_df.groupby(['symbol', 'date', 'currency'], as_index = False).agg({'direction': 'sum', 'amount': 'sum'})

    daily_positions_df['date']  = daily_positions_df['date'].astype('datetime64[ns]')
    daily_positions_df['amount'] = daily_positions_df['amount'].astype('float64')

    daily_positions_df['outstanding_position'] = daily_positions_df.groupby(['symbol'])['direction'].cumsum()
    daily_positions_df['cost_cumsum'] = daily_positions_df.groupby(['symbol'])['amount'].cumsum()

    # Forward fill for each date
    daily_positions_df = pd.merge(dates_per_ticker, daily_positions_df, left_on = ['date', 'symbol'], right_on = ['date', 'symbol'] , how = 'outer')
    #daily_positions_df.loc[daily_positions_df['direction'].isna() == True,  'direction'] = 0
    daily_positions_df = daily_positions_df.sort_values(by = 'date')
    daily_positions_df['outstanding_position']  =  daily_positions_df.groupby('symbol')['outstanding_position'].ffill()
    daily_positions_df['cost_cumsum']  =  daily_positions_df.groupby('symbol')['cost_cumsum'].ffill()
    daily_positions_df['currency']  =  daily_positions_df.groupby('symbol')['currency'].ffill()

    # Drop rows where there is no outstanding position
    filtered_out = (
        ((daily_positions_df['outstanding_position'].isna() == True) | (daily_positions_df['outstanding_position'] == 0))  &
        ((daily_positions_df['direction'].isna() == True) | (daily_positions_df['direction'] == 0))
                    )
    daily_positions_filtered =  daily_positions_df.loc[~filtered_out, :]

    # Get the daily prices and fx
    daily_positions_filtered = pd.merge(daily_positions_filtered, price_series_df[['symbol', 'price']], left_on = ['date', 'symbol'], right_on = ['date', 'symbol'] , how = 'left')
    daily_positions_filtered = pd.merge(daily_positions_filtered, price_series_df[['symbol', 'price']], left_on = ['date', 'currency'], right_on = ['date', 'symbol'] , how = 'left')
    del daily_positions_filtered['symbol_y'] # this will the symbol for currency
    daily_positions_filtered.rename(columns = {'symbol_x' : 'symbol', 'price_x': 'price', 'price_y': 'fx'}, inplace = True)
    daily_positions_filtered.loc[daily_positions_filtered['currency'] == 'EUR', 'fx'] = 1
    daily_positions_filtered.loc[daily_positions_filtered['amount'].isna() == True, 'amount'] = 0
    
    logger.info("Daily position table updated")

    return daily_positions_filtered
