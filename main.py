'''
main.py

Main script for running portfolio analytics.

This script:
- Loads transaction and market data
- Runs core data transformation
- Prepares data for portfolio analysis, risk, and reporting modules

Project structure: https://github.com/pslaby/portfolio-analytics

'''


# imports standard packages
import os
import pandas as pd
import logging

os.getcwd()
os.chdir(r"D:\Investing\XTB\Repos")

#imports local modules
import library.data_loader as data_loader #imports all data
import library.daily_position as daily_position #Creates daily position for each share
import library.create_metrics_history  as create_metrics_history  # creates portfolio view
import library.settings  as settings  # constants/paths
import library.reporting as reporting

import importlib

importlib.reload(reporting)

settings.setup_logging()
logger = logging.getLogger(__name__)


#pandas params
pd.set_option('display.max_columns', None) #to display all columns
pd.set_option('display.width', 500)  #use the entire width to display the columns
pd.set_option('display.max_rows', 1000)



def load_data(source: str):
    # 1: Checks connections
    data_loader.check_yfinance_connection()
    
    # 3: Loads data from XTB
    prtf_file_loaded = data_loader.load_prtf_file(folder_path = source)
    orders_df = data_loader.read_orders_df(prtf_file_loaded)
    
    # 4: Receives ticker level data
    tickers_df = data_loader.create_tickers_df(tickers_dict, fx_dict)
   
    # 4: gets price time series
    price_series_df  = data_loader.download_tickers_prices(tickers_df, history_start, history_end)
   
    # 2 Prepare outputs
    outputs = {
                "orders_df": orders_df,
                "tickers_df": tickers_df,
                "price_series_df": price_series_df
                }
    
    return outputs


def create_metrics(outputs):
    orders_df       = outputs["orders_df"]
    tickers_df      = outputs["tickers_df"]
    price_series_df = outputs["price_series_df"]
    
    # creates DF with daily position        
    daily_positions_df = daily_position.create_position_df(orders_df, tickers_df, price_series_df, history_start)

    #prepares pnl related items from Daily Position DF
    pnl_items_obj = create_metrics_history.PnlItems(orders_df)
    pnl_items_obj.prepare()
    
    #creates new object from the daily position DF. New object has a daily_asset_metrics DF which looks at each asset individually
    metrics_obj = create_metrics_history.DailyMetrics(daily_positions_df)
    metrics_obj.calculate_mv()
    metrics_obj.include_other_pnl_items(pnl_items_obj.pnl_items_df_agg)
    metrics_obj.calc_pnl()
    
    #creates daily_portfolio_metrics DF which has portfolio level data
    metrics_obj.create_daily_portfolio_metrics()
    metrics_obj.establish_bmk(price_series_df, 'IUSA.DE')    
    metrics_obj.calc_prtf_nav()
    metrics_obj.calc_sharpe()

    return metrics_obj



#%% actual running of functions

# Folder paths
xtb_input = settings.xtb_input

# manual mappings for tickers and fx, xtb_ticker to yfinance ticker and crncy
tickers_dict = settings.tickers_dict
fx_dict = settings.fx_dict


# History timeframe
history_start = settings.history_start
history_end = settings.history_end

# Prepares data
outputs = load_data(xtb_input)


# stores the outputs just in case they are needed later
orders_df       = outputs["orders_df"]
tickers_df      = outputs["tickers_df"]
price_series_df = outputs["price_series_df"]

data_loader.check_constants_exist(tickers_df, orders_df)


# creates metrics object, it is and object that contains 2 DF - daily asset metrics and daily portfolio metrics
portfolio = create_metrics(outputs)

# main data objects:

#daily position
daily_position = daily_position.create_position_df(orders_df, tickers_df, price_series_df, history_start) # this step is duplicated

#pnl pet each asset
daily_asset_metrics = portfolio.daily_asset_metrics

#daily portfolio metrics
daily_portfolio_metrics = portfolio.daily_portfolio_metrics


#%% Reporting



reporting.overview_per_ticker(portfolio)

reporting.print_crnt_prtf_stats(portfolio, price_series_df)

reporting.simulate_bmk_rtn('IUSA.DE', portfolio, price_series_df)


reporting.plot_ticker_mv('ENR.DE', portfolio)

reporting.graph_assets_mv(portfolio, one_graph = True)

reporting.graph_mv_stacked(portfolio)


   
  

#%% -- INTEREST
interest_orders_df = orders_df.loc[orders_df['Type'].isin(['Free funds interests tax', 'Free funds interests'])]
interest_orders_df[interest_orders_df.Time >= '2024-01-01']['Amount'].sum()





