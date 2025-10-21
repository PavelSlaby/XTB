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


import library.data_loader as data_loader

import library.daily_position as daily_position #Creates daily position for each share
import library.create_metrics_history  as create_metrics_history  # creates portfolio view
import library.settings  as settings  # constants/paths

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

# creates metrics object
portfolio = create_metrics(outputs)

# main data objects:

#daily position
daily_position = daily_position.create_position_df(orders_df, tickers_df, price_series_df, history_start)

#pnl pet each asset
daily_asset_metrics = portfolio.daily_asset_metrics

#daily portfolio metrics
daily_portfolio_metrics = portfolio.daily_portfolio_metrics


#%% REporting
'''
Move this to a different module
'''   
   

#Curremt Statistics:
prtf_tot_rtn_ltd    = daily_portfolio_metrics['prtf_tot_rtn_ltd'].iloc[-1] * 100
nav_per_share       = daily_portfolio_metrics['NAV'].iloc[-1]
prtf_pnl_tot_ltd    = daily_portfolio_metrics['prtf_pnl_tot_ltd'].iloc[-1]
prtf_mv             = daily_portfolio_metrics['prtf_mv'].iloc[-1]
invested_own_funds  = daily_portfolio_metrics['prtf_cost_sum'].iloc[-1]
prtf_tot_rtn_1y     = daily_portfolio_metrics['1Y_rtn'].iloc[-1] * 100
prtf_sharpe_1y      = daily_portfolio_metrics['1Y_sharpe'].iloc[-1]

print(f"Total Portfolio Return LTD is: {prtf_tot_rtn_ltd:.2f}%" )
print(f"Total Portfolio Return 1Y is: {prtf_tot_rtn_1y:.2f} %")

print(f"NAV per share is: {nav_per_share:.2f}" )
print(f"LTD Total PnL is: {int(prtf_pnl_tot_ltd):,} EUR")
print(f"Sharpe 1Y is: {prtf_sharpe_1y:.2f}" )

print(f"MV portfolio is: {int(prtf_mv):,} EUR")
print(f"Currently invested own funds: {int(invested_own_funds):,} EUR" )

    
print("Biggest Gain/Loss/Drawdown:")
portfolio.get_biggest_daily_loss()
portfolio.get_biggest_daily_gain()
portfolio.get_maximum_drawdown()

print("VaR")
var = create_metrics_history.calc_hvar(portfolio.daily_asset_metrics, price_series_df)
var

create_metrics_history.backtest_hvar(portfolio.daily_portfolio_metrics, var[0])



from tabulate import tabulate

aggregated_stats = portfolio.daily_asset_metrics.groupby('Symbol').agg({
                                                        'pnl_tot_ltd': 'last',
                                                        'pnl_rel_tot_ltd': 'last', 
                                                        'pnl_dtd': 'last', 
                                                        'outstanding_position': 'last', 
                                                        'Price': 'last', 
                                                        'MV': 'last',  
                                                        'cost_cumsum': 'last'  
                                                        })

aggregated_stats['MV_%'] = aggregated_stats['MV'] / sum(aggregated_stats['MV']) * 100

aggregated_stats = aggregated_stats.rename(columns = {
                                    'pnl_tot_ltd' : 'PL_Tot_LTD', 
                                    'pnl_rel_tot_ltd' :  'PL_Tot_LTD_%',
                                    'pnl_dtd' : 'PL_DTD',
                                    'outstanding_position' : 'Outstanding_Position',
                                    'cost_cumsum' : 'Invested_Amount'
                                 })

aggregated_stats['PL_Tot_LTD_%'] = aggregated_stats['PL_Tot_LTD_%'] * 100

def format_accounting(value: float) -> str:
    return f"{int(value):,}"


def format_percent(value: float):
    return f"{value:.2f}"


aggregated_stats = aggregated_stats.sort_values(by = 'MV_%', ascending = False )


aggregated_stats['PL_Tot_LTD']      = aggregated_stats['PL_Tot_LTD'].apply(format_accounting)
aggregated_stats['PL_DTD']          = aggregated_stats['PL_DTD'].apply(format_accounting)
aggregated_stats['MV']              = aggregated_stats['MV'].apply(format_accounting)
aggregated_stats['Invested_Amount'] = aggregated_stats['Invested_Amount'].apply(format_accounting)

aggregated_stats['PL_Tot_LTD_%']    = aggregated_stats['PL_Tot_LTD_%'].apply(format_percent)
aggregated_stats['Price']           = aggregated_stats['Price'].apply(format_percent)
aggregated_stats['MV_%']            = aggregated_stats['MV_%'].apply(format_percent)



print("Overview per share")
print(tabulate(aggregated_stats, headers=aggregated_stats.columns, numalign="center", tablefmt="grid"))






#%% How much would the pnl be if I invested in X?
# assuming I would invest exactly the same amount of money I did at the same times I did, just in a different ticker...

benchmark_symbol = 'IUSA.DE'

import matplotlib.pyplot as plt

total_position = daily_portfolio_metrics


daily_positions_df = daily_asset_metrics

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

pnl_daily_df = daily_asset_metrics


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





