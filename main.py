"""
main.py

Main script for running portfolio analytics.

This script:
- Loads transaction and market data
- Runs core data transformation
- Prepares data for portfolio analysis, risk, and reporting modules
- total return seems to be smaller then the actual return

Project structure: https://github.com/pslaby/portfolio-analytics
"""
import numpy as np
from matplotlib import pyplot as plt

'''
    TODO: 
    
    - next step is to review the  var in print_current_portfolio_metrics - I skipped var in that one....
        and then its all reviewed and done. 
        - var is being calculated on t-1 date...try to make it live...
    
    - portfolio return versus total portfolio return seems weird too big of a difference.....
    
        - sortino ratio
        - add realized and mtm of unrealized position into the main table...
        - double check since when the dataloader loads the market prices - what timeframe I need for VaR, and if it does not download unnecessarily too long history
        - make sure all different transaction types are loaded, it seems some could not be - correction etc...
    
    check in data loader - stock splits: 'OD7F.MU'
'''



# Import standard packages
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Import local modules
from library import data_loader
from library import daily_position
from library import create_metrics_history
from library import settings
from library import reporting

# setting current working directory
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()


# Loading Constants
xtb_input = settings.XTB_INPUT_FILE_PATH    # Folder paths
tickers_dict = settings.TICKERS_DICT        # Manual mappings for tickers and fx
fx_dict = settings.FX_DICT                  # Manual mappings for tickers and fx
history_start = settings.CALCS_START_DATE   # History timeframe
history_end = settings.CALCS_END_DATE
benchmark_ticker = settings.BENCHMARK_TICKER
default_plot_ticker = settings.DEFAULT_PLOT_TICKER

# Logging
settings.setup_logging()
logger = logging.getLogger(__name__)

# Pandas parameters
pd.set_option('display.max_columns', None) # to display all columns
pd.set_option('display.width', 500)  # use the entire width to display the columns
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
    
    # Creates DF with daily position
    daily_positions_df = daily_position.create_position_df(orders_df, tickers_df, price_series_df, history_start)

    # Prepares pnl related items from Daily Position DF
    pnl_items_obj = create_metrics_history.PnlItems(orders_df)
    pnl_items_obj.prepare()
    
    # Creates new object from the daily position DF. New object has a daily_asset_metrics DF which looks at each asset individually
    metrics_obj = create_metrics_history.DailyMetrics(daily_positions_df)
    metrics_obj.calculate_mv()
    metrics_obj.include_other_pnl_items(pnl_items_obj.pnl_items_other_sum)

    metrics_obj.get_closed_positions(pnl_items_obj.pnl_closed_positions)

    metrics_obj.calc_pnl()
    
    # Creates daily_portfolio_metrics DF which has portfolio level data
    metrics_obj.create_daily_portfolio_metrics()
    metrics_obj.establish_bmk(price_series_df, benchmark_ticker)
    metrics_obj.calc_prtf_nav()
    metrics_obj.calc_sharpe()

    return metrics_obj


def run_reporting(portfolio, price_series_df):
    reporting.overview_per_ticker(portfolio)
    reporting.print_current_portfolio_metrics(portfolio, price_series_df)
    reporting.simulate_bmk_rtn(benchmark_ticker, portfolio, price_series_df)
    reporting.plot_ticker_mv(default_plot_ticker, portfolio)
    reporting.graph_assets_mv(portfolio, one_graph=True)
    reporting.graph_mv_stacked(portfolio)

    logger.info("Printing fundamentals for portfolio assets..takes a bit...")

    keys = portfolio.daily_asset_metrics.loc[portfolio.daily_asset_metrics['date'] == datetime.today().strftime('%Y-%m-%d'), 'symbol'] # picks only those that still have a position
    filtered = {k: settings.TICKERS_DICT[k] for k in keys if k in settings.TICKERS_DICT}

    reporting.print_financials(filtered, settings.DATAPOINTS)

#%% Actual running of functions ----------------------------------------------------------------------------------------

# Load data-------------------------------------------------------------------------------------------------------------

logger.info("Loading data............")
outputs = load_data(xtb_input)

# Stores the outputs just in case they are needed later
orders_df       = outputs["orders_df"]
tickers_df      = outputs["tickers_df"]
price_series_df = outputs["price_series_df"]


# Data Checks
data_loader.check_constants_exist(tickers_df, orders_df)

# Creates Main Metrics -------------------------------------------------------------------------------------------------
logger.info("Creating metrics...................")
portfolio = create_metrics(outputs)
"""
Creates metrics object, it is an object that contains 2 DF:
    daily_asset_metrics - contains metrics per asset
    daily_portfolio_metrics - contains metrics per entire portfolio
"""

daily_asset_metrics = portfolio.daily_asset_metrics
daily_portfolio_metrics = portfolio.daily_portfolio_metrics


#%% Reporting ---------------------------------------------------------------------------------------------------------
logger.info("Running reporting.............")
run_reporting(portfolio, price_series_df)


#### Understanding of selected stocks relationships/correlations etc....


import numpy as np
import seaborn as sns
import datetime as dt

#### Prep
returns = price_series_df
returns['return'] = price_series_df.groupby('symbol')['price'].pct_change()

# Remove non-trading days
returns.replace(0, np.nan, inplace=True)
returns.dropna(inplace=True)

# Filter only the currently held symbols
current_holding_filter = daily_asset_metrics.loc[(daily_asset_metrics['date'] == dt.datetime.today().strftime("%Y-%m-%d")) & (daily_asset_metrics['outstanding_position'] != 0), 'symbol']
returns = returns.loc[returns['symbol'].isin(current_holding_filter), :]


#################################
# Show risk per stock
#################################

# box plot per stock
fig, ax = plt.subplots(figsize=(13, 8))
returns.boxplot(by = 'symbol', column = 'return', ax = ax)
ax.set_title('Return Distribution by Symbol')
ax.set_xlabel('')
ax.set_ylabel('1D Return')
plt.suptitle('')  # removes pandas automatic title
plt.xticks(rotation=30)
plt.tight_layout()

# violin plot
fig, ax = plt.subplots(figsize=(13, 8))
sns.violinplot(x = 'symbol', y = 'return', data = returns, inner='quartile', ax = ax)
ax.set_title('Return Distribution by Symbol')
ax.set_xlabel('')
ax.set_ylabel('1D Return')
ax.text(
    0.01, -0.10,
    "Violin is drawn using a kernel density estimate of the underlying distribution; inner lines show quartiles (25%, 50%, 75%).",
    transform=ax.transAxes,
    fontsize=10,
    ha='left',
)


# Distribution of returns
returns_pivot = returns.reset_index().pivot(index='date', columns='symbol', values='return').dropna()

stats_table = pd.DataFrame(returns_pivot.mean() * 100, columns=['mean'])
stats_table['std'] = returns_pivot.std() * 100



## Correlation
big_mv_filter   = ['DTLE.UK', 'ENR.DE', 'IONQ.US', 'GOOGC.US', 'IUSA.DE', 'NVDA.US', 'ORSTED.DK', 'QBTS.US', 'RGTI.US', 'VWS.DK', 'XAIX.DE']
energy_filter   = ['DTLE.UK', 'ENR.DE', 'IUSA.DE', 'ORSTED.DK', 'VWS.DK']

fig, ax = plt.subplots(figsize=(13, 8))
sns.heatmap(returns_pivot[big_mv_filter].corr(), annot= True, ax = ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Correlation: Big MV Stocks')


fig, ax = plt.subplots(figsize=(13, 8))
sns.heatmap(returns_pivot[energy_filter].corr(), annot= True, ax = ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Correlation: Energy Stocks')




## Comparison with normal distribution 2 biggest positions

from scipy import stats
from matplotlib import pyplot as plt


graphed_symbol = 'ENR.DE'
fig, ax = plt.subplots(figsize=(4, 4))
stats.probplot(returns_pivot[graphed_symbol], plot=ax)
ax.set_title('QQ-Plot: ' + graphed_symbol)

graphed_symbol = 'IUSA.DE'
fig, ax = plt.subplots(figsize=(4, 4))
stats.probplot(returns_pivot[graphed_symbol], plot=ax)
ax.set_title('QQ-Plot: ' + graphed_symbol)









# Distribution of returns of all instruments
# this is kinda useless as it mixes high risk with low risk instruments....
fig, ax = plt.subplots()
returns['return'].plot.hist(bins = 1000, xlim= [-0.40, 0.40], density = True, ax = ax)
returns['return'].plot.density(ax = ax)
plt.title('Distribution of returns of all instruments')



