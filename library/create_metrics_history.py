'''
create_metrics_history.py

This module contains functions to create the history of different metrics from pnl to var

functions:

todo: this should definitely be an object on which I just apply functions that append the object by having more data

dependencies:

'''

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PnlItems():
    def __init__(self, orders_df: pd.DataFrame):
        self.orders_df = orders_df
        self.pnl_items_df = None
        self.pnl_items_df_agg = None
        
    def prepare(self):
        #filter relevant types
        filtered = self.orders_df.loc[
            self.orders_df['Type'].isin(['DIVIDENT', 'Withholding tax', 'SEC fee']) , ['Date', 'Symbol', 'Type', 'Comment', 'Amount', ] ].copy()
        
        # Grouped per date and symbol
        filtered = filtered.groupby(['Date', 'Symbol', 'Type', 'Comment'])['Amount'].sum().reset_index()
        
        self.pnl_items_df_agg = filtered.groupby(['Date', 'Symbol'])['Amount'].sum().reset_index()
        self.pnl_items_df_agg['Date'] = pd.to_datetime(self.pnl_items_df_agg['Date'])

      
        
        
        
class DailyMetrics():
    def __init__(self, daily_positions_df: pd.DataFrame):
        self.daily_asset_metrics = daily_positions_df.loc[
            (daily_positions_df['outstanding_position'] != 0) | (daily_positions_df['direction'] != 0)].copy()
        
    def calculate_mv(self):
        #MV
        self.daily_asset_metrics['MV'] = (
                                        self.daily_asset_metrics['Price'] * 
                                        self.daily_asset_metrics['outstanding_position'] * 
                                        self.daily_asset_metrics['fx']
                                        )
        
    def include_other_pnl_items(self, other_pnl_items):
        daily_asset_metrics = pd.merge(self.daily_asset_metrics, other_pnl_items, left_on = ['Date', 'Symbol'], right_on = ['Date', 'Symbol'], how = 'left' )
        # are the following lines neccessary?
        daily_asset_metrics = daily_asset_metrics.loc[
            (daily_asset_metrics['outstanding_position'] != 0) | (daily_asset_metrics['direction'] != 0)]    
        daily_asset_metrics['Amount'] = daily_asset_metrics['Amount'].fillna(0)
        daily_asset_metrics =  daily_asset_metrics.rename(columns = {'Amount': 'other_pnl'})
        self.daily_asset_metrics = daily_asset_metrics

    
    def calc_pnl(self):
        #Ads several PnL columns
        daily_asset_metrics = self.daily_asset_metrics
        # LTD
        daily_asset_metrics['pnl_ltd'] = daily_asset_metrics['MV'] + daily_asset_metrics['cost_cumsum']  
        # DTD
        daily_asset_metrics = daily_asset_metrics.sort_values(['Symbol', 'Date'])
        daily_asset_metrics['pnl_dtd'] = daily_asset_metrics.groupby('Symbol')['pnl_ltd'].diff()
        daily_asset_metrics.loc[daily_asset_metrics['pnl_dtd'].isna(), 'pnl_dtd' ] = daily_asset_metrics['pnl_ltd']
        # DTD Tot
        daily_asset_metrics['pnl_tot_dtd'] = daily_asset_metrics['pnl_dtd'] + daily_asset_metrics['other_pnl']
        # LTD Tot
        daily_asset_metrics['pnl_tot_ltd'] = daily_asset_metrics.groupby(['Symbol'])['pnl_tot_dtd'].cumsum()     
        # maybe not neccessary: daily_asset_metrics['pnl_ltd'] = daily_asset_metrics.groupby(['Symbol'])['pnl_dtd'].cumsum()
        # LTD rel
        daily_asset_metrics['pnl_rel_ltd'] = daily_asset_metrics['pnl_ltd'] / daily_asset_metrics['cost_cumsum'] *-1
        # LTD Tot rel
        daily_asset_metrics['pnl_rel_tot_ltd'] = daily_asset_metrics['pnl_tot_ltd'] / daily_asset_metrics['cost_cumsum'] *-1
        # DTD rel
        daily_asset_metrics['pnl_rel_dtd'] = daily_asset_metrics['pnl_dtd'] / (daily_asset_metrics['MV']   - daily_asset_metrics['pnl_dtd'])
        self.daily_asset_metrics = daily_asset_metrics
        
        logger.info('PnL Calculated')
    

    def create_daily_portfolio_metrics(self):
         daily_asset_metrics = self.daily_asset_metrics
         #pivots the daily_asset_metrics so that we can calculate statistics on the total
         pivoted = daily_asset_metrics[['Date',  'Symbol', 'pnl_tot_ltd', 'pnl_ltd', 'MV', 'cost_cumsum', 'cost',  'pnl_tot_dtd', 'pnl_rel_dtd']].pivot(index='Date', columns='Symbol', values=[ 'pnl_tot_ltd', 'pnl_rel_dtd', 'MV', 'cost', 'cost_cumsum', 'pnl_ltd', 'pnl_tot_dtd'])
         pivoted['prtf_mv']            = pivoted['MV'].sum(axis = 1)
         pivoted['prtf_cost_sum']      = pivoted['cost_cumsum'].sum(axis = 1) * -1
         pivoted['prtf_pnl_tot_ltd']   = pivoted['pnl_tot_ltd'].sum(axis = 1)
         pivoted['prtf_pnl_ltd']       = pivoted['pnl_ltd'].sum(axis = 1)
         pivoted['prtf_tot_rtn_ltd']   = pivoted['prtf_pnl_tot_ltd'] / pivoted['prtf_cost_sum'] 
         pivoted['prtf_rtn_ltd']       = pivoted['prtf_pnl_ltd'] / pivoted['prtf_cost_sum'] 
         pivoted['prtf_cost_dtd']      = pivoted['cost'].sum(axis = 1)  * -1
         self.daily_portfolio_metrics = pivoted
         
         logger.info('daily_portfolio_metrics DF created')
         
         return self.daily_portfolio_metrics
                
    def establish_bmk(self, price_series_df: pd.DataFrame, symbol: str):
        bmk_price_series = price_series_df.loc[price_series_df['Symbol']== symbol, 'Price'].copy()
        bmk_price_series = pd.DataFrame(bmk_price_series)
        bmk_price_series.columns = pd.MultiIndex.from_tuples([('Price', 'bmk')])
        merged = pd.merge(self.daily_portfolio_metrics , bmk_price_series, how = 'left', left_index = True, right_on = 'Date' )
        #sets the price index to 1 for better comparison
        merged['Price'] = merged['Price'] / merged.iloc[0, merged.columns.get_loc('Price')].iloc[0]
        self.daily_portfolio_metrics = merged
        
    def calc_prtf_nav(self):
        # calculates the NAV of the portfolio
        total_position = self.daily_portfolio_metrics

        total_position['Total_Units'] = 1.0
        total_position['NAV'] = 1.0

        for i in range(len(total_position['Total_Units'])):
            
            if i == 0: ## probably faster if I get rif of the IF and just do it beforehad, so that the if conditiona does not hvae to eb evaluated each time
                total_position.iloc[i, total_position.columns.get_loc('Total_Units')] =  total_position.iloc[i, total_position.columns.get_loc('prtf_cost_dtd')]
                total_position.iloc[i, total_position.columns.get_loc('NAV')] =  (
                                                total_position.iloc[i, total_position.columns.get_loc('prtf_mv')].iloc[0] 
                                              / total_position.iloc[i, total_position.columns.get_loc('Total_Units')].iloc[0]
                                                                                  )

            else:
                total_position.iloc[i, total_position.columns.get_loc('Total_Units')] = ( 
                                                                                 total_position.iloc[i - 1, total_position.columns.get_loc('Total_Units')].iloc[0]
                                                                                 + 
                                                                                 total_position.iloc[i, total_position.columns.get_loc('prtf_cost_dtd')].iloc[0] 
                                                                                                                                                                
                                                                                 /total_position.iloc[i - 1, total_position.columns.get_loc('NAV')].iloc[0]
                                                                                         )
              
                total_position.iloc[i, total_position.columns.get_loc('NAV')] = (
                                                                                   total_position.iloc[i, total_position.columns.get_loc('prtf_mv')].iloc[0] 
                                                                                 / total_position.iloc[i, total_position.columns.get_loc('Total_Units')].iloc[0]
                                                                                )
                                                                                                
                total_position['unit_rtn_tot_dtd'] = total_position['NAV'] / total_position['NAV'].shift(1) -1
        
        self.daily_portfolio_metrics = total_position
       
    def calc_sharpe(self, risk_free_rate = 0.03):
        #calculates sharpe ratio, risk free rate is assumed 3% unless provided
        #1Y Sharpe ratio
        total_position = self.daily_portfolio_metrics
        total_position['1Y_rtn'] = total_position["unit_rtn_tot_dtd"].rolling(window=365).apply(lambda x: (x +1).prod(), raw=True) - 1 
        total_position['1Y_excess_rtn'] = total_position['1Y_rtn'] - risk_free_rate
        total_position['1Y_excess_std_dev'] = total_position['1Y_excess_rtn'].rolling(window=365).apply(lambda x: x.std(), raw=True)
        total_position['1Y_sharpe'] = total_position['1Y_excess_rtn']  / total_position['1Y_excess_std_dev'] 
        
        self.daily_portfolio_metrics = total_position
      
    def get_biggest_daily_loss(self):
        #Calculates biggest daily loss
        daily_portfolio_metrics = self.daily_portfolio_metrics
        unit_rtn = daily_portfolio_metrics['unit_rtn_tot_dtd'].fillna(0)
        biggest_daily_loss_index = daily_portfolio_metrics.index[unit_rtn == unit_rtn.min()]
        prev_day = biggest_daily_loss_index - pd.Timedelta(days = 1)

        loss = (daily_portfolio_metrics.loc[daily_portfolio_metrics.index == biggest_daily_loss_index[0], 'prtf_mv'].iloc[0] 
                        - daily_portfolio_metrics.loc[daily_portfolio_metrics.index == prev_day[0], 'prtf_mv'].iloc[0]
                )

        return print("Biggest daily loss was: " + f"{loss:,.0f}"  + " EUR on: " + str(biggest_daily_loss_index[0].strftime("%Y-%m-%d")))


    def get_biggest_daily_gain(self):
        #Calculates biggest daily loss
        daily_portfolio_metrics = self.daily_portfolio_metrics
        
        unit_rtn = daily_portfolio_metrics['unit_rtn_tot_dtd'].fillna(0)
        biggest_daily_gain_index = daily_portfolio_metrics.index[unit_rtn == unit_rtn.max()]
        prev_day = biggest_daily_gain_index - pd.Timedelta(days = 1)
        


        gain = (daily_portfolio_metrics.loc[daily_portfolio_metrics.index == biggest_daily_gain_index[0], 'prtf_mv'].iloc[0]   
                - daily_portfolio_metrics.loc[daily_portfolio_metrics.index == prev_day[0], 'prtf_mv'].iloc[0]
                )
        
        return print("Biggest daily gain was: " +  f"{gain:,.0f}" + " EUR on: " + str(biggest_daily_gain_index[0].strftime("%Y-%m-%d")))

    
    def get_maximum_drawdown(self):
        # calculates maximum drawdown
        total_position = self.daily_portfolio_metrics
        max_drawdown = 1
        for i in total_position.index[1:]:
            crnt_value = total_position.loc[total_position.index == i, 'NAV'].iloc[0]
            prev_peak = total_position.loc[total_position.index < i, 'NAV'].max()
            prev_peak_date = total_position.loc[total_position.index < i, 'NAV'].idxmax()
            
            drawdown = crnt_value / prev_peak 
            drawdown_abs = total_position.loc[total_position.index == i, 'prtf_mv'].iloc[0] - total_position.loc[total_position.index == prev_peak_date, 'prtf_mv'].iloc[0]
            drawdown_abs_adj = sum(total_position.loc[(total_position.index >= prev_peak_date) & (total_position.index < i), 'prtf_cost_dtd'])
            
            crnt_value - prev_peak
            if drawdown <= max_drawdown:
                max_drawdown = drawdown 
                max_drawdown_abs = drawdown_abs - drawdown_abs_adj
                peak_date = prev_peak_date
                through_date = i
               
        
        return print("Max drawdown was: " +  f"{((max_drawdown -1) * 100):,.1f}" + "% (" + f"{max_drawdown_abs:,.0f}"  + " EUR ) from: " + str(peak_date.strftime("%Y-%m-%d")) + " to " + str(through_date.strftime("%Y-%m-%d")))


def calc_hvar(daily_asset_metrics, price_series_df, confidence_lvl = 95):
    #calculates VaR for the most recent date
    df = daily_asset_metrics.loc[:, ['Date', 'Symbol', 'outstanding_position', 'MV'] ]
    #var date
    var_date = df.iloc[-1]['Date']
    
    #curernt market value
    crnt_mv =  df.loc[df['Date'] == var_date, 'MV'].sum()          

    #current positions
    df = df.loc[df['Date'] == var_date, ['Symbol', 'MV', 'outstanding_position']]

    var = price_series_df.loc[price_series_df['Symbol'].isin(df['Symbol'].tolist()), ['Price', 'Symbol', 'crncy']].reset_index()
    var = pd.merge(var, df[['Symbol',  'outstanding_position']], left_on = 'Symbol', right_on = 'Symbol', how = 'left' )
    var = pd.merge(var, price_series_df[['Symbol', 'Price']].reset_index(), left_on = ['Date', 'crncy'], right_on = ['Date', 'Symbol'] , how = 'left')
    del var['Symbol_y']
    var.rename(columns = {'Symbol_x' : 'Symbol', 'Price_x': 'Price', 'Price_y': 'fx'}, inplace = True)
    var.loc[var['fx'].isna(), 'fx'] = 1
    
    # Calculate MV of the portfolio for the past dates, assuming I would have held the current amount of shares.
    var['prtf_mv'] =  var['Price'] * var['fx'] * var['outstanding_position'] 

    # drop records when prices are not available, such as 1.1.YY
    var = var.dropna()

    # produce dailly MV and the returns...
    returns = pd.DataFrame()
    returns['prtf_mv'] =  var.groupby('Date').agg({'prtf_mv' : 'sum'})
    returns['rtn'] = returns['prtf_mv'] / returns['prtf_mv'].shift(1) - 1

    var_rel = np.percentile(returns['rtn'][1:], 100 - confidence_lvl) #first value is nan, thats why the slice

    var_abs = crnt_mv * var_rel
        
    return [round(var_rel, 4), round(var_abs, 1)]


def backtest_hvar(daily_portfolio_metrics, var_rel, confidence_lvl = 95):
    #backetsting var by counting how many exceptions occured in the history relative to the amount of observations
    exceptions = (daily_portfolio_metrics.loc[:, ['unit_rtn_tot_dtd']] <=  var_rel).sum()
    observations = len(daily_portfolio_metrics) # number of days of existence of portfolio
    exception_rate = exceptions.iloc[0] / observations
    allowed_exception_rate = (100 - confidence_lvl)/100
    if exception_rate < allowed_exception_rate:
        result = 'backtest succesfull'
    else:
        result = 'backtest failed, too many VaR violations'
    
    return print(result + ', exception rate is: ' +  f"{exception_rate*100:,.2f}" + '%, against allowed exception rate of: ' + f"{allowed_exception_rate*100:,.2f}" + "%")
    
    
    
 



