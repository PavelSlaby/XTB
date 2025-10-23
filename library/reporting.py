# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 21:22:52 2025

reporting

@author: pavel
"""

import library.settings  as settings
import logging
from tabulate import tabulate
import library.create_metrics_history  as create_metrics_history  # creates portfolio view

logger = logging.getLogger(__name__)

def format_accounting(value: float) -> str:
    return f"{int(value):,}"


def format_percent(value: float):
    return f"{value:.2f}"


# Current Statistics:
def print_crnt_prtf_stats(portfolio, price_series_df):
    logger.info('Printing current portfolio stats')

    daily_portfolio_metrics = portfolio.daily_portfolio_metrics

    prtf_tot_rtn_ltd = daily_portfolio_metrics['prtf_tot_rtn_ltd'].iloc[-1] * 100
    nav_per_share = daily_portfolio_metrics['NAV'].iloc[-1]
    prtf_pnl_tot_ltd = daily_portfolio_metrics['prtf_pnl_tot_ltd'].iloc[-1]
    prtf_mv = daily_portfolio_metrics['prtf_mv'].iloc[-1]
    invested_own_funds = daily_portfolio_metrics['prtf_cost_sum'].iloc[-1]
    prtf_tot_rtn_1y = daily_portfolio_metrics['1Y_rtn'].iloc[-1] * 100
    prtf_sharpe_1y = daily_portfolio_metrics['1Y_sharpe'].iloc[-1]

    print('\n PnL:')
    print(f"Total Portfolio Return LTD is: {prtf_tot_rtn_ltd:.2f}%")
    print(f"Total Portfolio Return 1Y is: {prtf_tot_rtn_1y:.2f} %")

    print(f"NAV per share is: {nav_per_share:.2f}")
    print(f"LTD Total PnL is: {int(prtf_pnl_tot_ltd):,} EUR")

    print(f"MV portfolio is: {int(prtf_mv):,} EUR")
    print(f"Currently invested own funds: {int(invested_own_funds):,} EUR")

    print('\n Risk:')
    print(f"Sharpe 1Y is: {prtf_sharpe_1y:.2f}")

    portfolio.get_biggest_daily_loss()
    portfolio.get_biggest_daily_gain()
    portfolio.get_maximum_drawdown()

    var = create_metrics_history.calc_hvar(portfolio.daily_asset_metrics, price_series_df)
    print(f"1 Day VaR is: {var[0]*100}% or {format_accounting(var[1])} EUR")

    create_metrics_history.backtest_hvar(portfolio.daily_portfolio_metrics, var[0])






def overview_per_ticker(portfolio):
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










plt.rcParams['figure.figsize'] = [8, 8]



def graph_assets_mv(xtb_symbol, portfolio, one_graph = True):
    daily_asset_metrics = portfolio.daily_asset_metrics

    if one_graph == True:
        ## 2]  Show MV of each symbol in one graph
          for i in list(daily_asset_metrics['Symbol'].unique()):
            x_axis = daily_asset_metrics.loc[daily_asset_metrics['Symbol'] == i, 'Date']
            y_axis = daily_asset_metrics.loc[daily_asset_metrics['Symbol'] == i, 'MV']

            plt.plot(x_axis, y_axis, label=i)
            plt.title('MV Growth')
            plt.legend(loc=0)

        plt.show()

    elif one_graph == False:             print('aa')


        for i in list(daily_asset_metrics['Symbol'].unique()):
            x_axis = daily_asset_metrics.loc[daily_asset_metrics['Symbol'] == i, 'Date']
            y_axis = daily_asset_metrics.loc[daily_asset_metrics['Symbol'] == i, 'MV']

            plt.plot(x_axis, y_axis, label=i)
            plt.title(i)
            plt.legend(loc=0)
            plt.show()


def graph_mv_stacked(portfolio):
    ##  Show MV of each symbol in the same graph Stacked
    daily_asset_metrics = portfolio.daily_asset_metrics

    pivoted_df = daily_asset_metrics[['Date', 'Symbol', 'MV']].pivot(index='Date', columns='Symbol', values='MV')
    pivoted_df.fillna(0, inplace=True)

    # sort it by when I invested in it....
    sorting_list = []
    for i in pivoted_df.columns:
        sorting_list.append([i, pivoted_df.loc[:, i].loc[pivoted_df.loc[:, i] != 0].index[0]])

    sorting_list = sorted(sorting_list, key=lambda item: item[1])
    column_order = [i[0] for i in sorting_list]

    fig, ax = plt.subplots()
    ax.stackplot(pivoted_df.index, pivoted_df[column_order].T.values, labels=pivoted_df[column_order].columns)
    ax.set_title('Market Value Growth')
    ax.set_ylabel('EUR')
    plt.legend(loc=2)
    plt.show()






