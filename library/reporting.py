# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 21:22:52 2025

reporting

@author: pavel
"""

import logging
from tabulate import tabulate
import library.create_metrics_history  as create_metrics_history  # creates portfolio view
import matplotlib.pyplot as plt
import pandas as pd
import library.data_loader as data_loader
import library.settings as settings

logger = logging.getLogger(__name__)

def format_accounting(value) -> str:
    if pd.isna(value):
        return ""
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
    print(f"1 Day VaR is: {format_percent(var[0]*100)}% or {format_accounting(var[1])} EUR")

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


def simulate_bmk_rtn(benchmark_symbol, portfolio, price_series_df):
    daily_portfolio_metrics = portfolio.daily_portfolio_metrics

    daily_positions_df = portfolio.daily_asset_metrics

    invested_amount = daily_positions_df[['Date', 'cost']].groupby('Date').sum('cost')
    benchmark_price_series_df = price_series_df.loc[price_series_df['Symbol'] == benchmark_symbol, ['Price']]

    invested_amount_price = pd.merge(invested_amount, benchmark_price_series_df, how='left', left_on='Date',
                                     right_on='Date')
    invested_amount_price.loc[invested_amount_price['cost'] == 0, 'cost'] = 0  # np.nan
    invested_amount_price['direction'] = invested_amount_price['cost'] * -1 / invested_amount_price['Price']
    invested_amount_price['cost_cumsum'] = invested_amount_price['cost'].cumsum() * -1
    invested_amount_price['direction_cumsum'] = invested_amount_price['direction'].cumsum()
    invested_amount_price['MV'] = invested_amount_price['direction_cumsum'] * invested_amount_price['Price']
    invested_amount_price['Total_Rel_Rtn'] = invested_amount_price['MV'] / invested_amount_price['cost_cumsum'] - 1

    # graph
    x_axis = daily_portfolio_metrics.index
    y_axis1 = daily_portfolio_metrics['prtf_pnl_ltd']
    y_axis2 = daily_portfolio_metrics['prtf_mv']
    y_axis3 = daily_portfolio_metrics['prtf_cost_sum']
    y_axis4 = daily_portfolio_metrics['prtf_tot_rtn_ltd']
    y_axis5 = invested_amount_price['MV']
    y_axis6 = invested_amount_price['Total_Rel_Rtn']

    fig, ax1 = plt.subplots()
    ax1.plot(x_axis, y_axis1, label='PnL')
    ax1.plot(x_axis, y_axis2, label='MV')
    ax1.plot(x_axis, y_axis3, label='Invested Capital')
    ax1.plot(x_axis, y_axis5, label=benchmark_symbol)

    ax1.set_ylabel('EUR')

    ax2 = ax1.twinx()
    ax2.plot(x_axis, y_axis4, color='orange', label='% Return')
    ax2.plot(x_axis, y_axis6, color='red', label=str(benchmark_symbol) + ' return')

    ax2.set_ylabel('% Rtn', color='orange')
    ax2.tick_params(labelcolor='orange')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    prtf_tot_rtn_last = daily_portfolio_metrics['prtf_tot_rtn_ltd'].tail(1)[0] * 100
    prtf_rtn_last = daily_portfolio_metrics['prtf_rtn_ltd'].tail(1)[0] * 100
    total_rel_rtn_last = invested_amount_price['Total_Rel_Rtn'].tail(1)[0] * 100

    plt.show()

    print("Benchmark symbol is: " + benchmark_symbol)
    print("note that this simulation does not consider DIVIDENTs....")
    print("Portfolio return was: " + str(prtf_rtn_last.round(3)) + "%")
    print("Portfolio total return was: " + str(prtf_tot_rtn_last.round(3)) + "%")
    print("Simulated benchmark return was: " + str(total_rel_rtn_last.round(3)) + "%")
    print("Excess non-DIVIDENT return was: " + str((prtf_rtn_last - total_rel_rtn_last).round(3)) + "%")


def plot_ticker_mv(xtb_symbol, portfolio):
    portfolio.daily_asset_metrics.loc[portfolio.daily_asset_metrics['Symbol'] == xtb_symbol, ['Symbol', 'MV'] ].plot()
    plt.show()




plt.rcParams['figure.figsize'] = [8, 8]



def graph_assets_mv(portfolio, one_graph = True):
    daily_asset_metrics = portfolio.daily_asset_metrics

    if one_graph:
        # All symbols in one graph
        for i in daily_asset_metrics['Symbol'].unique():
            x_axis = daily_asset_metrics.loc[daily_asset_metrics['Symbol'] == i, 'Date']
            y_axis = daily_asset_metrics.loc[daily_asset_metrics['Symbol'] == i, 'MV']
            plt.plot(x_axis, y_axis, label=i)

        plt.title('MV Growth')
        plt.legend(loc=0)
        plt.show()

    else:
        # Separate graph per symbol
        for i in daily_asset_metrics['Symbol'].unique():
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



def print_financials(tickers, datapoints):
    financials = data_loader.load_financials(datapoints, tickers)

   # financials.loc[:, 'freeCashflow'] = financials.loc[:, 'freeCashflow'].astype("Int64").apply(format_accounting)
    financials.loc[:, 'returnOnEquity'] = financials.loc[:, 'returnOnEquity'].apply(format_percent)

    print(tabulate(financials, headers=financials.columns, numalign="center", tablefmt="grid", showindex=False))
