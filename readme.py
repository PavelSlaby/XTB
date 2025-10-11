
# Diagram

#Highest Level



'''
XTB Portfolio Analytics Toolkit
--------------------------------

Purpose:
    - to analyze a financial portfolio, to calculate its pnl and risk, and prepare reports


requirements:
    - a csv file formatted in a certain way


Data Sources
    
    Current:
        - Excel directly exported from XTB 
        - Yahoo for prices and FX
        - hardcoded DF to store the static data
    Future:
        - try parquet?


----------------------
ETL - data_loader.py
            - loading data
            - clean (excel)
            - functions to export outputs

    
Daily_Datasets
    - updates all tables, that have a daily stored history....
        -- in a future, this can be redone so that it only appends....
    - daily_position.py
        - the most basic dataset
    - cal_daily_metrics.py
        - this will call the below module - risk_metrics, it to calculate all the daily stored metrics like VaR etc...

Risk_metrics
    - calculates pnl, sharpe etc...
    - simply functions that calculate bunch of different things....



Reporting Layer (charts, summaries, dashboards)


config
    - settings.py - constants, paths....







'''