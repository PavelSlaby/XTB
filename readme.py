
'''
XTB Portfolio Analytics Toolkit
--------------------------------

Purpose:
    - to analyze a financial portfolio, to calculate its pnl and risk, and prepare reports

requirements:
    - an XTB excel file

Data Sources
    Current:
        - Excel directly exported from XTB 
        - Yahoo for prices and FX
        - hardcoded DF to store the static data in the settings.py file

-----------------------------------------------------
Files:

main.py     #Entry point for running the full workflow
library/
    - setings.py                # contains configuration/parameters
    - reporting.py              # creates functions to create reporting tables/graphs
    - data_loader.py            # functions to load all data sources and do basic data cleaning/transformations
    - daily_posiiton.py         # creates daily position out of the XTB data source -> this is then used as a source table for the asset/portfolio metrics
    - cal_daily_metrics.py      # creates asset/portfolio pnl/metrics and all other metrics historically






'''