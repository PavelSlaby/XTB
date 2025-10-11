# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 20:39:31 2025

@author: pavel
"""



# imports standard packages
from datetime import datetime
import logging 


# Folder paths


# manual mappings for tickers and fx, xtb_ticker to yfinance ticker and crncy
tickers_dict = {
                'CRWD.US'  : ['CRWD', 'USD'] , 
                'DTLE.UK'  : ['DTLE.L', 'EUR'],
                'ECAR.UK'  : ['ECAR.L', 'USD'],
                'GOOGC.US' : ['GOOG', 'USD'],
                'NVDA.US'  : ['NVDA', 'USD'],
                'ORSTED.DK': ['ORSTED.CO', 'DKK'],
                'VWS.DK'   : ['VWS.CO' , 'DKK'],
                'TSLA.US'  : ['TSLA', 'USD'],
                'IUSA.DE'  : ['IUSA.DE', 'EUR'], # TODO: maybe I should change the ticker here
                'XAIX.DE'  : ['XAIX.DE', 'EUR'], # TODO: could also change it for (IE00BGV5VN51.SG
                'ENR.DE'   : ['ENR.DE', 'EUR'],
                'AMEM.DE'  : ['AEEM.PA', 'EUR'],
                'QBTS.US'  : ['QBTS', 'USD'],
                'CHPT.US'  : ['CHPT','USD'],
                'OD7F.DE'  : ['OD7F.MU','EUR'], #there are several similat tickets
                'RGTI.US'  : ['RGTI','USD'],
                'IONQ.US'  : ['IONQ','USD']
               }

fx_dict = {
           'DKK' : 'DKKEUR=X',
           'USD' : 'USDEUR=X',
           'EUR' : 'EUREUR=X'
           }

# History timeframe
history_start = '2023-01-01'
history_end = datetime.today()



def setup_logging():
    logging.basicConfig(
                     format="{asctime} : {levelname} : {name} : {message}",
                     style="{",
                     datefmt="%Y-%m-%d %H:%M:%S",
                     level=logging.INFO
                    )





