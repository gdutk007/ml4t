import datetime as dt
import os
import io
from io import TextIOWrapper
import re
import numpy as np
import math
import pandas as pd
from util import get_data, plot_data
from marketsimcode import *

def testPolicy(symbol="JPM", st=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
   # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(st, ed)
    trades = get_data([symbol], dates)
    # forward fill and backfill data
    trades.fillna(method='ffill', inplace=True)
    trades.fillna(method='bfill', inplace=True)
    # only keep data for symbols in our portfolio
    trades.drop(columns=["SPY"], inplace=True)
    prices = trades.copy()
    trades[:] = 0.0
    prev = 0.0


    # filling in trades table
    for i in range(len(prices)-1):
      if prices.iloc[i][0] < prices.iloc[i+1][0]:
         if prev == 1000:
            prev += 0
         elif prev == 0:
            prev += 1000
         elif prev == -1000:
            prev += 2000
      else:
         if prev == 1000:
            prev += -2000
         elif prev == 0:
            prev += -1000
         elif prev == -1000:
            prev += 0
      trades.iloc[i][0]= prev
    # last two prices are increasing so just hold
    trades.iloc[-1][0] = prev
    return trades
    #import pdb; pdb.set_trace();