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
from TheoreticallyOptimalStrategy import *
import matplotlib.pyplot as plt
from indicators import *


def author(): 
  return 'gdutka3'


def calculate_stats(port_vals):
   port_vals = port_vals/port_vals[0]
   # don't need alloced, alloced = normed*allocs
   # port_vals = alloced.sum(axis=1)
   cum_ret = port_vals[-1]/port_vals[0] - 1
   daily_ret = (port_vals[1:]/port_vals[:-1]) - 1 
   adr = daily_ret.mean(axis=0)
   sddr = daily_ret.std(ddof=1)
   sr =  math.sqrt(252) * adr/sddr
   # import pdb; pdb.set_trace()
   return cum_ret, adr, sddr, sr


def printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals):
   print()
   print()
   print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		  		 		  		  		    	 		 		   		 		  
   print()  		  	   		  		 		  		  		    	 		 		   		 		  
   print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
   print()  		  	   		  		 		  		  		    	 		 		   		 		  
   print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
   print()  		  	   		  		 		  		  		    	 		 		   		 		  
   print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
   print()  		  	   		  		 		  		  		    	 		 		   		 		  
   print(f"Final Portfolio Value: {portVals.iloc[-1][0]}")
   print()
   print()

if __name__ == "__main__":
   start_date=dt.datetime(2008,1,1)
   end_date=dt.datetime(2009,12,31)

   # get optimal trades and benchmark trades df
   tosTrades = testPolicy("JPM",start_date,end_date, 10000)
   benchmarkTrades = benchmark("JPM",start_date,end_date, 10000)

   # comput portvals for optimal trades and benchmark trades
   portVals = compute_portvals(tosTrades, start_val=100000)
   benchportVals = compute_portvals(benchmarkTrades, start_val=100000)

   # get data and print it for optimal trades and benchmark trades
   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(benchportVals.to_numpy())
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, benchportVals)
   
   portVals = portVals/portVals.iloc[0]
   benchportVals = benchportVals/benchportVals.iloc[0]
   ax = portVals.plot()
   benchportVals.plot(ax=ax)
   plt.xlabel('dates')
   plt.ylabel('Normalized return')
   plt.title('Optimal Strategy vs Benchmark')
   plt.legend(["Optimal","Benchmark"])
   #plt.show()
   plt.savefig('./images/TOS.png')

   # getting indicator data for JPM prices
   dates = pd.date_range(start_date, end_date)
   prices = get_data(["JPM"], dates)
   # forward fill and backfill data
   prices.fillna(method='ffill', inplace=True)
   prices.fillna(method='bfill', inplace=True)
   # only keep data for symbols in our portfolio
   prices.drop(columns=["SPY"], inplace=True)
   getIndicators(prices, start_date, end_date)


