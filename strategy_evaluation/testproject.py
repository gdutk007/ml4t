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
# from TheoreticallyOptimalStrategy import *
import matplotlib.pyplot as plt
from indicators import *
from ManualStrategy import *
import pdb


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
   print(f"Sharpe Ratio of portfolio: {sharpe_ratio}")  		  	   		  		 		  		  		    	 		 		   		 		  	 		  
   print(f"Cumulative Return of portfolio: {cum_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  	 		  
   print(f"Standard Deviation of portfolio: {std_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  	 		  
   print(f"Average Daily Return of portfolio: {avg_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  	 		  
   print(f"Final Portfolio Value: {portVals.iloc[-1][0]}")
   print()

def graph_port_vals_in_sample(benchmark_vals, strategy_vals):
   benchmark_vals = benchmark_vals/benchmark_vals.iloc[0]
   strategy_vals = strategy_vals/strategy_vals.iloc[0]
   ax = strategy_vals.plot(color='red')
   benchmark_vals.plot(ax=ax,color='purple')
   plt.xlabel('dates')
   plt.ylabel('Normalized return')
   plt.title('In-Sample Manual Strategy vs Benchmark')
   plt.legend(["Manual Strategy","Benchmark"])
   plt.savefig('./images/in-sample-manual-strategy.png')

def graph_port_vals_out_sample(benchmark_vals, strategy_vals):
   benchmark_vals = benchmark_vals/benchmark_vals.iloc[0]
   strategy_vals = strategy_vals/strategy_vals.iloc[0]
   ax = strategy_vals.plot(color='red')
   benchmark_vals.plot(ax=ax,color='purple')
   plt.xlabel('dates')
   plt.ylabel('Normalized return')
   plt.title('Out-Sample Manual Strategy vs Benchmark')
   plt.legend(["Manual Strategy","Benchmark"])
   plt.savefig('./images/out-sample-manual-strategy.png')


if __name__ == "__main__":
   #start_date=dt.datetime(2008,1,1)
   #end_date=dt.datetime(2009,12,31)
   # get optimal trades and benchmark trades df
   # tosTrades = testPolicy("JPM",start_date,end_date, 10000)
   # benchmarkTrades = benchmark("JPM",start_date,end_date, 10000)

   # comput portvals for optimal trades and benchmark trades
   # portVals = compute_portvals(tosTrades, start_val=100000)
   # benchportVals = compute_portvals(benchmarkTrades, start_val=100000)

   # get data and print it for optimal trades and benchmark trades
   # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   # print()
   # print('Statistics of Optimal Strategy:\n')
   # printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(benchportVals.to_numpy())
   # print('Statistics of Benchmark:\n')
   # printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, benchportVals)
   
   # portVals = portVals/portVals.iloc[0]
   # benchportVals = benchportVals/benchportVals.iloc[0]
   # ax = portVals.plot(color='red')
   # benchportVals.plot(ax=ax,color='purple')
   # plt.xlabel('dates')
   # plt.ylabel('Normalized return')
   # plt.title('Optimal Strategy vs Benchmark')
   # plt.legend(["Optimal","Benchmark"])
   # #plt.show()
   # plt.savefig('./images/TOS.png')

   # # getting indicator data for JPM prices
   # dates = pd.date_range(start_date, end_date)
   # prices = get_data(["JPM"], dates)
   # # forward fill and backfill data
   # prices.fillna(method='ffill', inplace=True)
   # prices.fillna(method='bfill', inplace=True)
   # # only keep data for symbols in our portfolio
   # prices.drop(columns=["SPY"], inplace=True)
   # getIndicators(prices, start_date, end_date)
   ms = ManualStrategy()
   start_date = dt.datetime(2008, 1, 1)
   end_date = dt.datetime(2009,12,31)
   commission = 0#9.95
   impact = 0#0.0005

   manual_strategy_trades = ms.testPolicy(symbol = "JPM", sd=start_date , ed=end_date , sv = 100000)
   benchmark_in_sample_manual = ms.benchmark(symbol = "JPM", sd=start_date , ed=end_date , sv = 100000)
   portVals = compute_portvals(manual_strategy_trades, start_val=100000, commission=commission,impact=impact)
   benchmark_in_sample_portvals = compute_portvals(benchmark_in_sample_manual, start_val=100000, commission=commission,impact=impact)
   graph_port_vals_in_sample(benchmark_in_sample_portvals, portVals)

   # get data and print it for optimal trades and benchmark trades
   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   print('******** in sample manual strategy ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   print('***********************************************\n')

   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(benchmark_in_sample_portvals.to_numpy())
   print('******** in sample benchmark ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, benchmark_in_sample_portvals)
   print('***********************************************\n')

   # Out of sample: January 1, 2010 to December 31, 2011
   start_date = dt.datetime(2010, 1, 1)
   end_date = dt.datetime(2011,12,31)
   manual_strategy_trades = ms.testPolicy(symbol = "JPM", sd=start_date , ed=end_date , sv = 100000)
   benchmark_in_sample_manual = ms.benchmark(symbol = "JPM", sd=start_date , ed=end_date , sv = 100000)
   portVals = compute_portvals(manual_strategy_trades, start_val=100000, commission=commission,impact=impact)
   benchmark_in_sample_portvals = compute_portvals(benchmark_in_sample_manual, start_val=100000, commission=commission,impact=impact)
   graph_port_vals_out_sample(benchmark_in_sample_portvals, portVals)

   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   print('******** Out-sample manual strategy ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   print('***********************************************\n')

   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(benchmark_in_sample_portvals.to_numpy())
   print('******** out-sample benchmark ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, benchmark_in_sample_portvals)
   print('***********************************************\n')

   #pdb.set_trace()
