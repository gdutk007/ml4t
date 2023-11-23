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
import matplotlib.pyplot as plt
from indicators import *
from ManualStrategy import *
import StrategyLearner  as ln
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

def graph_port_vals_in_sample(benchmark_vals, strategy_vals, plot_vertical_lines=False, trades=[]):
   benchmark_vals = benchmark_vals/benchmark_vals.iloc[0]
   strategy_vals = strategy_vals/strategy_vals.iloc[0]
   ax = strategy_vals.plot(color='red')
   benchmark_vals.plot(ax=ax,color='purple')
   plt.xlabel('dates')
   plt.ylabel('Normalized return')
   if plot_vertical_lines:
      buy_dates = trades.index[trades['JPM'] > 0].tolist()
      sell_dates = trades.index[trades['JPM'] < 0].tolist()
      plt.vlines(colors='blue',x=buy_dates,ymin=strategy_vals.loc[buy_dates,0]-0.2,ymax=strategy_vals.loc[buy_dates,0]+0.2)
      plt.vlines(colors='black',x=sell_dates,ymin=strategy_vals.loc[sell_dates,0]-0.2,ymax=strategy_vals.loc[sell_dates,0]+0.2)

   plt.title('In-Sample Manual Strategy vs Benchmark')
   plt.legend(["Manual Strategy","Benchmark"])
   plt.savefig('./images/in-sample-manual-strategy.png')

def graph_port_vals_out_sample(benchmark_vals, strategy_vals, plot_vertical_lines=False, trades=[]):
   benchmark_vals = benchmark_vals/benchmark_vals.iloc[0]
   strategy_vals = strategy_vals/strategy_vals.iloc[0]
   ax = strategy_vals.plot(color='red')
   benchmark_vals.plot(ax=ax,color='purple')
   plt.xlabel('dates')
   plt.ylabel('Normalized return')
   if plot_vertical_lines:
      buy_dates = trades.index[trades['JPM'] > 0].tolist()
      sell_dates = trades.index[trades['JPM'] < 0].tolist()
      plt.vlines(colors='blue',x=buy_dates,ymin=strategy_vals.loc[buy_dates,0]-0.2,ymax=strategy_vals.loc[buy_dates,0]+0.2)
      plt.vlines(colors='black',x=sell_dates,ymin=strategy_vals.loc[sell_dates,0]-0.2,ymax=strategy_vals.loc[sell_dates,0]+0.2)

   plt.title('Out-Sample Manual Strategy vs Benchmark')
   plt.legend(["Manual Strategy","Benchmark"])
   plt.savefig('./images/out-sample-manual-strategy.png')

def test_strategy_learner():
   start_date = dt.datetime(2008, 1, 1)
   end_date = dt.datetime(2009,12,31)
   commission = 9.95
   impact = 0.0005
   learner = ln.StrategyLearner(verbose = False, impact = 0.0, commission=0.0) # constructor 
   learner.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase 
   trades = learner.testPolicy(symbol = "JPM", sd=start_date, ed=end_date, sv = 100000) # testing phase
   portVals = compute_portvals(trades, start_val=100000, commission=commission,impact=impact)
   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   print('******** in-sample strategy learner  ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   print('***********************************************\n')
   # out of sample JPM
   #  January 1, 2010 to December 31, 2011
   start_date = dt.datetime(2010, 1, 1)
   end_date = dt.datetime(2011,12,31)
   trades = learner.testPolicy(symbol = "JPM", sd=start_date, ed=end_date, sv = 100000) # testing phase
   portVals = compute_portvals(trades, start_val=100000, commission=commission,impact=impact)
   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   print('******** out-of-sample strategy learner  ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   print('***********************************************\n')

def test_manual_strategy():
   ms = ManualStrategy()
   start_date = dt.datetime(2008, 1, 1)
   end_date = dt.datetime(2009,12,31)
   commission = 9.95
   impact = 0.0005

   manual_strategy_trades_insample = ms.testPolicy(symbol = "JPM", sd=start_date , ed=end_date , sv = 100000)
   benchmark_trades = ms.benchmark(symbol = "JPM", sd=start_date , ed=end_date , sv = 100000)
   portVals = compute_portvals(manual_strategy_trades_insample, start_val=100000, commission=commission,impact=impact)
   benchmark_portvals = compute_portvals(benchmark_trades, start_val=100000, commission=commission,impact=impact)
   graph_port_vals_in_sample(benchmark_portvals, portVals, True, manual_strategy_trades_insample)

   # get data and print it for optimal trades and benchmark trades
   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   print('******** in sample manual strategy ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   print('***********************************************\n')

   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(benchmark_portvals.to_numpy())
   print('******** in sample benchmark ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, benchmark_portvals)
   print('***********************************************\n')

   # Out of sample: January 1, 2010 to December 31, 2011
   start_date = dt.datetime(2010, 1, 1)
   end_date = dt.datetime(2011,12,31)
   manual_strategy_trades_outsample = ms.testPolicy(symbol = "JPM", sd=start_date , ed=end_date , sv = 100000)
   benchmark_trades = ms.benchmark(symbol = "JPM", sd=start_date , ed=end_date , sv = 100000)
   portVals = compute_portvals(manual_strategy_trades_outsample, start_val=100000, commission=commission,impact=impact)
   benchmark_portvals = compute_portvals(benchmark_trades, start_val=100000, commission=commission,impact=impact)
   graph_port_vals_out_sample(benchmark_portvals, portVals, True, manual_strategy_trades_outsample)

   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   print('******** Out-sample manual strategy ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   print('***********************************************\n')

   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(benchmark_portvals.to_numpy())
   print('******** out-sample benchmark ************')
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, benchmark_portvals)
   print('***********************************************\n')

def run_manual_strategy(symbol,sd, ed, impact, comission):
   ms = ManualStrategy()
   manual_strategy_trades = ms.testPolicy(symbol = symbol, sd=sd , ed=ed , sv = 100000)
   portVals = compute_portvals(manual_strategy_trades, start_val=100000, commission=comission,impact=impact)
   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   return portVals,manual_strategy_trades

def run_benchmark(symbol,sd, ed, impact, comission):
   ms = ManualStrategy()
   benchmark_trades = ms.benchmark(symbol = symbol, sd=sd, ed=ed , sv = 100000)
   benchmark_portvals = compute_portvals(benchmark_trades, start_val=100000, commission=comission,impact=impact)
   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(benchmark_portvals.to_numpy())
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, benchmark_portvals)
   return benchmark_portvals

def train_learner( symbol, sd, ed, impact, commission):
   learner = ln.StrategyLearner(verbose = False, impact = impact, commission=commission)
   learner.add_evidence(symbol = symbol, sd=sd, ed=ed, sv = 100000)
   return learner

def run_strategy_learner(learner_obj, symbol, sd, ed, impact, commission):
   trades = learner_obj.testPolicy(symbol=symbol, sd=sd, ed=ed, sv = 100000)
   portVals = compute_portvals(trades, start_val=100000, commission=commission,impact=impact)
   cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portVals.to_numpy())
   printData(sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, portVals)
   return portVals


def experiment1():
   commission = 9.95
   impact = 0.0005
   #Training / in-sample: January 1, 2008 to December 31, 2009. 
   sd = dt.datetime(2008, 1, 1)
   ed = dt.datetime(2009,12,31)
   print("*** manual strategy using indicators ***")
   ms_port_vals, manual_strategy_trades = run_manual_strategy("JPM", sd, ed, impact, commission)
   buy_dates = manual_strategy_trades.index[manual_strategy_trades['JPM'] > 0].tolist()
   sell_dates = manual_strategy_trades.index[manual_strategy_trades['JPM'] < 0].tolist()
   print(" ******")
   print("*** Benchmark ***")
   bench_portvals = run_benchmark("JPM", sd, ed, impact, commission)
   print(" ******")
   learner_obj = train_learner("JPM", sd, ed, impact, commission)
   print("*** strategy learner ***")
   ai_portvals = run_strategy_learner(learner_obj,"JPM", sd, ed, impact, commission)
   print(" ******")
   ms_port_vals = ms_port_vals/ms_port_vals.iloc[0]
   bench_portvals = bench_portvals/bench_portvals.iloc[0]
   ai_portvals = ai_portvals/ai_portvals.iloc[0]
   plt.clf()
   ax = ms_port_vals.plot()
   bench_portvals.plot(ax=ax)
   ai_portvals.plot(ax=ax)
   plt.xlabel('Dates')
   plt.ylabel('Normalized return')
   plt.vlines(colors='blue',x=buy_dates,ymin=ms_port_vals.loc[buy_dates,0]-0.2,ymax=ms_port_vals.loc[buy_dates,0]+0.2)
   plt.vlines(colors='black',x=sell_dates,ymin=ms_port_vals.loc[sell_dates,0]-0.2,ymax=ms_port_vals.loc[sell_dates,0]+0.2)
   plt.title('JPM In-Sample Manual Strategy vs Benchmark vs Random Forest')
   plt.legend(["Manual Strategy","Benchmark", "Random Forest"])
   plt.savefig('./images/experiment1-insample.png')

   # Testing / out-of-sample: January 1, 2010 to December 31 2011. 
   sd = dt.datetime(2010, 1, 1)
   ed = dt.datetime(2011,12,31)
   print("*** manual strategy using indicators ***")
   ms_port_vals,manual_strategy_trades = run_manual_strategy("JPM", sd, ed, impact, commission)
   buy_dates = manual_strategy_trades.index[manual_strategy_trades['JPM'] > 0].tolist()
   sell_dates = manual_strategy_trades.index[manual_strategy_trades['JPM'] < 0].tolist()
   print(" ******")
   print("*** Benchmark ***")
   bench_portvals = run_benchmark("JPM", sd, ed, impact, commission)
   print(" ******")
   learner_obj = train_learner("JPM", sd, ed, impact, commission)
   print("*** strategy learner ***")
   ai_portvals = run_strategy_learner(learner_obj,"JPM", sd, ed, impact, commission)
   print(" ******")
   ms_port_vals = ms_port_vals/ms_port_vals.iloc[0]
   bench_portvals = bench_portvals/bench_portvals.iloc[0]
   ai_portvals = ai_portvals/ai_portvals.iloc[0]
   plt.clf()
   ax = ms_port_vals.plot()
   bench_portvals.plot(ax=ax)
   ai_portvals.plot(ax=ax)
   plt.xlabel('Dates')
   plt.ylabel('Normalized return')
   plt.vlines(colors='blue',x=buy_dates,ymin=ms_port_vals.loc[buy_dates,0]-0.2,ymax=ms_port_vals.loc[buy_dates,0]+0.2)
   plt.vlines(colors='black',x=sell_dates,ymin=ms_port_vals.loc[sell_dates,0]-0.2,ymax=ms_port_vals.loc[sell_dates,0]+0.2)
   plt.title('JPM Out-Of-Sample Manual Strategy vs Bencchmark vs Random Forest')
   plt.legend(["Manual Strategy","Benchmark", "Random Forest"])
   plt.savefig('./images/experiment1-outsample.png')

def experiment2():
   commission = 9.95
   impact = 0.0005
   #Training / in-sample: January 1, 2008 to December 31, 2009. 
   sd = dt.datetime(2008, 1, 1)
   ed = dt.datetime(2009,12,31)
   strat_portvals = []
   legend = []
   for i in range(10):
      learner_obj = train_learner("JPM", sd, ed, impact*i, commission )
      ret_values = run_strategy_learner( learner_obj,"JPM", sd, ed, impact, commission )
      # ret_values = ret_values/ret_values[0]
      strat_portvals.append( ret_values )
      legend.append(impact*i)
   ax = strat_portvals[0].plot()
   for i in range( 1, len ( strat_portvals ) ):
      print(i)
      strat_portvals[i].plot(ax=ax)
   plt.title('JPM changing impact on Random Forest')
   plt.legend(legend )
   plt.savefig('./images/experiment2.png')

if __name__ == "__main__":
   experiment2()

