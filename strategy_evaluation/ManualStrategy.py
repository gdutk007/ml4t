import datetime as dt
import random
import pandas as pd
import util as ut
from indicators import getIndicators
from marketsimcode import *
import numpy as np

class ManualStrategy(object):
   
   ms_strategy_trades = []
   ms_portvals = []
   benchmark_trades = []
   benchmark_portvals = []
   def author( self ): 
      return 'gdutka3'

   def testPolicy(self, symbol, sd,ed,sv):
      dates = pd.date_range(sd, ed)
      trades = ut.get_data([symbol], dates)

      # forward fill and backfill data
      trades.fillna(method='ffill', inplace=True)
      trades.fillna(method='bfill', inplace=True)
      
      # only keep data for symbols in our portfolio
      trades.drop(columns=["SPY"], inplace=True)
      prices = trades.copy()
      df_indicators = getIndicators(prices, sd, ed)
      df_indicators['macd_diff'] = df_indicators['macd'] - df_indicators['macd_signal']
      df_indicators['macd_diff_shift'] = df_indicators['macd_diff'].shift(1) # get crossover
      df_signals = pd.DataFrame(index=df_indicators.index)
      df_signals['bbp'] = [1 if x <= -0.03 else -1 if x >= 1.03 else 0 for x in df_indicators['bbp']]
      df_signals['sma_ratio'] = [1 if x <= 0.955 else -1 if x >= 1.055 else 0 for x in df_indicators['price_sma_ratio']]
      df_signals['macd'] = 0

      for index,row in df_indicators.iterrows():
         if row['macd_diff'] > 0.0 and row['macd_diff_shift'] < 0.0:
            df_signals.at[index,'macd'] = 1
         elif row['macd_diff'] < 0.0 and row['macd_diff_shift'] > 0.0:
            df_signals.at[index,'macd'] = -1

      df_trades = pd.DataFrame(index=df_indicators.index)
      df_trades[trades.columns[0]] = 0
      holding = 0
      for i in df_trades.index:
         if holding == 0:
            if (df_signals.ix[i,'bbp'] > 0) | (df_signals.ix[i,'macd'] > 0) | (df_signals.ix[i,'sma_ratio'] > 0) :
               df_trades.ix[i,0] = 1000
               holding = 1000
            elif (df_signals.ix[i,'bbp'] < 0) | (df_signals.ix[i,'macd'] < 0) | (df_signals.ix[i,'sma_ratio'] < 0):
               df_trades.ix[i,0] = -1000
               holding = -1000
         elif holding == 1000:
            if (df_signals.ix[i,'bbp'] < 0) | ((df_signals.ix[i,'macd'] < 0) | (df_signals.ix[i,'sma_ratio'] < 0)):
               df_trades.ix[i,0] = -2000
               holding = -1000
         elif holding == -1000:
            if (df_signals.ix[i,'bbp'] > 0) | ((df_signals.ix[i,'macd'] > 0) | (df_signals.ix[i,'sma_ratio'] > 0)):
               df_trades.ix[i,0] = 2000
               holding = 1000
      return df_trades

   def benchmark(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
      # Read in adjusted closing prices for given symbols, date range
      dates = pd.date_range(sd, ed)
      trades = ut.get_data([symbol], dates)
      # forward fill and backfill data
      trades.fillna(method='ffill', inplace=True)
      trades.fillna(method='bfill', inplace=True)
      # only keep data for symbols in our portfolio
      trades.drop(columns=["SPY"], inplace=True)
      prices = trades.copy()
      trades[:] = 0.0
      # benchmark only has 1000 shares held long
      trades.iloc[0][0] = 1000
      return trades
