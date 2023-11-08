import datetime as dt
import random
import pandas as pd
import util as ut
from indicators import getIndicators
import numpy as np
import pdb;

class ManualStrategy(object):

   def testPolicy(self, symbol, sd,ed,sv):
      dates = pd.date_range(sd, ed)
      trades = ut.get_data([symbol], dates)
      
      # forward fill and backfill data
      trades.fillna(method='ffill', inplace=True)
      trades.fillna(method='bfill', inplace=True)
      
      # only keep data for symbols in our portfolio
      trades.drop(columns=["SPY"], inplace=True)
      prices = trades.copy()
      # might use this trades df later
      #trades[:] = 0.0
      #trades.iloc[0][0] = 1000
      df_indicators = getIndicators(prices, sd, ed)
      df_buy = pd.DataFrame(index=df_indicators.index)
      df_sell = df_buy.copy()
      # set up buy signals
      df_buy['sma'] = np.where(df_indicators['sma'] < df_indicators['price'],1,0)
      df_buy['bb'] = np.where(df_indicators['price'] <= df_indicators['lower band'],1,0)
      df_buy['macd'] = np.where(df_indicators['macd'] > df_indicators['macd_signal'],1,0)
      df_buy['momentum'] = np.where(df_indicators['momentum'] > 0,1,0)
      # set up sell signals
      df_sell['sma'] = np.where(df_indicators['sma'] > df_indicators['price'],1,0) 
      df_sell['bb'] = np.where(df_indicators['price'] >= df_indicators['upper band'],1,0)
      df_sell['macd'] = np.where(df_indicators['macd'] < df_indicators['macd_signal'],1,0)
      df_sell['momentum'] = np.where(df_indicators['momentum'] < 0,1,0)
      # after getting indicators we have to standardize and normalize them
      df_trades = pd.DataFrame(index=df_indicators.index)
      df_trades[trades.columns[0]] = 0
      holding = 0
      for i in df_trades.index:
         if holding == 0:
            if df_buy.ix[i,'bb'] == 1 and  df_buy.ix[i,'momentum'] > 0 or  df_buy.ix[i,'macd'] == 1:
               print('here')
               df_trades.ix[i,0] = 1000
               holding = 1000
            elif df_sell.ix[i,'bb'] == 1 and  df_sell.ix[i,'momentum'] > 0 or  df_sell.ix[i,'macd'] == 1:
               df_trades.ix[i,0] = -1000
               holding = -1000
         elif holding == 1000:
            if df_sell.ix[i,'bb'] == 1 and  df_sell.ix[i,'momentum'] > 0 or  df_sell.ix[i,'macd'] == 1:
               df_trades.ix[i,0] = -2000
               holding = -1000
         elif holding == -1000:
            if df_buy.ix[i,'bb'] == 1 and  df_buy.ix[i,'momentum'] > 0 or  df_buy.ix[i,'macd'] == 1:
               df_trades.ix[i,0] = 2000
               holding = 1000

      return df_trades
