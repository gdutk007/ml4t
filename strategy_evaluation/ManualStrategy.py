import datetime as dt
import random
import pandas as pd
import util as ut
from indicators import getIndicators
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
      trades[:] = 0.0
      trades.iloc[0][0] = 1000
      
      df_indicators = getIndicators(prices, sd, ed)
      # after getting indicators we have to standardize and normalize them
      
      pdb.set_trace()
      #return df_indicators
