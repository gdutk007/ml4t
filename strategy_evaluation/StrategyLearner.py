""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Gianfranco Dutka
GT User ID: gdutka3
GT ID: 903890585
"""

import datetime as dt
import random

import pandas as pd
import util as ut
import RTLearner as rt
import BagLearner as ln
from indicators import getIndicators
import numpy as np
import pdb;


class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner =  ln.BagLearner(learner=rt.RTLearner, 
                                kwargs={"leaf_size":5}, bags=13, boost=False, verbose=False)
        
        def author( self ): 
            return 'gdutka3'

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int
        """

        # add your code to do learning here  		  	   		  		 		  		  		    	 		 		   		 		  

        # example usage of the old backward compatible util function  		  	   		  		 		  		  		    	 		 		   		 		  
        # syms = [symbol]  		  	   		  		 		  		  		    	 		 		   		 		  
        # dates = pd.date_range(sd, ed)  		  	   		  		 		  		  		    	 		 		   		 		  
        # prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		  		 		  		  		    	 		 		   		 		  
        # prices = prices_all[syms]  # only portfolio symbols  		  	   		  		 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        # forward fill and backfill data
        prices_all.fillna(method='ffill', inplace=True)
        prices_all.fillna(method='bfill', inplace=True)
        # only keep data for symbols in our portfolio
        prices_all.drop(columns=["SPY"], inplace=True)
        prices = prices_all.copy()
        df_indicators = getIndicators(prices, sd, ed)
        df_indicators['macd_diff'] = df_indicators['macd'] - df_indicators['macd_signal']
        df_indicators['macd_diff_shift'] = df_indicators['macd_diff'].shift(1) # get crossover
        df_signals = pd.DataFrame(index=df_indicators.index)
        #df_signals['bbp'] = [1 if x <= -0.03 else -1 if x >= 1.03 else 0 for x in df_indicators['bbp']]
        #df_signals['sma_ratio'] = [1 if x <= 0.955 else -1 if x >= 1.055 else 0 for x in df_indicators['price_sma_ratio']]
        df_signals['bbp'] = df_indicators['bbp']
        df_signals['sma_ratio'] = df_indicators['price_sma_ratio']
        df_signals['macd'] = 0
        for index,row in df_indicators.iterrows():
            if row['macd_diff'] > 0.0 and row['macd_diff_shift'] < 0.0:
                df_signals.at[index,'macd'] = 1
            elif row['macd_diff'] < 0.0 and row['macd_diff_shift'] > 0.0:
                df_signals.at[index,'macd'] = -1

        #pdb.set_trace()
        df_signals['return'] = (prices_all.shift(-15)/prices_all)-1.0
        ybuy = 0.07 + 0.7*self.impact
        ysell = -0.07 + 0.7*self.impact
        df_signals['target'] = [1 if x > ybuy else -1 if x < ysell else 0 for x in df_signals['return']]
        train_x = df_indicators[['bbp','price_sma_ratio','macd_diff']]
        train_y = df_signals['target'] 
        #pdb.set_trace()
        self.learner.add_evidence(train_x.to_numpy(),train_y.to_numpy())
        # pdb.set_trace()
        # print('things did not crash!')

    # this method should use the existing policy and test it against new data  		  	   		  		 		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		  		 		  		  		    	 		 		   		 		  
        self,  		  	   		  		 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		  		 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		  		 		  		  		    	 		 		   		 		  
    ):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		  		 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        # getting pricess, then backfill and front fill
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices_all.fillna(method='ffill', inplace=True)
        prices_all.fillna(method='bfill', inplace=True)

        # creating the trades dataframe and setting everything to 0
        trades = prices_all[[symbol,]]  # only portfolio symbols  		  	   		  		 		  		  		    	 		 		   		 		  
        prices_all.drop(columns=["SPY"], inplace=True)
        trades.values[:, :] = 0

        # getting indicators dataframe, then we're going to query
        df_indicators = getIndicators(prices_all.copy(), sd, ed)
        df_indicators['macd_diff'] = df_indicators['macd'] - df_indicators['macd_signal']
        #pdb.set_trace()
        pred_y = self.learner.query(df_indicators[['bbp','price_sma_ratio','macd_diff']].to_numpy())
        # pdb.set_trace()
        
        holding = 0
        j = 0
        for i in trades.index:
            if holding == 0:
                if pred_y[j] > 0 :
                    trades.ix[i,0] = 1000
                    holding = 1000
                elif pred_y[j] < 0:
                    trades.ix[i,0] = -1000
                    holding = -1000
            elif holding == 1000:
                if pred_y[j] < 0:
                    trades.ix[i,0] = -2000
                    holding = -1000
            elif holding == -1000:
                if pred_y[j] > 0:
                    trades.ix[i,0] = 2000
                    holding = 1000
            j += 1
        # if self.verbose:  		  	   		  		 		  		  		    	 		 		   		 		  
        #     print(type(trades))  # it better be a DataFrame!  		  	   		  		 		  		  		    	 		 		   		 		  
        # if self.verbose:  		  	   		  		 		  		  		    	 		 		   		 		  
        #     print(trades)  		  	   		  		 		  		  		    	 		 		   		 		  
        # if self.verbose:  		  	   		  		 		  		  		    	 		 		   		 		  
        #     print(prices_all)  		  	   		  		 		  		  		    	 		 		   		 		  
        # pdb.set_trace()
        return trades  		  	   		  		 		  		  		    	 		 		   		 		  


if __name__ == "__main__":
    print("One does not simply think up a strategy")
