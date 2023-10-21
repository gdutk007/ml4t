""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt
import os
import io
from io import TextIOWrapper
import re
import numpy as np
import math
import pandas as pd
from util import get_data, plot_data


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


def compute_portvals_test(
        start_date,
        end_date,
        ticker
            ):
    start_date = dt.datetime(2008, 1, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2008, 6, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    portvals = get_data([ticker], pd.date_range(start_date, end_date))  		  	   		  		 		  		  		    	 		 		   		 		  
    portvals = portvals[[ticker]]  # remove SPY  		  	   		  		 		  		  		    	 		 		   		 		  
    rv = pd.DataFrame(index=portvals.index, data=portvals.values)  		  	   		  		 		  		  		    	 		 		   		 		  
    return rv  		  	   		  		 		  		  		    	 		 		   		 		  
    
def compute_portvals(  		  	   		  		 		  		  		    	 		 		   		 		  
    df,
    start_val=1000000
):
    ############### create prices data frame ##################
    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(df.index[0], df.index[-1])
    prices_all = get_data(['JPM'], dates)
    prices_all.drop(columns=["SPY"], inplace=True)
    # forward fill and backfill data
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    # only keep data for symbols in our portfolio
    prices_all['cash'] = 1.0
    ################ end prices dataframe ####################

    ################ create trades dataframe #################
    trades_df = pd.DataFrame(0.0, index=prices_all.index, columns=prices_all.columns)
    for index, row in df.iterrows():
        share_price = prices_all.loc[index][0]   #loc[ row.loc[ 'Date' ] ][ row.loc[ 'Symbol' ] ]
        share_amount = df.loc[index][0]          #row.loc['Shares']
        if share_amount < 0:
            trades_df.loc[index][0] = share_amount
            trades_df.loc[index][1] = share_price*share_amount*(-1)
        else:
            trades_df.loc[index][0] = share_amount
            trades_df.loc[index][1] = share_price*share_amount*(-1)
    ################# End of the trades dataframe ############

    ################  create holdings dataframe ############## 
    new_start_date = dates[0] - dt.timedelta(days=1)
    new_end_date = dates[-1] + dt.timedelta(days=1)
    starting_balance = pd.DataFrame(0, index=[new_start_date], columns=prices_all.columns)
    starting_balance['cash'] = start_val
    holdings_df = pd.concat([starting_balance, trades_df], ignore_index=False)
    for i in range(1,len(holdings_df)):
        diff = holdings_df.iloc[i] + holdings_df.iloc[i-1]
        holdings_df.iloc[i] = diff
    # break out of loop
    ############# End if holdings dataframe #################

    ############# create port vals dataframe #################
    portvals_df = prices_all*holdings_df
    portvals_df = portvals_df.sum(axis=1)
    # drop first row because it was appended just to calculate the
    # holdings
    portvals_df.drop(index=[new_start_date], axis=0, inplace=True)
    rv = pd.DataFrame(index=portvals_df.index, data=portvals_df.values)
    return rv
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def author(): 
  return 'gdutka3'


# def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
#     """  		  	   		  		 		  		  		    	 		 		   		 		  
#     Helper function to test code  		  	   		  		 		  		  		    	 		 		   		 		  
#     """  		  	   		  		 		  		  		    	 		 		   		 		  
#     # this is a helper function you can use to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
#     # note that during autograding his function will not be called.  		  	   		  		 		  		  		    	 		 		   		 		  
#     # Define input parameters  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
#     #of = "./orders/orders-02.csv"  		  	   		  		 		  		  		    	 		 		   		 		  
#     of = "./orders/orders-11.csv"
#     sv = 1000000
#     commission = 0
#     impact = 0
# if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
#     test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
