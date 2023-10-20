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
    tradesDf,
    start_val=1000000,  		  	   		  		 		  		  		    	 		 		   		 		  
    # commission=9.95,  		  	   		  		 		  		  		    	 		 		   		 		  
    # impact=0.005,  		  	   		  		 		  		  		    	 		 		   		 		  
):
    # if isinstance(orders_file, str):
    #     df = pd.read_csv(orders_file)
    # elif isinstance(orders_file, TextIOWrapper):
    #     data = io.StringIO(orders_file.read())
    #     print(data)
    #     df = pd.read_csv(data)
    # else:
    #     print("Error occurred, orders_file is not file-object or string path.")
    #     exit(0)
    # sort by date
    df = df.sort_values(by='Date',ascending=True)

    ############### create prices data frame ##################
    start_date = df['Date'].values[0]
    end_date = df['Date'].values[-1]
    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(df['Symbol'].unique(), dates)
    # forward fill and backfill data
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    # only keep data for symbols in our portfolio
    prices_all = prices_all[prices_all.columns.intersection( df['Symbol'].unique() )]
    prices_all['cash'] = 1.0
    prices_all.drop_duplicates(keep='first',inplace=True)
    ################ end prices dataframe ####################

    ################ create trades dataframe #################
    trades_df = pd.DataFrame(0.0, index=prices_all.index, columns=prices_all.columns)
    for index, row in df.iterrows():
        share_price = prices_all.loc[ row.loc[ 'Date' ] ][ row.loc[ 'Symbol' ] ]
        share_amount = row.loc['Shares']
        buy_sell = 1.0
        if row.loc['Order'] == 'SELL':
            buy_sell = -1.0
        share_price = share_price + share_price*impact*buy_sell
        temp =  trades_df.loc[ row.loc['Date'] ]
        temp[ row.loc['Symbol'] ] += buy_sell*share_amount
        temp[ 'cash' ] += buy_sell*share_price*share_amount*(-1.0) - commission
        trades_df.loc[ row.loc['Date'] ] = temp
    # break loop
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
    ############# Ebd if holdings dataframe #################

    ############# create port vals dataframe #################
    portvals_df = prices_all*holdings_df
    portvals_df = portvals_df.sum(axis=1)
    # drop first row because it was appended just to calculate the
    # holdings
    portvals_df.drop(index=[new_start_date], axis=0, inplace=True)
    rv = pd.DataFrame(index=portvals_df.index, data=portvals_df.values)
    # import pdb;pdb.set_trace()
    return rv
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def author(): 
  return 'gdutka3'


def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		  		 		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    #of = "./orders/orders-02.csv"  		  	   		  		 		  		  		    	 		 		   		 		  
    of = "./orders/orders-11.csv"
    sv = 1000000
    commission = 0
    impact = 0
    # Process orders
    portvals_fund = compute_portvals(orders_file=of, start_val=sv)#,commission=9.95, impact=impact)
    #start_date = "2011-01-10"
    #end_date = "2011-12-21"
    #portvals = compute_portvals_test(start_date,end_date,"SPY")
    #portvals_test = compute_portvals_test(start_date,end_date,"IBM")
    # import pdb; pdb.set_trace()
    if isinstance(portvals_fund, pd.DataFrame):  		  	   		  		 		  		  		    	 		 		   		 		  
        #portvals = portvals[portvals.columns[0]]  # just get the first column
        #portvals_test =  portvals_test[portvals_test.columns[0] ]
        portvals_fund =  portvals_fund[portvals_fund.columns[0] ]
    else:  		  	   		  		 		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    
    # Get portfolio stats  		  	   		  		 		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		  		 		  		  		    	 		 		   		 		  
    
    #start_date = dt.datetime(2008, 1, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    #end_date = dt.datetime(2008, 6, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portvals_test.to_numpy())
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = calculate_stats(portvals.to_numpy())
  		  	   		  		 		  		  		    	 		 		   		 		  
    # # Compare portfolio against $SPX  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Date Range: {start_date} to {end_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print()  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print()  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print()  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print()  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print()  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		  		 		  		  		    	 		 		   		 		  

    # import pdb;pdb.set_trace()
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_stats(portvals_fund.to_numpy())
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals_fund[-1]}")  		  	   		  		 		  		  		    	 		 		   		 		  



  		  	   		  		 		  		  		    	 		 		   		 		  
# if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
#     test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
