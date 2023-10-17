""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		 
import scipy.optimize as spo
import math
  		  	   		  		 		  		  		    	 		 		   		 		  

def calculate_stats(allocs,prices_np):
    normed = prices_np/prices_np[0]
    alloced = normed*allocs
    port_vals = alloced.sum(axis=1)
    cum_ret = port_vals[-1]/port_vals[0] - 1
    daily_ret = (port_vals[1:]/port_vals[:-1]) - 1 
    adr = daily_ret.mean(axis=0)
    sddr = daily_ret.std(ddof=1)
    sr =  math.sqrt(252) * adr/sddr
    # import pdb; pdb.set_trace()
    return cum_ret, adr, sddr, sr

# Function to optimize
def objective_function(allocs, prices_np):
    cum_ret, adr, sddr, sr = calculate_stats(allocs, prices_np)
    return -1*sr

# This is the function that will be tested by the autograder  		  	   		  		 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		  		 		  		  		    	 		 		   		 		  
def optimize_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD"],  		  	   		  		 		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		  		 		  		  		    	 		 		   		 		  
):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  		 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  		 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  		 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  		 		  		  		    	 		 		   		 		  
    statistics.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  		 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		  		 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  		 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  		 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		  		 		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates) # automatically adds SPY

    # ffill and bfill data
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"] # only SPY, for comparison later
    
    prices_np = prices.to_numpy()

    # normalizing prices for each stock
    prices_np_normed = prices_np/prices_np[0]
    # tuples with the min and max bounds for each row
    bnds = [(0,1)]*len(prices_np_normed[0])
    # constraints so sum is equal to 1
    constraints = ( { 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) } )   

    # find the allocations for the optimal portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case  		  	   		  		 		  		  		    	 		 		   		 		  
    
    num_symbols = len(prices_np_normed[0])
    allocsList = [1/num_symbols]*num_symbols
    allocs = np.asarray( allocsList )

    res = spo.minimize(objective_function,
                       allocs.flatten(),
                       args=prices_np,
                       method='SLSQP',
                       bounds=bnds,
                       constraints=constraints
                       )
    allocs = np.asarray(res.x)
    # the results should return the list of allocations 		  		  		    	 		 		   		 		  
    cr, adr, sddr, sr = calculate_stats(allocs,prices_np)
    
    # Get daily portfolio value  
    
    normed_prices = prices/prices.loc[prices.index[0].strftime('%Y-%m-%d')]
    alloced = normed_prices*allocs
    port_val  = alloced.sum(axis=1)
    spy_val = prices_SPY/prices_SPY.loc[prices_SPY.index[0].strftime('%Y-%m-%d')]
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		  		 		  		  		    	 		 		   		 		  
    if gen_plot:  		  	   		  		 		  		  		    	 		 		   		 		  
        # add code to plot here
        df_temp = pd.concat(  		  	   		  		 		  		  		    	 		 		   		 		  
            [port_val, spy_val], keys=["Portfolio", "SPY"], axis=1  		  	   		  		 		  		  		    	 		 		   		 		  
        )
        plot = df_temp.plot(title="Daily Portfolio value vs SPY")
        plot.set_xlabel("Date")
        plot.set_ylabel("Normalized Stock price")
        fig = plot.get_figure()
        fig.savefig('./images/optimize_portfolio.png')
        #plot.set_xlim([0,300])
        #import pdb; pdb.set_trace()
        # plt.show()   		  		 		  		  		    	 		 		   		 		  
        pass  

    return allocs, cr, adr, sddr, sr		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
 		  		 		  		  		    	 		 		   		 		  
def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  		 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 6, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2009, 6, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    symbols = ["IBM", "X", "GLD", "JPM"]
    #symbols = ["AAPL", "GLD", "GOOG"] # ["IBM", "X", "GLD", "JPM"]	  	   		  		 		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True#False  		  	   		  		 		  		  		    	 		 		   		 		  
    )  	   		  		 		  		  		    	 		 		   		 		    	   		  		 		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		  		 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
