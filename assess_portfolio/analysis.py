"""Analyze a portfolio.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2017, Georgia Tech Research Corporation  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332-0415  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		  		 		  		  		    	 		 		   		 		  
import math

#             cum_ret=0.198105963655,  		  	   		  		 		  		  		    	 		 		   		 		  
#             avg_daily_ret=0.000763106152672,  		  	   		  		 		  		  		    	 		 		   		 		  
#             sharpe_ratio=1.30798398744, 

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
  		  	   		  		 		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		  	   		  		 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		  		 		  		  		    	 		 		   		 		  
def assess_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		  		 		  		  		    	 		 		   		 		  
    allocs=[0.1, 0.2, 0.3, 0.4],  		  	   		  		 		  		  		    	 		 		   		 		  
    sv=1000000,  		  	   		  		 		  		  		    	 		 		   		 		  
    rfr=0.0,  		  	   		  		 		  		  		    	 		 		   		 		  
    sf=252.0,  		  	   		  		 		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		  		 		  		  		    	 		 		   		 		  
):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Estimate a set of test points given the model we built.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param syms: A list of 2 or more symbols that make up the portfolio (note that your code should support any symbol in the data directory)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		  		 		  		  		    	 		 		   		 		  
    :param allocs:  A list of 2 or more allocations to the stocks, must sum to 1.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type allocs: list  		  	   		  		 		  		  		    	 		 		   		 		  
    :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :param rfr: The risk free return per sample period that does not change for the entire date range (a single number, not an array)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type rfr: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param sf: Sampling frequency per year  		  	   		  		 		  		  		    	 		 		   		 		  
    :type sf: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  		 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: A tuple containing the cumulative return, average daily returns,  		  	   		  		 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, Sharpe ratio and end value  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		  		 		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		  		 		  		  		    	 		 		   		 		  

    # ffill and bfill data
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)

    prices = prices_all[syms]  # only portfolio symbols  		  	   		  		 		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		  		 		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
    
    prices_np = prices.to_numpy()
	  	   		  		 		  		  		    	 		 		   		 		  
    # Get daily portfolio value  		  	   		  		 		  		  		    	 		 		   		 		  
    port_val = prices_SPY  # add code here to compute daily portfolio values  		  	   		  		 		  		  		    	 		 		   		 		  	  	   		  		 		  		  		    	 		 		   		 		  
    # Get portfolio statistics (note: std_daily_ret = volatility)  		  	   		  		 		  		  		    	 		 		   		 		  
    cr, adr, sddr, sr = calculate_stats(allocs,prices_np)

    #import pdb; pdb.set_trace() 		  		 		  		  		    	 		 		   		 		  
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		  		 		  		  		    	 		 		   		 		  
    if gen_plot:  		  	   		  		 		  		  		    	 		 		   		 		  
        # add code to plot here  		  	   		  		 		  		  		    	 		 		   		 		  
        df_temp = pd.concat(  		  	   		  		 		  		  		    	 		 		   		 		  
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1  		  	   		  		 		  		  		    	 		 		   		 		  
        )  		  	   		  		 		  		  		    	 		 		   		 		  
        pass  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Add code here to properly compute end value  		  	   		  		 		  		  		    	 		 		   		 		  
    ev = sv  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    return cr, adr, sddr, sr, ev  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
# inputs=dict(  		  	   		  		 		  		  		    	 		 		   		 		  
#             start_date="2010-01-01",  		  	   		  		 		  		  		    	 		 		   		 		  
#             end_date="2010-12-31",  		  	   		  		 		  		  		    	 		 		   		 		  
#             symbol_allocs=OrderedDict(  		  	   		  		 		  		  		    	 		 		   		 		  
#                 [("AXP", 0.0), ("HPQ", 0.0), ("IBM", 0.0), ("HNZ", 1.0)]  		  	   		  		 		  		  		    	 		 		   		 		  
#             ),  		  	   		  		 		  		  		    	 		 		   		 		  
#             start_val=1000000,  		  	   		  		 		  		  		    	 		 		   		 		  
#         ),  		  	   		  		 		  		  		    	 		 		   		 		  
#         outputs=dict(  		  	   		  		 		  		  		    	 		 		   		 		  
#             cum_ret=0.198105963655,  		  	   		  		 		  		  		    	 		 		   		 		  
#             avg_daily_ret=0.000763106152672,  		  	   		  		 		  		  		    	 		 		   		 		  
#             sharpe_ratio=1.30798398744,  		  	   		  		 		  		  		    	 		 		   		 		  
#         )

def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Performs a test of your code and prints the results  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # This code WILL NOT be tested by the auto grader  		  	   		  		 		  		  		    	 		 		   		 		  
    # It is only here to help you set up and test your code  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		  		 		  		  		    	 		 		   		 		  
    # Note that ALL of these values will be set to different values by  		  	   		  		 		  		  		    	 		 		   		 		  
    # the autograder!  		  	   		  		 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2010, 1, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2010, 12, 31)  		  	   		  		 		  		  		    	 		 		   		 		  
    symbols = ["AXP", "HPQ", "IBM", "HNZ"]  		  	   		  		 		  		  		    	 		 		   		 		  
    allocations = [0.0, 0.0, 0.0, 1.0]  		  	   		  		 		  		  		    	 		 		   		 		  
    start_val = 1000000		  	   		  		 		  		  		    	 		 		   		 		  
    risk_free_rate = 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    sample_freq = 252 		  	   		  		 		  		  		    	 		 		   		 		  		 		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    cr, adr, sddr, sr, ev = assess_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
        sd=start_date,  		  	   		  		 		  		  		    	 		 		   		 		  
        ed=end_date,  		  	   		  		 		  		  		    	 		 		   		 		  
        syms=symbols,  		  	   		  		 		  		  		    	 		 		   		 		  
        allocs=allocations,  		  	   		  		 		  		  		    	 		 		   		 		  
        sv=start_val,  		  	   		  		 		  		  		    	 		 		   		 		  
        gen_plot=False,  		  	   		  		 		  		  		    	 		 		   		 		  
    )  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Allocations: {allocations}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
