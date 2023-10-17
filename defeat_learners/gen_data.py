""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    	 		 		   		 		  
import math  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		  		 		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		  		 		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		  		 		  		  		    	 		 		   		 		  
    np.random.seed(seed)  		  	   		  		 		  		  		    	 		 		   		 		  
    # generate two feature matrix 2 cols, 100 datapoints
    x = np.random.random(size=(1000,4)) * 200 - 100
    # x = np.zeros((1000,1))# * 200 + 100
    # for i in range(1000):
    #     x[i] = i
    # generate y as function of x
    # y = x[:,0] + np.sin(x[:,1])
    # y = np.sin(x[:,0])
    y = 2*x[:,0] + 3*x[:,1] + 3*x[:,0] + 4*x[:,1]
    return x, y  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		  		 		  		  		    	 		 		   		 		  
    np.random.seed(seed)  		  	   		  		 		  		  		    	 		 		   		 		  
    # generate two feature matrix 2 cols, 100 datapoints
    x = np.random.random(size=(1000,2)) * 200 - 100
    # generate y as function of x
    #x = np.zeros((1000,1))# * 200 + 100
    #x = np.zeros((1000,1))# * 200 + 100
    #for i in range(1000):
    #    x[i] = i
    y = np.sum(np.power(x,2),axis=1)
    # import pdb;pdb.set_trace()
    return x, y  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def author():  		  	   		  		 		  		  		    	 		 		   		 		  
    return "gdutka3"  # Change this to your user ID  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("they call me Tim.")  		  	   		  		 		  		  		    	 		 		   		 		  
