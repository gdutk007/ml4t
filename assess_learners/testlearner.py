""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import math  		  	   		  		 		  		  		    	 		 		   		 		  
import sys  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl 
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as Bl
import InsaneLearner as Isl
import matplotlib.pyplot as plt
import base64
from io import BytesIO		 		  		  		    	 		 		   		 		  
import time


def Experiment1(train_x, train_y, test_x, test_y):
    max_leaf_size = 80
    rmse_in_sample = np.zeros(max_leaf_size, dtype=float)
    rmse_out_sample = np.zeros(max_leaf_size, dtype=float)
    corr_in_sample = np.zeros(max_leaf_size, dtype=float)
    corr_out_sample = np.zeros(max_leaf_size, dtype=float)
    for i in range(1, max_leaf_size+1):
        # create a learner and train it  		  	   		  		 		  		  		    	 		 		   		 		  
        learner = dtl.DTLearner(leaf_size=i,verbose=False)
        learner.add_evidence(train_x, train_y)  # train it

        # evaluate in sample  		  	   		  		 		  		  		    	 		 		   		 		  
        pred_y = learner.query(train_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
        rmse_in_sample[i-1] =  math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        corr_in_sample[i-1] =  np.corrcoef(pred_y, y=train_y)[0,1]

        # evaluate out of sample  		  	   		  		 		  		  		    	 		 		   		 		  
        pred_y = learner.query(test_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
        rmse_out_sample[i-1] =  math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        corr_out_sample[i-1] =  np.corrcoef(pred_y, y=test_y)[0,1]

    x = np.arange(1,max_leaf_size+1) # set leaf size
    fig, ax1 = plt.subplots()
    ax1.set_title("In-sample and Out-sample RMSE vs Leaf Size")
    ax1.set_xlabel("Leaf Size")
    ax1.set_ylabel("RMSE")
    ax1.text(0.5, 0.5, 'gdutka3', transform=ax1.transAxes,
        fontsize=110, color='gray', alpha=0.5,
        ha='center', va='center', rotation=30)
    ax1.plot(x, rmse_in_sample)
    ax1.plot(x, rmse_out_sample)
    ax1.legend(["In-Sample", "Out-Sample"])
    tempfile = BytesIO()
    fig.savefig(tempfile,format="png")
    fig.savefig('./images/fig1.png')
    # encoded = base64.b64encode(tempfile.getvalue()).decode('utf-8')
    # html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br />'

def Experiment2(train_x, train_y, test_x, test_y):
    max_leaf_size = 25
    rmse_in_sample = np.zeros(max_leaf_size, dtype=float)
    rmse_out_sample = np.zeros(max_leaf_size, dtype=float)
    corr_in_sample = np.zeros(max_leaf_size, dtype=float)
    corr_out_sample = np.zeros(max_leaf_size, dtype=float)
    for i in range(1, max_leaf_size+1):
        # create a learner and train it
        if i % 10 == 0:
            print( "leaf size progress:",i)	  	   		  		 		  		  
          		  	   		  		 		  		  		    	 		 		   		 		  
        learner = Bl.BagLearner(learner=dtl.DTLearner, 
                                kwargs={"leaf_size":i}, bags=25, boost=False, verbose=False )
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample  		  	   		  		 		  		  		    	 		 		   		 		  
        pred_y = learner.query(train_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
        rmse_in_sample[i-1] =  math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        corr_in_sample[i-1] =  np.corrcoef(pred_y, y=train_y)[0,1]

        # evaluate out of sample  		  	   		  		 		  		  		    	 		 		   		 		  
        pred_y = learner.query(test_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
        rmse_out_sample[i-1] =  math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        corr_out_sample[i-1] =  np.corrcoef(pred_y, y=test_y)[0,1]

    x = np.arange(1,max_leaf_size+1) # set leaf size
    fig1, ax2 = plt.subplots()
    ax2.set_title("In-sample and Out-sample RMSE vs Leaf Size")
    ax2.set_xlabel("Leaf Size")
    ax2.set_ylabel("RMSE")
    ax2.text(0.5, 0.5, 'gdutka3', transform=ax2.transAxes,
        fontsize=110, color='gray', alpha=0.5,
        ha='center', va='center', rotation=30)
    ax2.plot(x,rmse_in_sample)
    ax2.plot(x,rmse_out_sample)
    ax2.legend(["In-Sample", "Out-Sample"])

    tempfile = BytesIO()
    fig1.savefig(tempfile,format="png")
    fig1.savefig('./images/fig2.png')
    # encoded = base64.b64encode(tempfile.getvalue()).decode('utf-8')
    # html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br />'

def Experiment3(train_x, train_y, test_x, test_y):
    
    max_leaf_size = 30
    
    mae_in_sample_dtl = np.zeros(max_leaf_size, dtype=float)
    
    
    mae_in_sample_rtl = np.zeros(max_leaf_size, dtype=float)
    
    mae_out_sample_dtl = np.zeros(max_leaf_size, dtype=float)
    mae_out_sample_rtl = np.zeros(max_leaf_size, dtype=float)
    
    for i in range(1, max_leaf_size+1):
        learner1 = dtl.DTLearner(leaf_size=i,verbose=False)
        learner2 = rtl.RTLearner(leaf_size=i,verbose=False)
        
        learner1.add_evidence(train_x, train_y)  # train it
        learner2.add_evidence(train_x, train_y)  # train it
        # evaluate in sample  		  	   		  		 		  		  		    	 		 		   		 		  
        pred_y_learner1 = learner1.query(train_x)  # get the predictions
        pred_y_learner2 = learner2.query(train_x)
        pred_y_learner1 = np.mean(np.abs(train_y - pred_y_learner1)) * 100
        pred_y_learner2 = np.mean(np.abs(train_y - pred_y_learner2)) * 100
        mae_in_sample_dtl[i-1] = pred_y_learner1
        mae_in_sample_rtl[i-1] = pred_y_learner2

        mae_out_sample_dtl = 0
        mae_out_sample_rtl = 0
   

    x = np.arange(1,max_leaf_size+1) # set leaf size
    fig3, ax3 = plt.subplots()
    ax3.set_title("In-sample DTL and RTL MAE vs Leaf Size")
    ax3.set_xlabel("Leaf Size")
    ax3.set_ylabel("MAE")
    ax3.text(0.5, 0.5, 'gdutka3', transform=ax3.transAxes,
        fontsize=110, color='gray', alpha=0.5,
        ha='center', va='center', rotation=30)
    ax3.plot(x,mae_in_sample_dtl)
    ax3.plot(x,mae_in_sample_rtl)
    ax3.legend(["DTL MAE", "RTL MAE"])
    tempfile = BytesIO()
    fig3.savefig(tempfile,format="png")
    fig3.savefig('./images/fig3.png')
    timestamp_dtl = []
    timestamp_rtl = []
    for i in range(1,20):
        start = time.time()
        learner = dtl.DTLearner(leaf_size=i,verbose=False)
        learner.add_evidence(train_x, train_y)  # train it  		    	 		 		   		 		  
        end = time.time()
        timestamp_dtl.append( end - start )

    for i in range(1,20):
        start = time.time()
        learner = rtl.RTLearner(leaf_size=i,verbose=False)
        learner.add_evidence(train_x, train_y)  # train it  		    	 		 		   		 		  
        end = time.time()
        timestamp_rtl.append( end - start )
    
    x = np.arange(1,20) # set leaf size
    fig4, ax4 = plt.subplots()
    ax4.set_title("DTL and RTL Time to Train vs Leaf Size")
    ax4.set_xlabel("Leaf Size")
    ax4.set_ylabel("sec")
    ax4.text(0.5, 0.5, 'gdutka3', transform=ax4.transAxes,
        fontsize=110, color='gray', alpha=0.5,
        ha='center', va='center', rotation=30)
    ax4.plot(x,timestamp_dtl)
    ax4.plot(x,timestamp_rtl)
    ax4.legend(["DTL time to train", "RTL time to train"])
    tempfile = BytesIO()
    fig4.savefig(tempfile,format="png")
    fig4.savefig('./images/fig4.png')


if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		  		 		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		  		 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])		  	   		  		 		  		  		    	 		 		   		 		  
    data = np.genfromtxt( sys.argv[1], delimiter=',' )
    data = data[1:, 1:]
    np.random.seed(903890585)
  		  	   		  		 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		  		 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		  		 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		  		 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  

    Experiment1(train_x, train_y, test_x, test_y)
    Experiment2(train_x, train_y, test_x, test_y)
    Experiment3(train_x, train_y, test_x, test_y)

    # insane learner
    # print('---------------------- insane learner ---------------------')
    # learner = Isl.InsaneLearner()
    # learner.add_evidence(train_x, train_y)  # train it
    # pred_y = learner.query(train_x)
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"RMSE: {rmse}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # c = np.corrcoef(pred_y, y=train_y)  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(f"corr: {c[0,1]}")  		  	   		  		 		  		  		    	 		 		   		 		  
    # print('---------------------- end insane learner ---------------------')

    # DtLearnerTests(train_x, train_y, test_x, test_y)

