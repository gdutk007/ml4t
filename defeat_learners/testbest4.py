""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Test best4 data generator.  (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import DTLearner as dt  		  	   		  		 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  		  	   		  		 		  		  		    	 		 		   		 		  
from gen_data import best_4_dt, best_4_lin_reg  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  

from sklearn import tree

def compare_os_rmse_scikit(learner1, learner2, x, y):  		  	   		  		 		  		  		    	 		 		   		 		  		  	   		  		 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		  		 		  		  		    	 		 		   		 		  
    train_rows = int(math.floor(0.6 * x.shape[0]))  		  	   		  		 		  		  		    	 		 		   		 		  
    test_rows = x.shape[0] - train_rows  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		  		 		  		  		    	 		 		   		 		  
    train = np.random.choice(x.shape[0], size=train_rows, replace=False)  		  	   		  		 		  		  		    	 		 		   		 		  
    test = np.setdiff1d(np.array(range(x.shape[0])), train)  		  	   		  		 		  		  		    	 		 		   		 		  
    train_x = x[train, :]  		  	   		  		 		  		  		    	 		 		   		 		  
    train_y = y[train]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_x = x[test, :]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_y = y[test]  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # train the learners  		  	   		  		 		  		  		    	 		 		   		 		  
    learner1.add_evidence(train_x, train_y)  # train it  		  	   		  		 		  		  		    	 		 		   		 		  
    learner2 = learner2.fit(train_x, train_y)  # train it  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # evaluate learner1 out of sample  		  	   		  		 		  		  		    	 		 		   		 		  
    pred_y = learner1.query(test_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse1 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # evaluate learner2 out of sample  		  	   		  		 		  		  		    	 		 		   		 		  
    pred_y = learner2.predict(test_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse2 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  		  	   		  		 		  		  		    	 		 		   		 		  
    return rmse1, rmse2 
# compare two learners' rmse out of sample  		  	   		  		 		  		  		    	 		 		   		 		  
def compare_os_rmse(learner1, learner2, x, y):  		  	   		  		 		  		  		    	 		 		   		 		  		  	   		  		 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		  		 		  		  		    	 		 		   		 		  
    train_rows = int(math.floor(0.6 * x.shape[0]))  		  	   		  		 		  		  		    	 		 		   		 		  
    test_rows = x.shape[0] - train_rows  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		  		 		  		  		    	 		 		   		 		  
    train = np.random.choice(x.shape[0], size=train_rows, replace=False)  		  	   		  		 		  		  		    	 		 		   		 		  
    test = np.setdiff1d(np.array(range(x.shape[0])), train)  		  	   		  		 		  		  		    	 		 		   		 		  
    train_x = x[train, :]  		  	   		  		 		  		  		    	 		 		   		 		  
    train_y = y[train]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_x = x[test, :]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_y = y[test]  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # train the learners  		  	   		  		 		  		  		    	 		 		   		 		  
    learner1.add_evidence(train_x, train_y)  # train it  		  	   		  		 		  		  		    	 		 		   		 		  
    learner2.add_evidence(train_x, train_y)  # train it  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # evaluate learner1 out of sample  		  	   		  		 		  		  		    	 		 		   		 		  
    pred_y = learner1.query(test_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse1 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # evaluate learner2 out of sample  		  	   		  		 		  		  		    	 		 		   		 		  
    pred_y = learner2.query(test_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse2 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  		  	   		  		 		  		  		    	 		 		   		 		  
    return rmse1, rmse2  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def plotfigs(x,y, figname):
    import matplotlib.pyplot as plt
    x = np.arange(-200,800) # set leaf size
    fig, ax1 = plt.subplots()
    ax1.set_title("y values of function")
    ax1.set_xlabel("x axis")
    ax1.set_ylabel("y output")
    ax1.text(0.5, 0.5, 'gdutka3', transform=ax1.transAxes,
        fontsize=110, color='gray', alpha=0.5,
        ha='center', va='center', rotation=30)
    ax1.plot(x, y)
    ax1.legend(["In-Sample", "Out-Sample"])
    p = './images/'+ figname
    fig.savefig(p)
    # import pdb; pdb.set_trace()


def additional_tests(seed):
    #import pdb; pdb.set_trace()  		  	   		  		 		  		  		    	 		 		   		 		  
    # create two learners and get data  		  	   		  		 		  		  		    	 		 		   		 		  
    lrlearner = lrl.LinRegLearner(verbose=False)  		  	   		  		 		  		  		    	 		 		   		 		  
    dtlearner = dt.DTLearner(verbose=False, leaf_size=6)  		  	   		  		 		  		  		    	 		 		   		 		  
    x, y = best_4_lin_reg(seed)  		  	   		  		 		  		  		    	 		 		   		 		  
    successes = 0
    failures = 0
    # plotfigs(x,y,'fig1')	  	   		  		 		  		  		    	 		 		   		 		  
    # compare the two learners  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # share results  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("best_4_lin_reg() results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE LR    : {rmse_lr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE DT    : {rmse_dt}")  		  	   		  		 		  		  		    	 		 		   		 		  
    if rmse_lr < 0.9 * rmse_dt:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("LR < 0.9 DT:  pass")  		  	   		  		 		  		  		    	 		 		   		 		  
    else:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("LR >= 0.9 DT:   ******** FAIL ********")  		  	   		  		 		  		  		    	 		 		   		 		  
    print  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # get data that is best for a random tree  		  	   		  		 		  		  		    	 		 		   		 		  
    lrlearner = lrl.LinRegLearner(verbose=False)  		  	   		  		 		  		  		    	 		 		   		 		  
    dtlearner = dt.DTLearner(verbose=False, leaf_size=6)  		  	   		  		 		  		  		    	 		 		   		 		  
    x, y = best_4_dt(seed)  		  	   		  		 		  		  		    	 		 		   		 		  
    # plotfigs(x,y,'fig2')	  	   		  		 		  		  		    	 		 		   		 		  
    # compare the two learners  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # share results  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("best_4_dt() results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE LR    : {rmse_lr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE DT    : {rmse_dt}")  		  	   		  		 		  		  		    	 		 		   		 		  
    if rmse_dt < 0.9 * rmse_lr:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("DT < 0.9 LR:  pass")  		  	   		  		 		  		  		    	 		 		   		 		  
    else:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("DT >= 0.9 LR:  ******** FAIL ********")  		  	   		  		 		  		  		    	 		 		   		 		  
    print  		  	   		  		 		  		  		    	 		 		   		 		  


def additional_tests_scikitdt(seed):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Performs a test of your code and prints the results  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    #import pdb; pdb.set_trace()  		  	   		  		 		  		  		    	 		 		   		 		  
    # create two learners and get data  		  	   		  		 		  		  		    	 		 		   		 		  
    lrlearner = lrl.LinRegLearner(verbose=False)  		  	   		  		 		  		  		    	 		 		   		 		  
    # dtlearner = dt.DTLearner(verbose=False, leaf_size=1)  		  	   		  		 		  		  		    	 		 		   		 		  
    dtlearner = tree.DecisionTreeRegressor()
    x, y = best_4_lin_reg(seed)  		  	   		  		 		  		  		    	 		 		   		 		  
    # plotfigs(x,y,'fig1')	  	   		  		 		  		  		    	 		 		   		 		  
    # compare the two learners  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse_lr, rmse_dt = compare_os_rmse_scikit(lrlearner, dtlearner, x, y)  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # share results  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("best_4_lin_reg() results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE LR    : {rmse_lr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE DT    : {rmse_dt}")  		  	   		  		 		  		  		    	 		 		   		 		  
    if rmse_lr < 0.9 * rmse_dt:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("LR < 0.9 DT:  pass")  		  	   		  		 		  		  		    	 		 		   		 		  
    else:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("LR >= 0.9 DT:  ********* FAIL ********")  		  	   		  		 		  		  		    	 		 		   		 		  
    print  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # get data that is best for a random tree  		  	   		  		 		  		  		    	 		 		   		 		  
    lrlearner = lrl.LinRegLearner(verbose=False)  		  	   		  		 		  		  		    	 		 		   		 		  
    dtlearner = tree.DecisionTreeRegressor()
    x, y = best_4_dt(seed)  		  	   		  		 		  		  		    	 		 		   		 		  
    # plotfigs(x,y,'fig2')	  	   		  		 		  		  		    	 		 		   		 		  
    # compare the two learners  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse_lr, rmse_dt = compare_os_rmse_scikit(lrlearner, dtlearner, x, y)  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # share results  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("best_4_dt() results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE LR    : {rmse_lr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE DT    : {rmse_dt}")  		  	   		  		 		  		  		    	 		 		   		 		  
    if rmse_dt < 0.9 * rmse_lr:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("DT < 0.9 LR:  pass")  		  	   		  		 		  		  		    	 		 		   		 		  
    else:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("DT >= 0.9 LR:  fail")  		  	   		  		 		  		  		    	 		 		   		 		  
    print  		  	   		  		 		  		  		    	 		 		   		 		  


def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Performs a test of your code and prints the results  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    #import pdb; pdb.set_trace()  		  	   		  		 		  		  		    	 		 		   		 		  
    # create two learners and get data  		  	   		  		 		  		  		    	 		 		   		 		  
    lrlearner = lrl.LinRegLearner(verbose=False)  		  	   		  		 		  		  		    	 		 		   		 		  
    dtlearner = dt.DTLearner(verbose=False, leaf_size=1)  		  	   		  		 		  		  		    	 		 		   		 		  
    x, y = best_4_lin_reg()  		  	   		  		 		  		  		    	 		 		   		 		  
    # plotfigs(x,y,'fig1')	  	   		  		 		  		  		    	 		 		   		 		  
    # compare the two learners  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # share results  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("best_4_lin_reg() results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE LR    : {rmse_lr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE DT    : {rmse_dt}")  		  	   		  		 		  		  		    	 		 		   		 		  
    if rmse_lr < 0.9 * rmse_dt:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("LR < 0.9 DT:  pass")  		  	   		  		 		  		  		    	 		 		   		 		  
    else:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("LR >= 0.9 DT:  fail")  		  	   		  		 		  		  		    	 		 		   		 		  
    print  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # get data that is best for a random tree  		  	   		  		 		  		  		    	 		 		   		 		  
    lrlearner = lrl.LinRegLearner(verbose=False)  		  	   		  		 		  		  		    	 		 		   		 		  
    dtlearner = dt.DTLearner(verbose=False, leaf_size=1)  		  	   		  		 		  		  		    	 		 		   		 		  
    x, y = best_4_dt()  		  	   		  		 		  		  		    	 		 		   		 		  
    # plotfigs(x,y,'fig2')	  	   		  		 		  		  		    	 		 		   		 		  
    # compare the two learners  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # share results  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("best_4_dt() results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE LR    : {rmse_lr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE DT    : {rmse_dt}")  		  	   		  		 		  		  		    	 		 		   		 		  
    if rmse_dt < 0.9 * rmse_lr:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("DT < 0.9 LR:  pass")  		  	   		  		 		  		  		    	 		 		   		 		  
    else:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("DT >= 0.9 LR:  fail")  		  	   		  		 		  		  		    	 		 		   		 		  
    print

    print('performing additional tests with gdutka dtlearner:')  		  	   		  		 		  		  		    	 		 		   		 		  
    for i in range(20):
        # import pdb; pdb.set_trace()
        randSeed = i + np.random.randint(1000,size=1)
        additional_tests(randSeed)
    print('performing additional tests with scikit learn dtlearner:')
    for i in range(20):
        # import pdb; pdb.set_trace()
        randSeed = i + np.random.randint(1000,size=1)
        additional_tests_scikitdt(randSeed)

  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
