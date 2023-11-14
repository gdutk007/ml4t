""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
class RTLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This is a Decision Tree learner
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    Tree = np.array([])  		 
    leaf_size = 1
    def __init__(self,leaf_size=1 ,verbose=False):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """ 
        self.leaf_size = leaf_size
        pass  # move along, these aren't the drones you're looking for  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        return "gdutka3" # replace with gatech username
  		  	   		  		 		  		  		    	 		 		   		 		  
    def add_evidence(self, x, y):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		  		 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		  		 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        y = y.reshape(y.shape[0],1)
        data = np.hstack((x,y))
        # Build model for decision trees   		  	   		  		 		  		  		    	 		 		   		 		  
        tree = self.build_tree(data)
        self.Tree = tree

    def traverse_tree(self, idx, point):
        if len(self.Tree) == 0:
            return 0.0
        if idx  > len(self.Tree):
            return self.Tree[len(self.Tree)-1][1]
        if self.Tree[idx][0] == -1.0:
            return self.Tree[idx][1]
        
        predVal = 0.0
        # extract feature index from 0th index
        feature = int(self.Tree[idx][0])
        
        # compare extracted feature with Tree's splitval
        if point[feature] <= self.Tree[idx][1]:
           predVal = self.traverse_tree( int(idx + self.Tree[idx][2]), point) 
        
        # compare extracted feature with Tree's splitval
        if point[feature] > self.Tree[idx][1]:
           predVal = self.traverse_tree( int(idx + self.Tree[idx][3]), point) 

        return predVal


    def query(self, points):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        predVals = np.zeros(len(points))
        for i in range(len(points)):
            predVals[i] = self.traverse_tree(0,points[i])
        return predVals

    def get_mode(self, data):
        count = {}
        for i in data:
            if i in count:
                count[i]+= 1
            else:
                count[i] = 1
        return max(count, key=count.get)

    def build_tree(self, data):
        if data.shape[0] <= self.leaf_size:
            if data.shape[0] == 0:
                return np.array([-1, 0, -1,-1])
            if data.shape[0] > 1:
                #mean = np.mean(data[:,-1])
                mode = self.get_mode(data[:,-1])
                return np.array([-1, mode, -1,-1])
            return np.array([-1, data[0][-1], -1,-1])
        elif (data[:,-1] == data[0][-1]).all():
            # values are all the same so we return y element
            return np.array([-1, data[0][-1], -1,-1])
        else:
            # determine best feature
            features = data[:,0:-1]
            y = data[:,-1]
            idx = np.random.choice(data.shape[1] - 1 )
            # now get split val
            splitVal = np.nanmedian(data[:,idx])
            #import pdb; pdb.set_trace()
            splitData = data[data[:,idx] <= splitVal]
            if np.array_equal(splitData , data):
                half_rows = int(0.5 * data.shape[0])
                splitData =  data[:half_rows]

            leftTree = self.build_tree(splitData)
            
            splitData = data[data[:,idx] > splitVal]
            if np.array_equal(splitData,data ):
                half_rows = int(0.5 * data.shape[0])
                splitData =  data[:half_rows]
            rightTree = self.build_tree(splitData)

            if leftTree.ndim == 1:
                root = np.array([idx,splitVal,1,2])
            else:    
                root = np.array([idx,splitVal,1,leftTree.shape[0]+1])
            return np.vstack([root,leftTree,rightTree])
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		  		 		  		  		    	 		 		   		 		  
