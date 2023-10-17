""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    	 		 		   		 		  
import warnings  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
class DTLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    Tree = []
    leaf_size = 1
    def __init__(self,leaf_size=2 ,verbose=False):  		  	   		  		 		  		  		    	 		 		   		 		  
        self.leaf_size = leaf_size
        pass  # move along, these aren't the drones you're looking for  		  	   		  		 		  		  		    	 		 		   		 		  
	  		 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		  		 		  		  		    	 		 		   		 		  
        return "gdutka3" # replace with gatech username
  		  	   		  		 		  		  		    	 		 		   		 		  
    def add_evidence(self, x, y):  		  	   		  		 		  		  		    	 		 		   		 		  
        y = y.reshape(-1,1)
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
        predVals = np.zeros(len(points))
        for i in range(len(points)):
            predVals[i] = self.traverse_tree(0,points[i])
        return predVals
    
    def build_tree(self, data):
        if data.shape[0] <= self.leaf_size:
            if data.shape[0] == 0:
                return np.array([-1, 0, -1,-1])
            if data.shape[0] > 1:
                return np.array([-1, np.mean(data[:,-1]), -1,-1])
            return np.array([-1, data[0][-1], -1,-1])
        elif (data[:,-1] == data[0][-1]).all():
            return np.array([-1, data[0][-1], -1,-1])
        else:
            # determine best feature
            features = data[:,0:-1]
            y = data[:,-1]
            coefs = self.get_feature(features,y)
            idx,coef = coefs.argmax(),coefs.max()
            # now get split val
            splitVal = np.median(data[:,idx])
            splitData = data[data[:,idx] <= splitVal]
            
            if np.array_equal(splitData , data):
                return np.array([-1, np.mean(data[:,-1]), -1,-1])

            leftTree = self.build_tree(splitData)
            
            splitData = data[data[:,idx] > splitVal]
            
            if np.array_equal(splitData , data):
                return np.array([-1, np.mean(data[:,-1]), -1,-1])

            rightTree = self.build_tree(splitData)

            if leftTree.ndim == 1:
                root = np.array([idx,splitVal,1,2])
            else:  
                root = np.array([idx,splitVal,1,leftTree.shape[0]+1])
            return np.vstack([root,leftTree,rightTree])
  		  	   		  		 		  		  		    	 		 		   		 		  
    def get_feature(self, data, y ):
        coef = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            matrix = np.corrcoef(data[:,i],y.flatten())
            coef[i] = abs(matrix[1][0])
        return coef
  		  	   		  		 		  		  		    	 		 		   		 		  

if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		  		 		  		  		    	 		 		   		 		  
