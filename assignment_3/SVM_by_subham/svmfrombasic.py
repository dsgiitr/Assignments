#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:37:01 2019

@author: subham
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

(X,y) =  make_blobs(n_samples=50,n_features=2,centers=2,cluster_std=1.05,random_state=11)

plt.scatter(X[:,0],X[:,1],marker='*',c=y)
plt.axis()
plt.show()

postiveX=[]
negativeX=[]
for i,v in enumerate(y):
    if v==0:
        negativeX.append(X[i])
    else:
        postiveX.append(X[i])

#data dictionary
data_dict = {-1:np.array(negativeX), 1:np.array(postiveX)}

#class for implementing SVM
class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.axis = self.fig.add_subplot(1,1,1)
        
    #Method for training the dataset
    def fit(self, data):
        self.data = data
        #opt_dict is a dictionary of { ||w|| : [w,b]}
        opt_dict = {}
        
        transforms = [[1,1],[-1,1],[1,-1],[-1,-1]]
        all_data = []
        
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        
        step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001]
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10
        
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized_flag = 0
            while not optimized_flag:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), 1*(self.max_feature_value*b_range_multiple), step*b_multiple):
                    for transform in transforms:
                        w_t = w * transform
                        #print(w_t)
                        found_street_points = False
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(xi, w_t)+b) >= 1:
                                    found_street_points = True
                        if not (found_street_points):
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                        
                if w[0] < 0:
                    optimized_flag = 1
                    print('optimized a step...')
                else:
                    w = w-step
            
            norms = sorted([mod_w for mod_w in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
 
    
    #Method for prediction of new data points
    def predict(self, features):
        #Predict the sign of (x.w+b)
        classify = np.sign(np.dot(np.array(features), self.w)+self.b)
        if classify == 0:
            print('there is a 50% chance of the data point being in either of the class!!!!')
        
        return classify
    
    def visualize(self, data):
        [[plt.scatter(x[0], x[1], s=100, color= self.colors[i], marker='*') for x in data[i]] for i in data]
        
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]
    
        hyp_x_min = self.min_feature_value * 0.9
        hyp_x_max = self.max_feature_value * 1.1
        
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        plt.plot([hyp_x_min,hyp_x_max],[psv1, psv2])
        
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        plt.plot([hyp_x_min,hyp_x_max],[nsv1, nsv2])
        
        dl1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        dl2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        plt.plot([hyp_x_min,hyp_x_max],[dl1, dl2])
        #print('printing the SVM plot')
        plt.show()
            
        
svm = SVM()
svm.fit(data = data_dict)
svm.visualize(data = data_dict)

