#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:38:15 2019

@author: subham
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

(X,y) =  make_blobs(n_samples=50,n_features=2,centers=2,cluster_std=1.05,random_state=40)

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

from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X, y)

