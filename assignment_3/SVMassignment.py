import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


center1 = (70, 60)
center2 = (90, 20)
distance = 10

import numpy as np
x1 = np.random.uniform(center1[0], center1[0] + distance, size=(50,))
y1 = np.random.normal(center1[1], distance, size=(50,)) 

x2 = np.random.uniform(center2[0], center2[0] + distance, size=(50,))
y2 = np.random.normal(center2[1], distance, size=(50,))


class SupportVecMac:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.axis = self.fig.add_subplot(1,1,1)

    #Method for training the dataset
    def fit(self, data):
        self.data = data
        #opt_dict is a dictionary of { ||w|| : [w,b]} the norm is the index
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
                        
                        optimpoints = False
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(xi, w_t)+b) >= 1:
                                    optimpoints = True
                        if not (optimpoints):
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized_flag = 1
                    print('Reducing Step Size')
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
        return classify

    def visualize(self, data):
       [[plt.scatter(x[0],x[1],s=100,c=self.colors[i]) for x in data_dict[i]] for i in data_dict]
       def hyperplane(x,w,b,v):
           return (-w[0]*x-b+v)/w[1]
       
       hyp_x_min= self.min_feature_value*0.9
       hyp_x_max = self.max_feature_value*1.1
        
        # (w.x+b)=1
        # positive support vector hyperplane
       pav1 = hyperplane(hyp_x_min,self.w,self.b,1)
       pav2 = hyperplane(hyp_x_max,self.w,self.b,1)
       plt.plot([hyp_x_min,hyp_x_max],[pav1,pav2],'k')
        
        # (w.x+b)=-1
        # negative support vector hyperplane
       nav1 = hyperplane(hyp_x_min,self.w,self.b,-1)
       nav2 = hyperplane(hyp_x_max,self.w,self.b,-1)
       plt.plot([hyp_x_min,hyp_x_max],[nav1,nav2],'k')
        
        # (w.x+b)=0
        # db support vector hyperplane
       db1 = hyperplane(hyp_x_min,self.w,self.b,0)
       db2 = hyperplane(hyp_x_max,self.w,self.b,0)
       plt.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')       



 

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.show()

c1=np.column_stack((x1, y1))
c2=np.column_stack((x2, y2))
data_dict = {-1:c1,1:c2}
print(data_dict)


import time
startime=time.time()
svm =SupportVecMac() # Linear Kernel
svm.fit(data=data_dict)
svm.visualize(data=data_dict)
endtime=time.time()
print("SVM Made from Scratch"+str(endtime-startime))



#Comparison with SKlearn
X=np.row_stack((c1, c2))
print(X)
k=np.ones((50,1))*-1
y=np.row_stack((np.ones((50, 1)),k))


startt=time.time()
from sklearn.svm import SVC
Classifier=SVC(kernel='linear') 
Classifier.fit(X,y.ravel())
endd=time.time()
print("SKLEARNS SVM"+str(1000*(endd-startt)))
print("Time is much less in case of Sklearn ")
