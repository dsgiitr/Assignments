import matplotlib.pyplot as plt
import numpy as np


class Support_Vector_Machine:
    def __init__(self,visualization=True):
        self.visualization=visualization 
        self.colors={1:'r',-1:'b'}
        if self.visualization :
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot(1,1,1)
            
    #train
       
    def fit(self,data):
        self.data=data
        #in the below dictionary key=||w||,values=[w,b]
        opt_dict={}
        
        transforms=[[1,1],
                    [-1,1], 
                    [-1,-1],
                    [1,-1]]
        
        
        all_data=[]          # all_data would contains all the feature values in the form of list
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
            
            
        self.max_feature_value=max(all_data)
        self.min_feature_value=min(all_data)
        all_data= None
        
        step_sizes=[self.max_feature_value*0.1,
                    self.max_feature_value*0.01,
                    self.max_feature_value*0.001]
        
        b_range_multiple=5
        b_multiple=5
        latest_optimum=self.max_feature_value*10
        
        # training of model
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformations in transforms:
                        w_transform=w*transformations
                        flag=True
                        for yi in self.data:
                            for xi in self.data[yi]:
                                
                                if not yi*(np.dot(xi,w_transform)+b)>=1:
                                    flag=False
                                
                        if flag==True:
                            opt_dict[np.linalg.norm(w_transform)]=[w_transform,b]
                
                if w[0]<0:
                    optimized=True
                    print('optimized a step')
                else:
                    w=w-step
            
            norms=sorted([p for p in opt_dict])
            self.w=opt_dict[norms[0]][0]
            self.b=opt_dict[norms[0]][1]
            latest_optimum=self.w[0]+step*2
 
    
    def predict(self,data):
        # sign(x.w+b)
        pred=[]
        for features in data:
            classification = np.sign(np.dot(np.array(features),self.w)+self.b)
            if classification !=0 and self.visualization:
                self.ax.scatter(features[0], features[1], s=30, marker='*', c=self.colors[classification])
            pred.append( classification)
        return pred 
    
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=30,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()



#creating toy dataset
from sklearn.datasets.samples_generator import make_blobs
(X,y)=make_blobs(n_samples=100,n_features=2,cluster_std=1.5,centers=2,random_state=1)
col=np.random.random(len(y))
col=list(col)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#creating our data dictonary
pos_x=[]
neg_x=[]
for i,j in enumerate(y_train):
    if j==0:
        neg_x.append(X_train[i])
    else:
        pos_x.append(X_train[i])

data_dict={-1:np.array(neg_x),1:np.array(pos_x)}


svm=Support_Vector_Machine()
svm.fit(data_dict)
#predicting values
y_pred=svm.predict(X_test)


svm.visualize()

for i in range(len(y_pred)):
    if y_pred[i]==-1:
        y_pred[i]=0
        
        
# using svm from scikit learn
        
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)

y_pred_skl=classifier.predict(X_test)

# comparing the results
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred_skl,y_pred)
