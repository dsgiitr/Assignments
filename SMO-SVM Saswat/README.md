## Problem Statement
Create a Linear SVM Classifier from scratch using Numpy or similar libraries. Train your model on a Toy Dataset (say it has 500 datapoints each for binary classification) which are linearly separable.

Compare your implementation in terms of runtime and accuracy with the one in sklearn.
## Approach
This is the Python implementation of John C. Platt's Sequential Minimal Optimization (SMO) for training a support vector machine (SVM). This program is based on pseudocode in Platt(1998).
Link to research paper: https://pdfs.semanticscholar.org/59ee/e096b49d66f39891eb88a6c84cc89acba12d.pdf

## Model
Numpy Arrays is used to store the datapoints and Values. The toy dataset is generated from Scikit-Learn's make_blob function.
The SMOModel Object consists of following parameters:
1. X : datapoints (ndArray)
2. y : target values (ndArray) {-1,1} (auto converts from {1,0})
3. kernel : np.dot by default 
4. C : Regularization parameter
5. tol : tolerance
6. eps : lagrange multiplier tolerance

## Results
The LinearSVC decision boundary and that produced by this model is almost similar. Both of them successfully classify a separable case with 100% accuracy. 

## Steps to Run
1. Import predict.py function
2. Load the Model with Data and Parameters
3. The train function returns the trained model.
4. Obtain the value of coefficients by model.W
5. Obtain the value of intercept by model.b
6. Predict using model.predict(X) where X is the input
