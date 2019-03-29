# Problem Statement:

Train a model to print the first `n` elements of the Fibonacci sequence starting at integers `a` and `b`. Note that `a` < `b`, and `n` <= 100 are user supplied arguments. You may not use the standard summing operation to print the next element.

##### Example I/O:
Input: a=5, b=7, n=5
Output: 5, 7, 12, 19, 31

Input: a=7, b=41, n=6
Output: 7, 41, 49, 90, 139, 229

## Steps to run 

The file train.ipynb contains the code for model creation and training and outputs the model's trained parameters to data.pt.
Predict.py contains the code to predict using the given input while using the parameters from data.pt file.

## Approach used

A RNN like model has been used to predict the sequence. 

![Model_Architecture.jpg](img/model_architecture.png?raw=true "Model Architecture")

The model is then feed backed its output n times to get the next n terms.

##  Results

The relative error in terms predicted by the model is less than 1%.



