## Problem Statement: 
Train a model to print the first `n` elements of the Fibonacci sequence starting at integers `a` and `b`. Note that `a` < `b`, and `n` <= 100 are user supplied arguments. You may not use the standard summing operation to print the next element.Example I/O:
Input: a=5, b=7, n=5
Output: 5, 7, 12, 19, 31
Input: a=7, b=41, n=6
Output: 7, 41, 49, 90, 139, 229

## Submission Guidelines
1. Make a proper readme.md including the approach, dataset, model, experiments, results and steps to run the project.
2. Include requirement.txt file.
3. Make a predict.py file in which a user will input a,b,n and get the results.

## Approach
To train a simple Linear Regression model using randomly generated dataset using basic functionalities of pytorch . 

## Dataset
Consisted of 3 columns : 
1. value 1
2. value 2
3. value 1 + value 2 
It was generated randomly and stored as a csv file 'mydata.csv' .

## Model
A The Module.nn class was modified to add a single linear layer of input_dim = 1 and output_dim = 2 . The learning_rate = 0.00000001. Stochastic Gradient Descent was used as optimizer function . epochs = 500. Round-off function was used to round the results to integers ( reduce the propagation of error ) . Trained entirely in CPU . 

## Results
The model was successfully trained to add the values in the columns to give the output without using the summation operator . model() was used to generate the required fibonnacci sequence.

## Experiments
An additional hidden layer (of size 3) was added but it didnot improve the model . 
## Steps
1. Run the file predict.py
2. Input a,b,n and generate fibonacci 
