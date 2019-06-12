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

## Experiments
1. I inserted little GUI in it because I recently learned little GUI and want to implement it

## Approach
1. We just have to make a model that can add two number.
2. Our model will take a,b and return a+b
3. We will make a function that will print fibonacci series
 
## Dataset
1. X has two features a,b and very big length
2. y has a+b and same length as of X

## Model
1. We trained our model using simple linear regression using pytorch
2. After training our model gives w=[[1],[1]] it is used for adding a,b
