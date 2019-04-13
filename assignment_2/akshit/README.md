## Problem Statement:

Train a model to print the first n elements of the Fibonacci sequence starting at integers a and b. Note that a < b, and n <= 100 are user supplied arguments. You may not use the standard summing operation to print the next element.Example I/O: Input: a=5, b=7, n=5 Output: 5, 7, 12, 19, 31 Input: a=7, b=41, n=6 Output: 7, 41, 49, 90, 139, 229

## Approach:

Trained a simple neural network which will take the latest 5 numbers in the sequence as input. If sequence contains less than 5 numbers, it will have 0 in their place.

## How to Run:

In predict.ipynb run the function fibonacci with 3 inputs; a,b and n; and you will get the fibonacci sequence.

## Dataset:

Dataset which was created is a text file containing 210 fibonacci sequences.

## Result:

Training error over all possible cases in the training set is less than 0.5%.

