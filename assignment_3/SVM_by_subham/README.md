## Approach

1. Generated the toy dataset using make_blobs
	- Approach
2. Converted the data into a dictionary data_dict with positive class as 1 and negative as -1.

3. Made a class named SVM with 3 method.
    -fit
    -predict
    -visualize

4. In method fit, searched for the optimum w and optimum b in the range of -(maximum feature value) to +(maximum feature value) initialy and then decreased the range with 0.1 then with 0.01 and lastly with 0.001. If any w and b are the required w and b then checked the condition yi*(xi*w+b) >= 1 to be true.

5. Visualised the data with the method visualize. 

## Steps to run

1. If want to run with the default dataset:
    -clone the repo.
    -run the svmfrombasic.py script in terminal by the command      => python svmfrombasic.py

2. If want to use a different dataset:
    -clone the repo
    -create your own dataset and change the python file accodingly and chnge the parameter of svm.fit and svm.visualize with your own dataset.