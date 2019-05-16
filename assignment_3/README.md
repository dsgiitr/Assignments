## Problem Statement

Create a Linear SVM Classifier from scratch using Numpy or similar libraries. 
Train your model on a Toy Dataset (say it has 500 datapoints each for binary classification) which are linearly separable.

Compare your implementation in terms of runtime and accuracy with the one in [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

You may refer to [this](https://datascience.stackexchange.com/questions/39071/create-a-binary-classification-dataset-python-sklearn-datasets-make-classifica) for building a dataset on similar grounds.

## Submission Guidelines

1. Make a proper readme.md including:
	- Approach
	- Dataset
	- Model
	- Results (Including Comparision with sklearn)
	- Steps to Run
	
2. Include the Requirements.txt file.

3. Make a predict.py file so that user may test the model on dataset.

4. Strictly submit the project scripts with .py file. 	(You may use notebooks but then export them to .py files before submitting)

5. Add function string comments wherever possible.

    Example: A function which applies some gaussian distance filter taking two args.
	    
	    def foo(arg1, arg2):
	    
            """
            Apply Gaussian distance filter to a numpy distance array
        
            Parameters
            ----------
            arg1 : np.array
                Description of arg1
            arg2 : int
                Description of arg2 
    
            Returns
            -------
            expanded_distance: shape (n+1)-d array
                Expanded distance matrix with the last dimension of length
                len(self.filter)
            """ 
