# SVM From Scratch
Approach
1. Generated the random dataset by defining a center and plotting random points at a fixed distance from it 

2. Assigned one of the cluster as value 1 and another as -1 

 xi.w + b <= -1   if yi = -1 (belongs to -ve class)
 xi.w + b >= +1	if yi = +1 (belongs to +ve class)

3. Made a class named SupportVecMac with 3 functions.
    -fit to find the required w and b using training data
    -predict to predict the outcomes on new data
    -visualize to plot the data
    
    
    Algorithm
    
    1.Started with random big value of w in this case 10 times max feature value and       decreased it slowly
    
    2.Initially selected step size as w0*0.1
    
    3. For b took a small value of b,
       b will range from (-b0 < b < +b0, step = step*b_multiple)
       
     4. Checked for points xi in dataset:
        Checked constraint for all transformation of w like (w,w), (-w,w), (w,-w), (-          w,-w)
        if not yi(xi.w+b)>=1 for all points then break
        Else find |w| and put it in dictionary as key and (w,b) as values
        
        If w<=0 then decrease step size, no need to check for negative as transformation already cover that
        
        On comparing the time taken it is found that time taken by SKlearn's SVM is much less than the one created from scratch
    

