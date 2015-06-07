# CDev
Cross device prediction for ICDM kaggle competition

## TODO
1. Domain knowledge    
We want to understand these cookies and IP things.     

2. Data pre-processing    
Note ``awk'' and ``sed'' are highly recommended cause they are fast.

3. Machine Learning     
See below.

4. Submission    
We have to follow the protocal from Kaggle.


## Possible paths     
Essentially, this is a identification problem. If it has to be related to some problem that I've tackled, that could be Face Verification problem. It is strongly recommended to go through some literature first.

Of course, we can form it as a supervised learning problem, then all the traditional schemes can be adopted (sorted by priority):   
1. Gradient Boosting Machines (kaggle-favored)   
2. SVM with different kernels    
3. Random Forest
4. ...

Making it a supervised learning problem doesn't make that much sense to me actually. Inspired from Face community, the following appoaches are really worth trying (sorted by priority):    
1. Siemise convolutional net
2. DrLIM       
3. Metric learning     


## Important notes
We always start from easiest thing, and get it complicated. This bottom-up path can make us always feel clear about what we are doing. Maybe it will waste some time at the begining, I faithfully believe that will pay off along the road we go.

For supervised, do scikit-learn on the raw data first.     
For another, do LDA (linear discriminative analysis, easest version metric) in Euclidean space.
