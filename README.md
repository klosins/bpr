# Code for Bagged Polynomial Regression and Neural Networks
This repo contains two main files mnist_bpr_multi.py and mnist_bpr_one.py.

+ mnist_bpr_multi.py: Runs bagged polynomial regression to predict all ten digits in the MNIST dataset, which can be downloaded from http://yann.lecun.com/exdb/mnist/. Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license

Code inputs: Our code for bagged polynomial regression takes in 5 inputs 
  - n_estimators = The number of base estimators in the ensemble. (fed into sklearn.ensemble.BaggingRegressor )
  - poly_degree = Specifies the maximal degree of the polynomial features.  (to be fed into sklearn.preprocessing.PolynomialFeatures)
  - max_samples = The number of samples to draw from covariates X to train each base estimator (with replacement by default) (fed into
  sklearn.ensemble.BaggingRegressor)
  - max_features = The number of features to draw from covariates X to train each base estimator (without replacement by default) (fed into
  sklearn.ensemble.BaggingRegressor)
  - c_reg = Inverse of regularization strength; must be a positive float. (fed into sklearn.linear_model.LogisticRegression)

Running code: To run code
  - First save mnist_bpr_multi.py in the same folder where the MNIST data is saved. 
  - Open terminal and navigatge to the directory with mnist_bpr_multi.py
  - Run by writting in terminal "python3 mnist_bpr_multi.py 10 2 60000 10 1"


+ mnist_bpr_one.py: Runs bagged polynomial regression to predict the digit 1 from the MNIST dataset. Same inputs as mnist_bpr_multi.py. 
