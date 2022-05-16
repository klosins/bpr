#data downloaded from http://yann.lecun.com/exdb/mnist/ 

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
import numpy as np
from sys import argv
from datetime import datetime
from mlxtend.data import loadlocal_mnist


n_estimators = int(argv[1])
poly_degree = int(argv[2])
max_samples = int(argv[3])
max_features = int(argv[4])
c_reg = float(argv[5])

# By default, creates a file called 'result.txt'
# or appends to one already existing.
#import pdb; pdb.set_trace()
if len(argv) == 6:
    filename = f"mnist_bpr_result.txt"

print(f"Running with parameters: {argv[1:]}")


# Load data
X_train, y_train = loadlocal_mnist(
	images_path='train-images-idx3-ubyte', 
	labels_path='train-labels-idx1-ubyte')



y_train = np.where(y_train == 1, 1, 0)

X_test, y_test = loadlocal_mnist(
	images_path='t10k-images-idx3-ubyte', 
	labels_path='t10k-labels-idx1-ubyte')

y_test = np.where(y_test == 1, 1, 0)


# Base estimator: logistic regression on polynomials
# Base estimator: logistic regression on polynomials
pipe = Pipeline([
    ("poly", PolynomialFeatures(poly_degree)),  # Preprocessing 1: polynomial expansion (only of the columns/rows selected by BaggingClassifier)
    ("scale", StandardScaler()),  # Prep 2: scaling
    ("logistic", LogisticRegression(C=c_reg, max_iter=100)) # logistic regression (C is the degree of regularization)
])




# Bagging estimator
estimator = BaggingRegressor( base_estimator=pipe, n_estimators = n_estimators, max_features=max_features, max_samples=max_samples, n_jobs=4 )

estimator.fit(X_train, y_train)


#for i in estimator.estimators_:
#	y_pred1 = i.predict(X_test)


yhat_test = estimator.predict(X_test)
yhat_test = np.where(.5 <= yhat_test , 1, 0)

yhat_train = estimator.predict(X_train)
yhat_train = np.where(.5 <= yhat_train , 1, 0)


accuracy_train = np.mean(yhat_train == y_train)
accuracy_test = np.mean(yhat_test == y_test)




with open(filename, "a") as f:
    print(
        str(datetime.now()),
        n_estimators, poly_degree,
        max_samples, max_features, c_reg,
        accuracy_train, accuracy_test,
    file=f)
