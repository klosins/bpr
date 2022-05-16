import logging
import random
from datetime import datetime
from os.path import join
from sys import argv
from time import time

import numpy as np
import pandas as pd
from mlxtend.data import loadlocal_mnist
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Setting random seed 
np.random.seed(2022)

# Logging and time check.
start_time = time()
logging.basicConfig(level=logging.INFO)

# Constants.
NUM_DIGITS = 10
DATA_DIR = "/bbkinghome/klosins/Documents/poly_reg_high_dim"

# Parse parameters.
n_estimators = int(argv[1])
poly_degree = int(argv[2])
max_samples = int(argv[3])
max_features = int(argv[4])
c_reg = float(argv[5])

# By default, creates a file called 'result.txt'.
# or appends to one already existing.
if len(argv) == 6:
    filename = "mnist_bpr_result.txt"
else:
    filename = argv[5]  # NOTE: make sure to see what happens when if is false

logging.info(f"Received parameters: {argv[1:]}")


# Load train data
logging.info("Loading training data.")
X_train, y_train = loadlocal_mnist(
    images_path=join(DATA_DIR, "train-images-idx3-ubyte"),
    labels_path=join(DATA_DIR, "train-labels-idx1-ubyte"),
)

# Load test data
logging.info("Loading test data.")
X_test, y_test = loadlocal_mnist(
    images_path=join(DATA_DIR, "t10k-images-idx3-ubyte"),
    labels_path=join(DATA_DIR, "t10k-labels-idx1-ubyte"),
)


# Base estimator: logistic regression on polynomials
pipe = Pipeline(
    [
        # Preprocessing 1: polynomial expansion (only of the columns/rows selected by BaggingClassifier)
        ("poly", PolynomialFeatures(poly_degree)),
        # Prep 2: scaling
        ("scale", StandardScaler()),
        # logistic regression (C is the degree of regularization)
        (
            "logistic",
            LogisticRegression(C=c_reg, max_iter=100),
        ), 
    ]
)
logging.info(f"Base estimator: {pipe}.")


# Estimate and compute train, test predictions for each digit.
estimator = {}
yhat_train = pd.DataFrame()
yhat_test = pd.DataFrame()
for i in range(NUM_DIGITS):

    # NOTE: It's good practice to log stuff. Adding one as example here, feel free to remove or add more as appropriate.
    logging.info(f"Estimating digit {i}.")

    # Create a bagging regressor with desired features.
    estimator[i] = BaggingRegressor(
        base_estimator=pipe,
        n_estimators=n_estimators,
        max_features=max_features,
        max_samples=max_samples,
        n_jobs=4,
    )  

    # One-hot encode labels for this digits.
    y_train_onehot = y_train == i

    # Fit bagging regressor using one-vs-all data.
    estimator[i].fit(X_train, y_train_onehot)

    # Compute and store training set raw predictions.
    yhat_train[i] = estimator[i].predict(X_train)

    # Compute and store test set raw predictions.
    yhat_test[i] = estimator[i].predict(X_test)


# Take the argmax across digits to find out predicted digits.
predicted_label_train = yhat_train.idxmax(1)
predicted_label_test = yhat_test.idxmax(1)

# Compute train, test accuracy.
accuracy_train = np.mean(predicted_label_train == y_train)
accuracy_test = np.mean(predicted_label_test == y_test)
logging.info(f"Train accuracy: {accuracy_train}.")
logging.info(f"Test accuracy: {accuracy_test}.")

# Append results to existing file, or create one if none exists.
with open(filename, "a") as f:
    print(
        str(datetime.now()),
        n_estimators,
        poly_degree,
        max_samples,
        max_features,
        c_reg,
        accuracy_train,
        accuracy_test,
        file=f,
    )

logging.info(f"Saved at {filename}.")


# More logging as appropriate.
end_time = time()
logging.info(f"Finished estimation after {end_time - start_time} seconds.")
