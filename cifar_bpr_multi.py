import logging
from datetime import datetime
from sys import argv, executable
from time import time

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

print(f"Using Python executable: {executable}")

def load_data(file_path):
    # Load crop dataset
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:].values  # Features are from the 2nd column to the last
    y = data.iloc[:, 0].values   # Labels are in the first column
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Setting random seed
np.random.seed(2022)

# Logging and time check.
start_time = time()
logging.basicConfig(level=logging.INFO)

# Constants.
DATA_FILE_PATH = "/Users/sylviaklosin/Documents/GradSchool/research/bag_poly/crop/WinnipegDataset.txt"  # Update this with your dataset path
# Parse parameters.
n_estimators = 60#int(argv[1])
poly_degree = 2#int(argv[2])
max_samples = 50000#float(argv[3])
max_features = 60 #float(argv[4])
c_reg = 10#float(argv[5])

# By default, creates a file called 'result.txt' or appends to one already existing.
filename = "crop_classification_result.txt"

logging.info(f"Received parameters: {argv[1:]}")

X_train, X_test, y_train, y_test = load_data(DATA_FILE_PATH)

# Determine the number of unique classes
NUM_CLASSES = len(np.unique(y_train))

# Base estimator: logistic regression on polynomials
pipe = Pipeline(
    [
        ("poly", PolynomialFeatures(poly_degree)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=c_reg, max_iter=1000, solver='liblinear')),
    ]
)
logging.info(f"Base estimator: {pipe}.")

# Estimate and compute train, test predictions for each class.
estimator = {}
yhat_train = pd.DataFrame()
yhat_test = pd.DataFrame()
for i in range(1,NUM_CLASSES):
    logging.info(f"Estimating class {i}.")

    # Create a bagging classifier with desired features.
    estimator[i] = BaggingClassifier(
        base_estimator=pipe,
        n_estimators=n_estimators,
        max_features=max_features,
        max_samples=max_samples,
        n_jobs=4,
    )

    # One-hot encode labels for this class.
    y_train_onehot = y_train == i

    # Fit bagging classifier using one-vs-all data.
    estimator[i].fit(X_train, y_train_onehot)

    # Compute and store training set raw predictions.
    yhat_train[i] = estimator[i].predict(X_train)

    # Compute and store test set raw predictions.
    yhat_test[i] = estimator[i].predict(X_test)

# Take the argmax across classes to find out predicted classes.
predicted_label_train = yhat_train.idxmax(1)
predicted_label_test = yhat_test.idxmax(1)

# Compute train, test accuracy.
accuracy_train = np.mean(predicted_label_train == y_train)
accuracy_test = np.mean(predicted_label_test == y_test)
logging.info(f"Train accuracy: {accuracy_train}.")
logging.info(f"Test accuracy: {accuracy_test}.")

# Append results to existing file, or create one if none exists.
#with open(filename, "a") as f:
#    print(
#        str(datetime.now()),
#        n_estimators,
#        poly_degree,
#        max_samples,
#        max_features,
#        c_reg,
#        accuracy_train,
#        accuracy_test,
#        file=f,
#    )

#logging.info(f"Saved at {filename}.")

# More logging as appropriate.
#end_time = time()
#logging.info(f"Finished estimation after {end_time - start_time} seconds.")
