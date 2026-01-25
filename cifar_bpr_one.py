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
    # Load crop dataset.
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:].values  # Features are from the 2nd column to the last.
    y = data.iloc[:, 0].values  # Labels are in the first column.
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Setting random seed.
np.random.seed(2022)

# Logging and time check.
start_time = time()
logging.basicConfig(level=logging.INFO)

# Constants.
DATA_FILE_PATH = "data/WinnipegDataset.txt"

# Parse parameters.
n_estimators = int(argv[1])
poly_degree = int(argv[2])
max_samples = int(argv[3])
max_features = int(argv[4])
c_reg = float(argv[5])
target_class = int(argv[6]) if len(argv) > 6 else 1

# By default, creates a file called 'result.txt' or appends to one already existing.
filename = "crop_classification_one_result.txt"

logging.info(f"Received parameters: {argv[1:]}")

X_train, X_test, y_train, y_test = load_data(DATA_FILE_PATH)

# One-vs-all labels for the target class.
y_train_onehot = y_train == target_class
y_test_onehot = y_test == target_class

# Base estimator: logistic regression on polynomials.
pipe = Pipeline(
    [
        ("poly", PolynomialFeatures(poly_degree)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=c_reg, max_iter=1000, solver="liblinear")),
    ]
)
logging.info(f"Base estimator: {pipe}.")

# Bagging classifier for one-vs-all task.
estimator = BaggingClassifier(
    estimator=pipe,
    n_estimators=n_estimators,
    max_features=max_features,
    max_samples=max_samples,
    n_jobs=4,
)

estimator.fit(X_train, y_train_onehot)

yhat_train = estimator.predict(X_train)
yhat_test = estimator.predict(X_test)

accuracy_train = np.mean(yhat_train == y_train_onehot)
accuracy_test = np.mean(yhat_test == y_test_onehot)
logging.info(f"Train accuracy: {accuracy_train}.")
logging.info(f"Test accuracy: {accuracy_test}.")

with open(filename, "a") as f:
    print(
        str(datetime.now()),
        n_estimators,
        poly_degree,
        max_samples,
        max_features,
        c_reg,
        target_class,
        accuracy_train,
        accuracy_test,
        file=f,
    )

# More logging as appropriate.
# end_time = time()
# logging.info(f"Finished estimation after {end_time - start_time} seconds.")
