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

COLNAMES = [
    "label",
    "sigHH_Rad05July", "sigHV_Rad05July", "sigVV_Rad05July", "sigRR_Rad05July", "sigRL_Rad05July", "sigLL_Rad05July",
    "Rhhvv_Rad05July", "Rhvhh_Rad05July", "Rhvvv_Rad05July", "Rrrll_Rad05July", "Rrlrr_Rad05July", "Rrlll_Rad05July",
    "Rhh_Rad05July", "Rhv_Rad05July", "Rvv_Rad05July", "Rrr_Rad05July", "Rrl_Rad05July", "Rll_Rad05July",
    "Ro12_Rad05July", "Ro13_Rad05July", "Ro23_Rad05July",
    "Ro12cir_Rad05July", "Ro13cir_Rad05July", "Ro23cir_Rad05July",
    "l1_Rad05July", "l2_Rad05July", "l3_Rad05July",
    "H_Rad05July", "A_Rad05July", "a_Rad05July",
    "HA_Rad05July", "H1mA_Rad05July", "1mHA_Rad05July", "1mH1mA_Rad05July",
    "PH_Rad05July", "rvi_Rad05July",
    "paulalpha_Rad05July", "paulbeta_Rad05July", "paulgamma_Rad05July",
    "krogks_Rad05July", "krogkd_Rad05July", "krogkh_Rad05July",
    "freeodd_Rad05July", "freedbl_Rad05July", "freevol_Rad05July",
    "yamodd_Rad05July", "yamdbl_Rad05July", "yamhlx_Rad05July", "yamvol_Rad05July",
    "sigHH_Rad14July", "sigHV_Rad14July", "sigVV_Rad14July", "sigRR_Rad14July", "sigRL_Rad14July", "sigLL_Rad14July",
    "Rhhvv_Rad14July", "Rhvhh_Rad14July", "Rhvvv_Rad14July", "Rrrll_Rad14July", "Rrlrr_Rad14July", "Rrlll_Rad14July",
    "Rhh_Rad14July", "Rhv_Rad14July", "Rvv_Rad14July", "Rrr_Rad14July", "Rrl_Rad14July", "Rll_Rad14July",
    "Ro12_Rad14July", "Ro13_Rad14July", "Ro23_Rad14July",
    "Ro12cir_Rad14July", "Ro13cir_Rad14July", "Ro23cir_Rad14July",
    "l1_Rad14July", "l2_Rad14July", "l3_Rad14July",
    "H_Rad14July", "A_Rad14July", "a_Rad14July",
    "HA_Rad14July", "H1mA_Rad14July", "1mHA_Rad14July", "1mH1mA_Rad14July",
    "PH_Rad14July", "rvi_Rad14July",
    "paulalpha_Rad14July", "paulbeta_Rad14July", "paulgamma_Rad14July",
    "krogks_Rad14July", "krogkd_Rad14July", "krogkh_Rad14July",
    "freeodd_Rad14July", "freedbl_Rad14July", "freevol_Rad14July",
    "yamodd_Rad14July", "yamdbl_Rad14July", "yamhlx_Rad14July", "yamvol_Rad14July",
    "B_Opt05July", "G_Opt05July", "R_Opt05July", "Redge_Opt05July", "NIR_Opt05July",
    "NDVI_Opt05July", "SR_Opt05July", "RGRI_Opt05July", "EVI_Opt05July", "ARVI_Opt05July",
    "SAVI_Opt05July", "NDGI_Opt05July", "gNDVI_Opt05July", "MTVI2_Opt05July",
    "NDVIre_Opt05July", "SRre_Opt05July", "NDGIre_Opt05July", "RTVIcore_Opt05July",
    "RNDVI_Opt05July", "TCARI_Opt05July", "TVI_Opt05July", "PRI2_Opt05July",
    "MeanPC1_Opt05July", "VarPC1_Opt05July", "HomPC1_Opt05July", "ConPC1_Opt05July",
    "DisPC1_Opt05July", "EntPC1_Opt05July", "SecMomPC1_Opt05July", "CorPC1_Opt05July",
    "MeanPC2_Opt05July", "VarPC2_Opt05July", "HomPC2_Opt05July", "ConPC2_Opt05July",
    "DisPC2_Opt05July", "EntPC2_Opt05July", "SecMomPC2_Opt05July", "CorPC2_Opt05July",
    "B_Opt14July", "G_Opt14July", "R_Opt14July", "Redge_Opt14July", "NIR_Opt14July",
    "NDVI_Opt14July", "SR_Opt14July", "RGRI_Opt14July", "EVI_Opt14July", "ARVI_Opt14July",
    "SAVI_Opt14July", "NDGI_Opt14July", "gNDVI_Opt14July", "MTVI2_Opt14July",
    "NDVIre_Opt14July", "SRre_Opt14July", "NDGIre_Opt14July", "RTVIcore_Opt14July",
    "RNDVI_Opt14July", "TCARI_Opt14July", "TVI_Opt14July", "PRI2_Opt14July",
    "MeanPC1_Opt14July", "VarPC1_Opt14July", "HomPC1_Opt14July", "ConPC1_Opt14July",
    "DisPC1_Opt14July", "EntPC1_Opt14July", "SecMomPC1_Opt14July", "CorPC1_Opt14July",
    "MeanPC2_Opt14July", "VarPC2_Opt14July", "HomPC2_Opt14July", "ConPC2_Opt14July",
    "DisPC2_Opt14July", "EntPC2_Opt14July", "SecMomPC2_Opt14July", "CorPC2_Opt14July",
]


def load_data(file_path, colnames=None):
    # Load crop dataset (file already includes a header row).
    data = pd.read_csv(file_path, header=0)
    if colnames is not None:
        if len(colnames) == data.shape[1]:
            data.columns = colnames
        else:
            logging.warning(
                "Provided %d column names but data has %d columns; keeping file headers.",
                len(colnames),
                data.shape[1],
            )
    X = data.iloc[:, 1:].values  # Features are from the 2nd column to the last.
    y = data.iloc[:, 0].values  # Labels are in the first column.
    return train_test_split(X, y, test_size=0.2, random_state=42)


def extract_linear_coeffs(estimator, feature_names):
    # Map linear-term coefficients back to original feature names.
    poly = estimator.named_steps["poly"]
    logistic = estimator.named_steps["logistic"]
    expanded_names = poly.get_feature_names_out(feature_names)
    coef = logistic.coef_.ravel()
    name_to_coef = {}
    for idx, name in enumerate(expanded_names):
        if name in feature_names:
            name_to_coef[name] = coef[idx]
    return name_to_coef


# Setting random seed.
RANDOM_SEED = 2022
np.random.seed(RANDOM_SEED)

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

CROP_LABELS = {
    1: "corn",
    2: "peas",
    3: "canola",
    4: "soybeans",
    5: "oats",
    6: "wheat",
    7: "broadleaf",
}
crop_name = CROP_LABELS.get(target_class, f"class{target_class}")
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"output/crop_one_{crop_name}_{date_str}_seed{RANDOM_SEED}.txt"
coeffs_filename = f"output/crop_one_coeffs_{crop_name}_{date_str}_seed{RANDOM_SEED}.csv"

logging.info(f"Received parameters: {argv[1:]}")

X_train, X_test, y_train, y_test = load_data(DATA_FILE_PATH, colnames=COLNAMES)


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

# Build coefficient table.
num_features = X_train.shape[1]
expected_features = COLNAMES[1:]
if len(expected_features) != num_features:
    logging.warning(
        "Expected %d feature names but data has %d features; using generic names.",
        len(expected_features),
        num_features,
    )
    all_feature_names = [f"x{i}" for i in range(num_features)]
else:
    all_feature_names = expected_features
rows = []
for idx, base_estimator in enumerate(estimator.estimators_):
    selected_idx = estimator.estimators_features_[idx]
    selected_names = [all_feature_names[i] for i in selected_idx]
    name_to_coef = extract_linear_coeffs(base_estimator, selected_names)
    row = {name: np.nan for name in all_feature_names}
    for name, coef in name_to_coef.items():
        row[name] = coef
    row["estimator_index"] = idx
    rows.append(row)

coeffs_df = pd.DataFrame(rows)
cols = ["estimator_index"] + all_feature_names
coeffs_df = coeffs_df[cols]
coeffs_df.to_csv(coeffs_filename, index=False)
logging.info(f"Saved coefficients to {coeffs_filename}.")

# More logging as appropriate.
# end_time = time()
# logging.info(f"Finished estimation after {end_time - start_time} seconds.")
