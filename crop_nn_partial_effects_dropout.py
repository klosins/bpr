import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(input_dim, num_classes, hidden1, hidden2, dropout, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden2, num_classes),
    )
    return model


def evaluate_model(model, X, y, device, batch_size):
    model.eval()
    correct = 0
    total = 0
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total if total else 0.0


def compute_pdp_ice(model, X_raw, scaler, feature_idx, class_idx, grid, subsample, seed, device):
    rng = np.random.default_rng(seed)
    num_samples = min(subsample, X_raw.shape[0])
    sample_idx = rng.choice(X_raw.shape[0], size=num_samples, replace=False)
    X_sample = X_raw[sample_idx]

    pdp = []
    ice = np.zeros((num_samples, len(grid)))
    for j, value in enumerate(grid):
        X_mod = X_raw.copy()
        X_mod[:, feature_idx] = value
        X_scaled = scaler.transform(X_mod)
        with torch.no_grad():
            logits = model(torch.from_numpy(X_scaled).float().to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        pdp.append(probs[:, class_idx].mean())

        X_sample_mod = X_sample.copy()
        X_sample_mod[:, feature_idx] = value
        X_sample_scaled = scaler.transform(X_sample_mod)
        with torch.no_grad():
            logits = model(torch.from_numpy(X_sample_scaled).float().to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        ice[:, j] = probs[:, class_idx]
    return np.array(pdp), ice


def resolve_feature_index(feature, feature_names):
    if feature.isdigit():
        idx = int(feature)
        if idx < 0 or idx >= len(feature_names):
            raise ValueError(f"Feature index {idx} out of range 0..{len(feature_names) - 1}")
        return idx
    if feature not in feature_names:
        raise ValueError(f"Feature '{feature}' not found in dataset headers.")
    return feature_names.index(feature)


def main():
    parser = argparse.ArgumentParser(description="Neural net PDP/ICE with dropout for crop features.")
    parser.add_argument("--data", default="data/WinnipegDataset.txt", help="Path to dataset.")
    parser.add_argument("--feature", default="NDVI_Opt05July", help="Feature name or index.")
    parser.add_argument("--class", dest="target_class", type=int, default=1, help="Target class label.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--grid-points", type=int, default=25, help="Number of PDP grid points.")
    parser.add_argument("--subsample", type=int, default=50, help="ICE subsample size.")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed.")
    parser.add_argument("--hidden1", type=int, default=128, help="Hidden layer 1 size.")
    parser.add_argument("--hidden2", type=int, default=64, help="Hidden layer 2 size.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)

    crop_labels = {
        1: "corn",
        2: "peas",
        3: "canola",
        4: "soybeans",
        5: "oats",
        6: "wheat",
        7: "broadleaf",
    }
    crop_name = crop_labels.get(args.target_class, f"class{args.target_class}")

    X_train_raw, X_test_raw, y_train, y_test = load_data(args.data, colnames=COLNAMES)
    mask_train = ~pd.DataFrame(X_train_raw).isna().any(axis=1)
    mask_test = ~pd.DataFrame(X_test_raw).isna().any(axis=1)
    X_train_raw = X_train_raw[mask_train]
    y_train = y_train[mask_train]
    X_test_raw = X_test_raw[mask_test]
    y_test = y_test[mask_test]

    num_features = X_train_raw.shape[1]
    expected_features = COLNAMES[1:]
    if len(expected_features) != num_features:
        logging.warning(
            "Expected %d feature names but data has %d features; using generic names.",
            len(expected_features),
            num_features,
        )
        feature_names = [f"x{i}" for i in range(num_features)]
    else:
        feature_names = expected_features

    feature_idx = resolve_feature_index(args.feature, feature_names)
    feature_name = feature_names[feature_idx]

    classes = np.unique(y_train)
    num_classes = len(classes)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    if args.target_class not in class_to_index:
        raise ValueError(f"Target class {args.target_class} not found in training labels.")
    class_idx = class_to_index[args.target_class]
    y_train = np.array([class_to_index[c] for c in y_train])
    y_test = np.array([class_to_index[c] for c in y_test])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        X_train.shape[1],
        num_classes,
        args.hidden1,
        args.hidden2,
        args.dropout,
        args.seed,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(args.epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    train_acc = evaluate_model(model, X_train, y_train, device, args.batch_size)
    test_acc = evaluate_model(model, X_test, y_test, device, args.batch_size)
    logging.info("Train accuracy: %.4f", train_acc)
    logging.info("Test accuracy: %.4f", test_acc)
    logging.info("Train error: %.4f", 1.0 - train_acc)
    logging.info("Test error: %.4f", 1.0 - test_acc)

    feature_values = X_train_raw[:, feature_idx]
    grid = np.linspace(feature_values.min(), feature_values.max(), args.grid_points)
    pdp, ice = compute_pdp_ice(
        model,
        X_train_raw,
        scaler,
        feature_idx,
        class_idx,
        grid,
        args.subsample,
        args.seed,
        device,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(ice.shape[0]):
        label = "ICE (individual curves)" if i == 0 else None
        ax.plot(grid, ice[i], color="gray", linewidth=1, alpha=0.3, label=label)
    ax.plot(grid, pdp, color="red", linewidth=2, label="PDP (average)")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Partial Dependence (probability)")
    ax.set_title(f"PDP + ICE for {feature_name} ({crop_name})")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        "output",
        f"nn_pdp_dropout{args.dropout}_{feature_name}_{crop_name}_{timestamp}.png",
    )
    fig.savefig(out_path, dpi=150)
    logging.info("Saved PDP/ICE plot to %s", out_path)
    plt.show()


if __name__ == "__main__":
    main()
