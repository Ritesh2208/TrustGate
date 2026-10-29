import json
import os
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


INPUT_FILE = "outputs/trustgate_standardized_500_labeled.json"
MODEL_FILE = "models/trustgate_500_22d_linear_svm.joblib"


LABELS = [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE"
]


def load_data():
    print("=" * 60)
    print("LOADING TRUSTGATE 500 DATA")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    X_hidden = []
    X_signals = []
    y = []

    for item in data:

        hidden = item["hidden_state"]

        if len(hidden) != 896:
            raise ValueError(
                f"ID {item['id']} has hidden dimension "
                f"{len(hidden)}, expected 896."
            )

        confidence = float(item["confidence"])
        entropy = float(item["entropy"])

        label = item["label"]

        if label not in LABELS:
            raise ValueError(
                f"Invalid label for ID {item['id']}: {label}"
            )

        X_hidden.append(hidden)
        X_signals.append([
            confidence,
            entropy
        ])
        y.append(label)

    X_hidden = np.asarray(X_hidden, dtype=np.float32)
    X_signals = np.asarray(X_signals, dtype=np.float32)
    y = np.asarray(y)

    print(f"Rows:              {len(y)}")
    print(f"Hidden dimensions: {X_hidden.shape[1]}")
    print(f"Signal dimensions: {X_signals.shape[1]}")

    print("\nLabel distribution:")

    for label in LABELS:
        print(
            f"  {label:<12}: "
            f"{np.sum(y == label)}"
        )

    return X_hidden, X_signals, y


def build_22d_representation(X_hidden, X_signals):
    """
    Convert:

        896-D hidden state
            ↓
        StandardScaler
            ↓
        PCA(20)
            ↓
        20-D hidden representation
            +
        confidence
            +
        entropy
            ↓
        22-D Trust Representation
    """

    print("\n" + "=" * 60)
    print("BUILDING 22-D TRUST REPRESENTATION")
    print("=" * 60)

    hidden_scaler = StandardScaler()

    X_hidden_scaled = hidden_scaler.fit_transform(
        X_hidden
    )

    pca = PCA(
        n_components=20,
        random_state=42
    )

    X_hidden_pca = pca.fit_transform(
        X_hidden_scaled
    )

    X_22d = np.hstack([
        X_hidden_pca,
        X_signals
    ])

    print(f"Original hidden: {X_hidden.shape}")
    print(f"PCA hidden:      {X_hidden_pca.shape}")
    print(f"Final 22-D:      {X_22d.shape}")

    print(
        f"PCA explained variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    return (
        X_22d,
        hidden_scaler,
        pca
    )


def train_model(X_22d, y):
    print("\n" + "=" * 60)
    print("TRAINING 22-D LINEAR SVM")
    print("=" * 60)

    # Scale the final 22-D representation.
    final_scaler = StandardScaler()

    X_scaled = final_scaler.fit_transform(
        X_22d
    )

    classifier = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=10000
    )

    classifier.fit(
        X_scaled,
        y
    )

    predictions = classifier.predict(
        X_scaled
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    print(f"Training accuracy:          {accuracy:.4f}")
    print(f"Training balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Training macro F1:           {macro_f1:.4f}")

    return final_scaler, classifier


def save_model(
    hidden_scaler,
    pca,
    final_scaler,
    classifier
):

    os.makedirs(
        os.path.dirname(MODEL_FILE),
        exist_ok=True
    )

    model_bundle = {
        "model_type": "TrustGate_22D_LinearSVM",

        "input_hidden_dimension": 896,

        "pca_dimension": 20,

        "final_dimension": 22,

        "signal_features": [
            "confidence",
            "entropy"
        ],

        "labels": LABELS,

        "hidden_scaler": hidden_scaler,

        "pca": pca,

        "final_scaler": final_scaler,

        "classifier": classifier
    }

    joblib.dump(
        model_bundle,
        MODEL_FILE
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(MODEL_FILE)


def main():

    X_hidden, X_signals, y = load_data()

    (
        X_22d,
        hidden_scaler,
        pca
    ) = build_22d_representation(
        X_hidden,
        X_signals
    )

    (
        final_scaler,
        classifier
    ) = train_model(
        X_22d,
        y
    )

    save_model(
        hidden_scaler,
        pca,
        final_scaler,
        classifier
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()