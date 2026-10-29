import json
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)


INPUT_FILE = "outputs/trustgate_standardized_500_labeled.json"


# ============================================================
# Load data
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

hidden = np.array(
    [x["hidden_state"] for x in data],
    dtype=np.float32
)

confidence = np.array(
    [x["confidence"] for x in data],
    dtype=np.float32
).reshape(-1, 1)

entropy = np.array(
    [x["entropy"] for x in data],
    dtype=np.float32
).reshape(-1, 1)

y = np.array(
    [x["label"] for x in data]
)


# ============================================================
# 896 hidden -> 20 dimensions
# ============================================================

hidden_scaled = StandardScaler().fit_transform(hidden)

pca = PCA(
    n_components=20,
    random_state=42
)

hidden_20 = pca.fit_transform(hidden_scaled)

X = np.hstack([
    hidden_20,
    confidence,
    entropy
])

print("X shape:", X.shape)
print("Label distribution:")

for label in ["RELIABLE", "UNCERTAIN", "UNRELIABLE"]:
    print(
        label,
        np.sum(y == label)
    )


# ============================================================
# Cross validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# Models
# ============================================================

models = {

    "SVM balanced": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        ))
    ]),

    "SVM UNCERTAIN emphasis": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            class_weight={
                "RELIABLE": 1.5,
                "UNCERTAIN": 3.0,
                "UNRELIABLE": 1.0
            },
            random_state=42,
            max_iter=10000
        ))
    ]),

    "SVM stronger UNCERTAIN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            class_weight={
                "RELIABLE": 1.5,
                "UNCERTAIN": 5.0,
                "UNRELIABLE": 1.0
            },
            random_state=42,
            max_iter=10000
        ))
    ]),

    "Logistic balanced": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ))
    ])
}


# ============================================================
# Evaluate
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    acc = accuracy_score(y, pred)

    bal = balanced_accuracy_score(
        y,
        pred
    )

    macro = f1_score(
        y,
        pred,
        average="macro"
    )

    results.append(
        (name, acc, bal, macro)
    )

    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal:.4f}")
    print(f"Macro F1:          {macro:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            pred,
            labels=[
                "RELIABLE",
                "UNCERTAIN",
                "UNRELIABLE"
            ],
            zero_division=0
        )
    )


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"{'Model':30s}"
    f"{'Accuracy':>12s}"
    f"{'Balanced':>12s}"
    f"{'Macro F1':>12s}"
)

for name, acc, bal, macro in results:

    print(
        f"{name:30s}"
        f"{acc:12.4f}"
        f"{bal:12.4f}"
        f"{macro:12.4f}"
    )