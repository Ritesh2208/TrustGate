import json
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


FEATURE_FILE = (
    "outputs/trustgate_standardized_500_labeled.json"
)


# ============================================================
# LOAD DATA
# ============================================================

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


labels = [
    item["label"]
    for item in data
]


# ============================================================
# BUILD COMPONENTS
# ============================================================

hidden = np.array(
    [
        item["hidden_state"]
        for item in data
    ],
    dtype=np.float32,
)

signals = np.array(
    [
        [
            item["confidence"],
            item["entropy"],
        ]
        for item in data
    ],
    dtype=np.float32,
)

fusion = np.concatenate(
    [hidden, signals],
    axis=1,
)


print("=" * 72)
print("TRUST COMPONENT COMPARISON - 500 SAMPLES")
print("=" * 72)

print("Hidden:", hidden.shape)
print("Signals:", signals.shape)
print("Fusion:", fusion.shape)


# ============================================================
# MODELS
# ============================================================

models = {

    "Hidden only": (
        hidden,
        Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=20)),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ])
    ),

    "Confidence + Entropy": (
        signals,
        Pipeline([
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ])
    ),

    "Hidden + Confidence + Entropy": (
        fusion,
        Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=20)),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ])
    ),
}


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


for name, (X, model) in models.items():

    accuracies = []
    balanced_scores = []
    macro_f1_scores = []

    for train_idx, test_idx in cv.split(
        X,
        labels,
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = [
            labels[i]
            for i in train_idx
        ]

        y_test = [
            labels[i]
            for i in test_idx
        ]

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        accuracies.append(
            accuracy_score(
                y_test,
                predictions,
            )
        )

        balanced_scores.append(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        )

        macro_f1_scores.append(
            f1_score(
                y_test,
                predictions,
                average="macro",
            )
        )

    print("\n" + "-" * 72)
    print(name)

    print(
        "Accuracy:",
        round(
            np.mean(accuracies),
            4,
        ),
    )

    print(
        "Balanced Accuracy:",
        round(
            np.mean(balanced_scores),
            4,
        ),
    )

    print(
        "Macro F1:",
        round(
            np.mean(macro_f1_scores),
            4,
        ),
    )
