import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


DATA_FILE = "outputs/trustgate_standardized_500_labeled.json"


print("=" * 70)
print("TRUSTGATE CLASS-WEIGHT EXPERIMENT")
print("=" * 70)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


X_hidden = []
X_signal = []
y = []


for item in data:

    X_hidden.append(
        np.asarray(
            item["hidden_state"],
            dtype=np.float32
        )
    )

    X_signal.append([
        float(item["confidence"]),
        float(item["entropy"])
    ])

    y.append(item["label"])


X_hidden = np.asarray(X_hidden, dtype=np.float32)
X_signal = np.asarray(X_signal, dtype=np.float32)
y = np.asarray(y)


# ------------------------------------------------------------
# Build 22-D representation inside each CV fold
# ------------------------------------------------------------

class TrustRepresentation:

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(
            n_components=20,
            random_state=42
        )
        self.final_scaler = StandardScaler()

    def fit_transform(self, X):

        hidden = X[:, :896]
        signals = X[:, 896:]

        hidden_scaled = self.scaler.fit_transform(hidden)

        hidden_pca = self.pca.fit_transform(
            hidden_scaled
        )

        fused = np.hstack([
            hidden_pca,
            signals
        ])

        return self.final_scaler.fit_transform(
            fused
        )

    def transform(self, X):

        hidden = X[:, :896]
        signals = X[:, 896:]

        hidden_scaled = self.scaler.transform(hidden)

        hidden_pca = self.pca.transform(
            hidden_scaled
        )

        fused = np.hstack([
            hidden_pca,
            signals
        ])

        return self.final_scaler.transform(
            fused
        )


X = np.hstack([
    X_hidden,
    X_signal
])


# ------------------------------------------------------------
# Candidate class weights
# ------------------------------------------------------------

weights = {

    "balanced": "balanced",

    "custom_1":
        {
            "RELIABLE": 1.0,
            "UNCERTAIN": 1.5,
            "UNRELIABLE": 1.0
        },

    "custom_2":
        {
            "RELIABLE": 1.5,
            "UNCERTAIN": 2.0,
            "UNRELIABLE": 1.0
        },

    "custom_3":
        {
            "RELIABLE": 2.0,
            "UNCERTAIN": 2.5,
            "UNRELIABLE": 1.0
        }
}


# ------------------------------------------------------------
# CV
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


for name, class_weight in weights.items():

    print()
    print("-" * 70)
    print("Testing:", name)
    print("Weights:", class_weight)

    fold_balanced = []
    fold_macro = []
    fold_accuracy = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        representation = TrustRepresentation()

        X_train_final = representation.fit_transform(
            X_train
        )

        X_test_final = representation.transform(
            X_test
        )

        classifier = LinearSVC(
            class_weight=class_weight,
            random_state=42,
            max_iter=10000
        )

        classifier.fit(
            X_train_final,
            y_train
        )

        predictions = classifier.predict(
            X_test_final
        )

        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        balanced = balanced_accuracy_score(
            y_test,
            predictions
        )

        macro = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        fold_accuracy.append(accuracy)
        fold_balanced.append(balanced)
        fold_macro.append(macro)

    result = {
        "name": name,
        "weights": class_weight,
        "accuracy": float(
            np.mean(fold_accuracy)
        ),
        "balanced_accuracy": float(
            np.mean(fold_balanced)
        ),
        "macro_f1": float(
            np.mean(fold_macro)
        )
    }

    results.append(result)

    print(
        f"Accuracy:          {result['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: {result['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:          {result['macro_f1']:.4f}"
    )


# ------------------------------------------------------------
# Select based on macro F1
# ------------------------------------------------------------

best = max(
    results,
    key=lambda x: (
        x["macro_f1"],
        x["balanced_accuracy"]
    )
)


print()
print("=" * 70)
print("BEST VALIDATED CONFIGURATION")
print("=" * 70)

print(
    "Configuration:",
    best["name"]
)

print(
    "Weights:",
    best["weights"]
)

print(
    f"Accuracy:          {best['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy: {best['balanced_accuracy']:.4f}"
)

print(
    f"Macro F1:          {best['macro_f1']:.4f}"
)


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

output = {
    "all_results": results,
    "selected": best
}


with open(
    "outputs/trustgate_class_weight_experiment.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


print()
print(
    "Saved:"
    " outputs/trustgate_class_weight_experiment.json"
)

print("=" * 70)