
import json
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# TRUSTGATE: FROZEN LINEAR SVM + PCA20
# INDEPENDENT NEXT-100 EVALUATION
# ============================================================

TRAIN_ORIGINAL_LABEL_FILE = "outputs/trustgate_human_evaluation_completed.csv"
TRAIN_NEXT_LABEL_FILE = "outputs/trustgate_human_evaluation_next_100_completed.csv"

ORIGINAL_FEATURE_FILE = "outputs/truthfulqa_100_combined_features.json"
NEXT_FEATURE_FILE = "outputs/trustgate_next_100_master_901.json"

OUTPUT_CSV = "outputs/trustgate_linear_svm_next100_predictions.csv"
OUTPUT_JSON = "outputs/trustgate_linear_svm_next100_evaluation.json"

SEED = 42

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2,
}

LABEL_NAMES = ["UNRELIABLE", "UNCERTAIN", "RELIABLE"]

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(x):
    return str(x).strip().upper()


def get_hidden(item):
    hidden = item.get("hidden_state", item.get("hidden_features"))

    if hidden is None:
        raise ValueError(f"No hidden representation for ID {item.get('id')}")

    hidden = np.asarray(hidden, dtype=np.float32)

    if hidden.size != 896:
        raise ValueError(
            f"Expected 896 hidden features for ID {item.get('id')}, "
            f"got {hidden.size}"
        )

    return hidden


def get_signal(item, name):
    # Original feature format:
    # confidence / entropy are top-level fields.
    if name in item:
        return float(item.get(name, 0.0))

    # Next-100 feature format:
    # signals are inside the "signals" dictionary.
    signals = item.get("signals", {})

    if isinstance(signals, dict):
        return float(signals.get(name, 0.0))

    return 0.0


def make_898(item):
    hidden = get_hidden(item)

    confidence = get_signal(item, "confidence")
    entropy = get_signal(item, "entropy")

    return np.concatenate(
        [
            hidden,
            np.array([confidence, entropy], dtype=np.float32),
        ]
    ).astype(np.float32)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 72)
print("TRUSTGATE — FROZEN LINEAR SVM + PCA20")
print("INDEPENDENT NEXT-100 EVALUATION")
print("=" * 72)

original_labels = pd.read_csv(TRAIN_ORIGINAL_LABEL_FILE)
next_labels = pd.read_csv(TRAIN_NEXT_LABEL_FILE)

original_features = load_json(ORIGINAL_FEATURE_FILE)
next_features = load_json(NEXT_FEATURE_FILE)

original_map = {str(x["id"]): x for x in original_features}
next_map = {str(x["id"]): x for x in next_features}


def build_dataset(labels_df, feature_map):
    X = []
    y = []
    ids = []

    for _, row in labels_df.iterrows():
        item_id = str(row["id"])
        label = normalize_label(row["human_label"])

        if label not in LABEL_MAP:
            raise ValueError(
                f"Unknown label '{label}' for ID {item_id}"
            )

        if item_id not in feature_map:
            raise ValueError(
                f"Feature record missing for ID {item_id}"
            )

        X.append(make_898(feature_map[item_id]))
        y.append(LABEL_MAP[label])
        ids.append(item_id)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        ids,
    )


X_original, y_original, original_ids = build_dataset(
    original_labels,
    original_map,
)

X_next, y_next, next_ids = build_dataset(
    next_labels,
    next_map,
)

X_train = np.concatenate([X_original, X_next], axis=0)
y_train = np.concatenate([y_original, y_next], axis=0)

print("\nTraining set:")
print("  Samples:", len(X_train))
print("  Features:", X_train.shape[1])

print("\nIndependent test set:")
print("  Samples:", len(X_next))
print("  Features:", X_next.shape[1])

print("\nTraining label distribution:")
for name, value in LABEL_MAP.items():
    print(f"  {name}: {int(np.sum(y_train == value))}")

print("\nNext-100 true label distribution:")
for name, value in LABEL_MAP.items():
    print(f"  {name}: {int(np.sum(y_next == value))}")


# ============================================================
# IMPORTANT
# ============================================================
# The optimization selected LinearSVM + PCA20 using the 200-sample
# CV experiment.
#
# Therefore, for this independent evaluation, we train the selected
# configuration on the available 200 labeled samples and evaluate
# on the independent Next-100 only if that Next-100 is NOT part of
# those 200 training labels.
#
# To avoid accidental test contamination, we explicitly use the
# ORIGINAL 100 as training data here and NEXT 100 as test data.
#
# This is the correct independent generalization experiment when
# Next-100 is the held-out set.
# ============================================================

X_fit = X_original
y_fit = y_original

print("\nFrozen evaluation protocol:")
print("  Train: Original 100")
print("  Test:  Independent Next-100")
print("  Pipeline: StandardScaler -> PCA20 -> LinearSVM")


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=SEED)),
        (
            "model",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=10000,
                random_state=SEED,
            ),
        ),
    ]
)

model.fit(X_fit, y_fit)

pred = model.predict(X_next)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_next, pred)
balanced = balanced_accuracy_score(y_next, pred)

macro_f1 = f1_score(
    y_next,
    pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_next,
    pred,
    average="weighted",
    zero_division=0,
)

cm = confusion_matrix(
    y_next,
    pred,
    labels=[0, 1, 2],
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 72)
print("INDEPENDENT NEXT-100 RESULTS")
print("=" * 72)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")

print("\nConfusion Matrix")
print("Rows = True, Columns = Predicted")
print("Labels:", LABEL_NAMES)
print(cm)

print("\nPrediction distribution:")
for name, value in LABEL_MAP.items():
    print(f"  {name}: {int(np.sum(pred == value))}")


# Save predictions
prediction_df = pd.DataFrame(
    {
        "id": next_ids,
        "true_label": [LABEL_NAMES[x] for x in y_next],
        "predicted_label": [LABEL_NAMES[x] for x in pred],
        "correct": (y_next == pred),
    }
)

prediction_df.to_csv(OUTPUT_CSV, index=False)

result = {
    "model": "LinearSVM",
    "pca_components": 20,
    "features": "896 hidden + confidence + entropy",
    "feature_dimension": 898,
    "train_samples": int(len(X_fit)),
    "test_samples": int(len(X_next)),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "labels": LABEL_NAMES,
    "confusion_matrix": cm.tolist(),
    "train_label_distribution": {
        name: int(np.sum(y_fit == value))
        for name, value in LABEL_MAP.items()
    },
    "test_label_distribution": {
        name: int(np.sum(y_next == value))
        for name, value in LABEL_MAP.items()
    },
    "prediction_distribution": {
        name: int(np.sum(pred == value))
        for name, value in LABEL_MAP.items()
    },
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)


print("\nSaved:")
print(" ", OUTPUT_CSV)
print(" ", OUTPUT_JSON)
