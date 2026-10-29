import json
import os

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# SETTINGS
# ============================================================

ANNOTATION_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

BASE_FEATURE_FILE = (
    "outputs/truthfulqa_100_combined_features.json"
)

RETRIEVAL_FILE = (
    "outputs/truthfulqa_100_retrieval_v3.json"
)

VERIFIER_FILE = (
    "outputs/truthfulqa_100_verified.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_signal_contribution.json"
)

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2,
}


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("TRUSTGATE SIGNAL CONTRIBUTION STUDY")
print("=" * 60)

annotations = pd.read_csv(
    ANNOTATION_FILE
)

with open(
    BASE_FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:
    base = json.load(f)

with open(
    RETRIEVAL_FILE,
    "r",
    encoding="utf-8"
) as f:
    retrieval = json.load(f)

with open(
    VERIFIER_FILE,
    "r",
    encoding="utf-8"
) as f:
    verifier = json.load(f)


print(
    "\nAnnotations:",
    len(annotations)
)

print(
    "Base records:",
    len(base)
)

print(
    "Retrieval records:",
    len(retrieval)
)

print(
    "Verifier records:",
    len(verifier)
)


# ============================================================
# INDEX
# ============================================================

base_by_id = {
    int(x["id"]): x
    for x in base
}

retrieval_by_id = {
    int(x["id"]): x
    for x in retrieval
}

verifier_by_id = {
    int(x["id"]): x
    for x in verifier
}


annotations["id"] = (
    pd.to_numeric(
        annotations["id"]
    ).astype(int)
)

annotations["human_label"] = (
    annotations["human_label"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# BUILD MATRICES
# ============================================================

hidden = []
confidence = []
entropy = []
self_consistency = []
retrieval_agreement = []
verifier_score = []
y = []


for _, row in annotations.iterrows():

    item_id = int(
        row["id"]
    )

    label = row["human_label"]

    if label not in LABEL_MAP:
        raise ValueError(
            f"Invalid label for ID {item_id}: {label}"
        )

    item = base_by_id[item_id]

    h = np.asarray(
        item["hidden_state"],
        dtype=np.float32
    )

    if len(h) != 896:
        raise ValueError(
            f"ID {item_id}: expected 896 hidden "
            f"features, got {len(h)}"
        )

    hidden.append(h)

    confidence.append(
        float(item["confidence"])
    )

    entropy.append(
        float(item["entropy"])
    )

    self_consistency.append(
        float(item["self_consistency"])
    )

    retrieval_agreement.append(
        float(
            retrieval_by_id[item_id][
                "retrieval_agreement"
            ]
        )
    )

    verifier_score.append(
        float(
            verifier_by_id[item_id][
                "verifier_score"
            ]
        )
    )

    y.append(
        LABEL_MAP[label]
    )


hidden = np.asarray(
    hidden,
    dtype=np.float32
)

signals = np.column_stack(
    [
        confidence,
        entropy,
        self_consistency,
        retrieval_agreement,
        verifier_score,
    ]
).astype(np.float32)

y = np.asarray(
    y,
    dtype=np.int64
)


print(
    "\nHidden:",
    hidden.shape
)

print(
    "Signals:",
    signals.shape
)


# ============================================================
# COMMON CV
# ============================================================

class_counts = np.bincount(
    y,
    minlength=3
)

minimum_class_count = int(
    class_counts.min()
)

if minimum_class_count < 2:
    raise ValueError(
        "At least two examples per class are required."
    )

n_splits = min(
    5,
    minimum_class_count
)

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)


# ============================================================
# MODEL
# ============================================================

def make_model():

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    name,
    X
):

    predictions = cross_val_predict(
        make_model(),
        X,
        y,
        cv=cv,
        method="predict"
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced = balanced_accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    print(
        f"{name:<35}"
        f"{accuracy:.4f}     "
        f"{balanced:.4f}     "
        f"{macro_f1:.4f}"
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "macro_f1": float(macro_f1),
        "predictions": predictions.tolist()
    }


# ============================================================
# EXPERIMENTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "INCREMENTAL SIGNAL EXPERIMENT"
)

print(
    "=" * 60
)

print(
    f"\n{'Representation':<35}"
    f"Accuracy     "
    f"Balanced Acc.     "
    f"Macro F1"
)

print("-" * 85)


results = {}


# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

results["hidden_only"] = evaluate(
    "Hidden only (896-D)",
    hidden
)


# ------------------------------------------------------------
# Individual signals
# ------------------------------------------------------------

signal_names = [
    "confidence",
    "entropy",
    "self_consistency",
    "retrieval_agreement",
    "verifier_score",
]


for i, signal_name in enumerate(
    signal_names
):

    X = np.column_stack(
        [
            hidden,
            signals[:, i]
        ]
    )

    results[
        f"hidden_plus_{signal_name}"
    ] = evaluate(
        f"Hidden + {signal_name}",
        X
    )


# ------------------------------------------------------------
# All signals
# ------------------------------------------------------------

fusion = np.hstack(
    [
        hidden,
        signals
    ]
)

results["full_fusion"] = evaluate(
    "Hidden + ALL 5 signals",
    fusion
)


# ============================================================
# DELTAS
# ============================================================

baseline = results[
    "hidden_only"
]

print(
    "\n" + "=" * 60
)

print(
    "DELTA FROM HIDDEN-ONLY BASELINE"
)

print(
    "=" * 60
)

print(
    f"\n{'Representation':<35}"
    f"Δ Balanced Acc.     "
    f"Δ Macro F1"
)

print("-" * 65)


delta_results = {}


for name, result in results.items():

    if name == "hidden_only":
        continue

    delta_balanced = (
        result["balanced_accuracy"]
        -
        baseline["balanced_accuracy"]
    )

    delta_f1 = (
        result["macro_f1"]
        -
        baseline["macro_f1"]
    )

    print(
        f"{name:<35}"
        f"{delta_balanced:+.4f}              "
        f"{delta_f1:+.4f}"
    )

    delta_results[name] = {
        "balanced_accuracy_delta":
            float(delta_balanced),
        "macro_f1_delta":
            float(delta_f1)
    }


# ============================================================
# BEST CONTRIBUTION
# ============================================================

best_signal = max(
    delta_results,
    key=lambda x:
        delta_results[x][
            "balanced_accuracy_delta"
        ]
)


print(
    "\n" + "=" * 60
)

print(
    "BEST INCREMENTAL CONTRIBUTION"
)

print(
    "=" * 60
)

print(
    best_signal
)

print(
    "Balanced accuracy delta:",
    f"{delta_results[best_signal]['balanced_accuracy_delta']:+.4f}"
)

print(
    "Macro F1 delta:",
    f"{delta_results[best_signal]['macro_f1_delta']:+.4f}"
)


# ============================================================
# SAVE
# ============================================================

output = {
    "experiment":
        "Incremental TrustGate signal contribution",

    "annotation_file":
        ANNOTATION_FILE,

    "records":
        int(len(y)),

    "hidden_dimension":
        896,

    "signal_dimension":
        5,

    "cv_folds":
        n_splits,

    "label_distribution": {
        "UNRELIABLE":
            int(class_counts[0]),
        "UNCERTAIN":
            int(class_counts[1]),
        "RELIABLE":
            int(class_counts[2]),
    },

    "signal_names":
        signal_names,

    "results":
        results,

    "deltas_from_hidden":
        delta_results,

    "best_incremental_signal":
        best_signal,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "SIGNAL CONTRIBUTION STUDY COMPLETE"
)

print(
    "=" * 60
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nExisting TrustGate files were not modified."
)

print("=" * 60)