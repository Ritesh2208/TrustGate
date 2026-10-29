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
    classification_report,
    confusion_matrix,
)


# ============================================================
# SETTINGS
# ============================================================

ANNOTATION_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

FEATURE_FILE = (
    "outputs/trustgate_100_master_901.json"
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
    "outputs/trustgate_independent_ablation.json"
)

LABELS = [
    "UNRELIABLE",
    "UNCERTAIN",
    "RELIABLE",
]

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2,
}


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("TRUSTGATE INDEPENDENT ABLATION EXPERIMENT")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    ANNOTATION_FILE,
    FEATURE_FILE,
    BASE_FEATURE_FILE,
    RETRIEVAL_FILE,
    VERIFIER_FILE,
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# LOAD FILES
# ============================================================

print("\nLoading independent annotations...")

annotations = pd.read_csv(
    ANNOTATION_FILE
)

print(
    "Annotation records:",
    len(annotations)
)


print("\nLoading 901-D TrustGate features...")

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    master = json.load(f)

print(
    "Master records:",
    len(master)
)


print("\nLoading hidden states...")

with open(
    BASE_FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    base = json.load(f)

print(
    "Base records:",
    len(base)
)


print("\nLoading retrieval signals...")

with open(
    RETRIEVAL_FILE,
    "r",
    encoding="utf-8"
) as f:

    retrieval = json.load(f)


print("\nLoading verifier signals...")

with open(
    VERIFIER_FILE,
    "r",
    encoding="utf-8"
) as f:

    verifier = json.load(f)


# ============================================================
# INDEX DATA BY ID
# ============================================================

master_by_id = {
    int(x["id"]): x
    for x in master
}

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


# ============================================================
# NORMALIZE ANNOTATIONS
# ============================================================

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
# BUILD DATA
# ============================================================

hidden_vectors = []
signal_vectors = []
fusion_vectors = []
labels = []
ids = []


print(
    "\nBuilding aligned representations..."
)


for _, row in annotations.iterrows():

    item_id = int(
        row["id"]
    )

    label = row[
        "human_label"
    ]

    if label not in LABEL_MAP:

        raise ValueError(
            f"Invalid label for ID "
            f"{item_id}: {label}"
        )


    # --------------------------------------------------------
    # Hidden state
    # --------------------------------------------------------

    base_item = base_by_id[item_id]

    hidden = np.asarray(
        base_item["hidden_state"],
        dtype=np.float32
    )


    if hidden.shape[0] != 896:

        raise ValueError(
            f"ID {item_id}: hidden state "
            f"has {hidden.shape[0]} dimensions."
        )


    # --------------------------------------------------------
    # Five trust signals
    # --------------------------------------------------------

    confidence = float(
        base_item["confidence"]
    )

    entropy = float(
        base_item["entropy"]
    )

    self_consistency = float(
        base_item["self_consistency"]
    )

    retrieval_agreement = float(
        retrieval_by_id[item_id][
            "retrieval_agreement"
        ]
    )

    verifier_score = float(
        verifier_by_id[item_id][
            "verifier_score"
        ]
    )


    signals = np.asarray(
        [
            confidence,
            entropy,
            self_consistency,
            retrieval_agreement,
            verifier_score,
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Existing 901-D vector
    # --------------------------------------------------------

    master_item = master_by_id[item_id]

    vector = np.asarray(
        master_item["feature_vector"],
        dtype=np.float32
    )


    if vector.shape[0] != 901:

        raise ValueError(
            f"ID {item_id}: feature vector "
            f"has {vector.shape[0]} dimensions."
        )


    # --------------------------------------------------------
    # Verify that the existing 901-D vector really contains
    # the 896 hidden dimensions + 5 trust signals.
    #
    # We locate each signal by its exact stored value.
    # --------------------------------------------------------

    signal_positions = []

    for signal in signals:

        positions = np.where(
            np.isclose(
                vector,
                signal,
                rtol=1e-5,
                atol=1e-6
            )
        )[0]

        if len(positions) == 0:

            raise ValueError(
                f"Could not locate trust signal "
                f"{signal} in 901-D vector "
                f"for ID {item_id}."
            )

        signal_positions.append(
            int(positions[0])
        )


    # Make sure the five signals occupy five
    # different positions.

    if len(
        set(signal_positions)
    ) != 5:

        raise ValueError(
            f"Signal positions overlap "
            f"for ID {item_id}: "
            f"{signal_positions}"
        )


    # --------------------------------------------------------
    # Remove the five signal positions.
    #
    # What remains must be exactly the 896-D hidden state.
    # --------------------------------------------------------

    hidden_from_master = np.delete(
        vector,
        signal_positions
    )


    if hidden_from_master.shape[0] != 896:

        raise ValueError(
            f"Removing five signals from "
            f"ID {item_id} did not produce "
            f"896 dimensions."
        )


    # --------------------------------------------------------
    # Numerical consistency check
    #
    # This verifies that the master 901-D representation
    # really contains the same hidden state.
    # --------------------------------------------------------

    if not np.allclose(
        hidden_from_master,
        hidden,
        rtol=1e-4,
        atol=1e-4
    ):

        raise ValueError(
            f"Hidden-state mismatch for "
            f"ID {item_id}."
        )


    hidden_vectors.append(
        hidden
    )

    signal_vectors.append(
        signals
    )

    fusion_vectors.append(
        vector
    )

    labels.append(
        LABEL_MAP[label]
    )

    ids.append(
        item_id
    )


# ============================================================
# MATRICES
# ============================================================

X_hidden = np.asarray(
    hidden_vectors,
    dtype=np.float32
)

X_signals = np.asarray(
    signal_vectors,
    dtype=np.float32
)

X_fusion = np.asarray(
    fusion_vectors,
    dtype=np.float32
)

y = np.asarray(
    labels,
    dtype=np.int64
)


# ============================================================
# DIMENSION CHECK
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "REPRESENTATION CHECK"
)

print(
    "=" * 60
)

print(
    "Hidden-only:",
    X_hidden.shape
)

print(
    "Signals-only:",
    X_signals.shape
)

print(
    "TrustGate fusion:",
    X_fusion.shape
)


if X_hidden.shape[1] != 896:
    raise ValueError(
        "Hidden representation is not 896-D."
    )

if X_signals.shape[1] != 5:
    raise ValueError(
        "Signal representation is not 5-D."
    )

if X_fusion.shape[1] != 901:
    raise ValueError(
        "Fusion representation is not 901-D."
    )


print(
    "\nCONFIRMED:"
)

print(
    "896 hidden + 5 signals = 901 fusion"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "INDEPENDENT LABEL DISTRIBUTION"
)

print(
    "=" * 60
)


for label in LABELS:

    count = int(
        np.sum(
            y == LABEL_MAP[label]
        )
    )

    print(
        f"{label:12s}: {count}"
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

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
# COMMON CV SPLITS
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
        "At least two examples per class "
        "are required."
    )


N_SPLITS = min(
    5,
    minimum_class_count
)


cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    X
):

    print(
        "\n" + "=" * 60
    )

    print(
        f"EVALUATING: {name}"
    )

    print(
        "=" * 60
    )


    model = create_model()


    predictions = cross_val_predict(
        model,
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
        labels=[
            0,
            1,
            2
        ],
        average="macro",
        zero_division=0
    )


    cm = confusion_matrix(
        y,
        predictions,
        labels=[
            0,
            1,
            2
        ]
    )


    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "Balanced accuracy:",
        round(
            balanced,
            4
        )
    )

    print(
        "Macro F1:",
        round(
            macro_f1,
            4
        )
    )


    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y,
            predictions,
            labels=[
                0,
                1,
                2
            ],
            target_names=LABELS,
            zero_division=0
        )
    )


    print(
        "Confusion matrix:"
    )

    print(
        cm
    )


    return {

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced),

        "macro_f1":
            float(macro_f1),

        "confusion_matrix":
            cm.tolist(),

        "predictions":
            predictions.tolist()
    }


# ============================================================
# RUN THREE MODELS
# ============================================================

results = {}


results["signals_only"] = evaluate(
    "SIGNALS ONLY (5-D)",
    X_signals
)


results["hidden_only"] = evaluate(
    "HIDDEN STATE ONLY (896-D)",
    X_hidden
)


results["trustgate_fusion"] = evaluate(
    "TRUSTGATE FUSION (901-D)",
    X_fusion
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRUSTGATE NOVELTY EXPERIMENT"
)

print(
    "=" * 60
)


print(
    f"\n{'Model':<25}"
    f"{'Accuracy':>12}"
    f"{'Balanced Acc.':>17}"
    f"{'Macro F1':>12}"
)

print(
    "-" * 66
)


for name, key in [
    ("Signals-only", "signals_only"),
    ("Hidden-only", "hidden_only"),
    ("TrustGate Fusion", "trustgate_fusion"),
]:

    r = results[key]

    print(
        f"{name:<25}"
        f"{r['accuracy']:>12.4f}"
        f"{r['balanced_accuracy']:>17.4f}"
        f"{r['macro_f1']:>12.4f}"
    )


# ============================================================
# FUSION DELTAS
# ============================================================

fusion = results[
    "trustgate_fusion"
]

signals = results[
    "signals_only"
]

hidden = results[
    "hidden_only"
]


print(
    "\n" + "=" * 60
)

print(
    "FUSION IMPROVEMENT / DIFFERENCE"
)

print(
    "=" * 60
)


print(
    "\nFusion vs Signals-only:"
)

print(
    "Balanced accuracy:",
    f"{fusion['balanced_accuracy'] - signals['balanced_accuracy']:+.4f}"
)

print(
    "Macro F1:",
    f"{fusion['macro_f1'] - signals['macro_f1']:+.4f}"
)


print(
    "\nFusion vs Hidden-only:"
)

print(
    "Balanced accuracy:",
    f"{fusion['balanced_accuracy'] - hidden['balanced_accuracy']:+.4f}"
)

print(
    "Macro F1:",
    f"{fusion['macro_f1'] - hidden['macro_f1']:+.4f}"
)


# ============================================================
# SAVE
# ============================================================

output = {

    "experiment":
        "Independent annotation ablation",

    "annotation_file":
        ANNOTATION_FILE,

    "feature_file":
        FEATURE_FILE,

    "records":
        len(y),

    "dimensions": {

        "signals_only":
            5,

        "hidden_only":
            896,

        "trustgate_fusion":
            901
    },

    "cv_folds":
        N_SPLITS,

    "label_distribution": {

        "UNRELIABLE":
            int(class_counts[0]),

        "UNCERTAIN":
            int(class_counts[1]),

        "RELIABLE":
            int(class_counts[2])
    },

    "results":
        results,

    "fusion_vs_signals": {

        "balanced_accuracy_delta":
            float(
                fusion["balanced_accuracy"]
                -
                signals["balanced_accuracy"]
            ),

        "macro_f1_delta":
            float(
                fusion["macro_f1"]
                -
                signals["macro_f1"]
            )
    },

    "fusion_vs_hidden": {

        "balanced_accuracy_delta":
            float(
                fusion["balanced_accuracy"]
                -
                hidden["balanced_accuracy"]
            ),

        "macro_f1_delta":
            float(
                fusion["macro_f1"]
                -
                hidden["macro_f1"]
            )
    }
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
    "INDEPENDENT ABLATION COMPLETE"
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
    "\nNo existing model or dataset was modified."
)

print("=" * 60)