import json
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = (
    "outputs/trustgate_100_labeled_v3.json"
)

ANNOTATION_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

OUTPUT_FILE = (
    "outputs/trustgate_independent_annotation_results.json"
)


LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2,
}

REVERSE_MAP = {
    0: "UNRELIABLE",
    1: "UNCERTAIN",
    2: "RELIABLE",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("TRUSTGATE INDEPENDENT ANNOTATION EVALUATION")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(FEATURE_FILE):

    raise FileNotFoundError(
        f"Missing feature file:\n{FEATURE_FILE}"
    )

if not os.path.exists(ANNOTATION_FILE):

    raise FileNotFoundError(
        f"Missing annotation file:\n{ANNOTATION_FILE}"
    )


# ============================================================
# LOAD TRUSTGATE FEATURES
# ============================================================

print("\nLoading TrustGate feature data...")

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_data = json.load(f)


print(
    "Feature records:",
    len(feature_data)
)


# ============================================================
# LOAD INDEPENDENT ANNOTATIONS
# ============================================================

print("\nLoading independent annotations...")

annotations = pd.read_csv(
    ANNOTATION_FILE
)


print(
    "Annotation records:",
    len(annotations)
)


# ============================================================
# VALIDATION
# ============================================================

if len(feature_data) != 100:

    raise ValueError(
        "Expected 100 feature records."
    )

if len(annotations) != 100:

    raise ValueError(
        "Expected 100 annotation records."
    )


required_columns = [
    "id",
    "question",
    "human_label",
    "human_confidence",
]


for column in required_columns:

    if column not in annotations.columns:

        raise ValueError(
            f"Missing annotation column: {column}"
        )


# ============================================================
# BUILD FEATURE DICTIONARY
# ============================================================

feature_by_id = {}

for item in feature_data:

    item_id = int(
        item["id"]
    )

    vector = item.get(
        "feature_vector"
    )

    if vector is None:

        raise ValueError(
            f"Missing feature_vector "
            f"for ID {item_id}"
        )


    if len(vector) != 901:

        raise ValueError(
            f"ID {item_id} has "
            f"{len(vector)} features. "
            f"Expected 901."
        )


    feature_by_id[item_id] = (
        np.asarray(
            vector,
            dtype=np.float32
        )
    )


# ============================================================
# BUILD ALIGNED DATASET
# ============================================================

X = []
y = []
ids = []


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


for _, row in annotations.iterrows():

    item_id = int(
        row["id"]
    )

    label = row[
        "human_label"
    ]


    if item_id not in feature_by_id:

        raise ValueError(
            f"No feature vector for ID {item_id}"
        )


    if label not in LABEL_MAP:

        raise ValueError(
            f"Invalid human label "
            f"for ID {item_id}: {label}"
        )


    X.append(
        feature_by_id[item_id]
    )

    y.append(
        LABEL_MAP[label]
    )

    ids.append(
        item_id
    )


X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int64
)


# ============================================================
# DATA SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "INDEPENDENT ANNOTATION DATA"
)

print(
    "=" * 60
)

print(
    "Feature matrix:",
    X.shape
)


print(
    "\nHuman label distribution:"
)


for label_id in [
    0,
    1,
    2
]:

    count = int(
        np.sum(
            y == label_id
        )
    )

    print(
        f"{REVERSE_MAP[label_id]:12s}: "
        f"{count}"
    )


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ============================================================
# CROSS-VALIDATION
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


n_splits = min(
    5,
    minimum_class_count
)


cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)


print(
    "\n" + "=" * 60
)

print(
    "CROSS-VALIDATION AGAINST "
    "INDEPENDENT ANNOTATIONS"
)

print(
    "=" * 60
)

print(
    "Cross-validation folds:",
    n_splits
)


print(
    "\nGenerating out-of-fold predictions..."
)


predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict"
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    predictions
)


balanced_accuracy = (
    balanced_accuracy_score(
        y,
        predictions
    )
)


macro_f1 = f1_score(
    y,
    predictions,
    average="macro",
    zero_division=0
)


print(
    "\n" + "=" * 60
)

print(
    "INDEPENDENT ANNOTATION RESULTS"
)

print(
    "=" * 60
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
        balanced_accuracy,
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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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
        target_names=[
            "UNRELIABLE",
            "UNCERTAIN",
            "RELIABLE"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

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
    "Confusion matrix:"
)

print(
    cm
)


# ============================================================
# PER-QUESTION RESULTS
# ============================================================

question_results = []


for i, item_id in enumerate(ids):

    question_row = annotations[
        annotations["id"] == item_id
    ].iloc[0]


    question_results.append({

        "id": int(item_id),

        "question": str(
            question_row["question"]
        ),

        "human_label":
            REVERSE_MAP[
                int(y[i])
            ],

        "predicted_label":
            REVERSE_MAP[
                int(predictions[i])
            ],

        "correct":
            bool(
                y[i] ==
                predictions[i]
            ),

        "human_confidence":
            int(
                question_row[
                    "human_confidence"
                ]
            ),
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "TrustGate evaluation "
        "against independent annotations",

    "feature_source":
        FEATURE_FILE,

    "annotation_source":
        ANNOTATION_FILE,

    "records":
        100,

    "feature_dimension":
        901,

    "label_distribution": {

        "UNRELIABLE":
            int(class_counts[0]),

        "UNCERTAIN":
            int(class_counts[1]),

        "RELIABLE":
            int(class_counts[2]),
    },

    "cross_validation_folds":
        n_splits,

    "accuracy":
        float(accuracy),

    "balanced_accuracy":
        float(balanced_accuracy),

    "macro_f1":
        float(macro_f1),

    "confusion_matrix":
        cm.tolist(),

    "question_results":
        question_results,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
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
    "INDEPENDENT ANNOTATION "
    "EVALUATION COMPLETE"
)

print(
    "=" * 60
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nIMPORTANT:"
)

print(
    "The model was NOT retrained "
    "on the independent annotations."
)

print(
    "The 901-D TrustGate representation "
    "from the existing pipeline was evaluated "
    "against the new labels."
)

print("=" * 60)