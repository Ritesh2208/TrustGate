import json
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/trustgate_100_labeled_v3.json"
)

OUTPUT_DIR = "outputs"

MODEL_FILE = (
    "outputs/trustgate_model.json"
)

SCORES_FILE = (
    "outputs/trustgate_100_scores.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING TRUSTGATE DATA")
print("=" * 60)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "Records:",
    len(data)
)


# ============================================================
# IMPORTANT:
# We use the V2 labels only for INITIAL EVALUATION.
#
# The final project should replace these with independently
# human-verified labels when available.
# ============================================================

label_map = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2
}


# ============================================================
# BUILD FEATURES
# ============================================================

X = []
y = []

ids = []
questions = []

for item in data:

    vector = item[
        "feature_vector"
    ]

    if len(vector) != 901:

        raise ValueError(
            f"ID {item['id']} "
            f"has {len(vector)} features. "
            f"Expected 901."
        )


    X.append(
        vector
    )

    label = item[
        "reliability_label"
    ]

    if label not in label_map:

        raise ValueError(
            f"Unknown label: {label}"
        )


    y.append(
        label_map[label]
    )

    ids.append(
        item["id"]
    )

    questions.append(
        item["question"]
    )


X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int64
)


print(
    "Feature matrix:",
    X.shape
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "LABEL DISTRIBUTION"
)

print(
    "=" * 60
)


unique, counts = np.unique(
    y,
    return_counts=True
)


reverse_map = {
    0: "UNRELIABLE",
    1: "UNCERTAIN",
    2: "RELIABLE"
}


for label_id, count in zip(
    unique,
    counts
):

    print(
        reverse_map[
            int(label_id)
        ],
        ":",
        int(count)
    )


# ============================================================
# SAFETY CHECK
# ============================================================

minimum_class_count = (
    counts.min()
)


if minimum_class_count < 2:

    raise ValueError(
        "At least two examples per class "
        "are required for evaluation."
    )


# ============================================================
# MODEL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "BUILDING TRUSTGATE MODEL"
)

print(
    "=" * 60
)


base_model = Pipeline(
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

n_splits = min(
    5,
    int(
        minimum_class_count
    )
)


cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)


print(
    "Cross-validation folds:",
    n_splits
)


print(
    "\nGenerating out-of-fold predictions..."
)


predictions = cross_val_predict(
    base_model,
    X,
    y,
    cv=cv,
    method="predict"
)


probabilities = cross_val_predict(
    base_model,
    X,
    y,
    cv=cv,
    method="predict_proba"
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


print(
    "\n" + "=" * 60
)

print(
    "TRUSTGATE CROSS-VALIDATION RESULTS"
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
    "\nConfusion matrix:"
)

print(
    cm
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRAINING FINAL TRUSTGATE MODEL"
)

print(
    "=" * 60
)


final_model = base_model.fit(
    X,
    y
)


# ============================================================
# PREDICTIONS ON FULL DATA
# ============================================================

final_predictions = (
    final_model.predict(
        X
    )
)


final_probabilities = (
    final_model.predict_proba(
        X
    )
)


# ============================================================
# CREATE TRUST SCORE
# ============================================================

# Probability of RELIABLE.
#
# This is deliberately called a "model confidence / trust
# score" rather than objective truth probability.

reliable_probability = (
    final_probabilities[
        :,
        label_map["RELIABLE"]
    ]
)


# ------------------------------------------------------------
# Convert model output to user-facing trust score.
# ------------------------------------------------------------

trust_scores = np.clip(
    reliable_probability,
    0.0,
    1.0
)


# ============================================================
# USER-FACING DECISION
# ============================================================

def trust_category(
    score
):

    if score >= 0.70:

        return "RELIABLE"

    elif score >= 0.40:

        return "UNCERTAIN"

    else:

        return "UNRELIABLE"


categories = [
    trust_category(
        score
    )
    for score in trust_scores
]


# ============================================================
# SAVE SCORES
# ============================================================

score_rows = []


for i in range(
    len(data)
):

    score_rows.append(
        {

            "id":
                ids[i],

            "question":
                questions[i],

            "true_label":
                reverse_map[
                    int(y[i])
                ],

            "predicted_label":
                reverse_map[
                    int(
                        final_predictions[i]
                    )
                ],

            "trust_score":
                float(
                    trust_scores[i]
                ),

            "category":
                categories[i],

            "prob_unreliable":
                float(
                    final_probabilities[
                        i,
                        0
                    ]
                ),

            "prob_uncertain":
                float(
                    final_probabilities[
                        i,
                        1
                    ]
                ),

            "prob_reliable":
                float(
                    final_probabilities[
                        i,
                        2
                    ]
                )
        }
    )


scores_df = pd.DataFrame(
    score_rows
)


scores_df.to_csv(
    SCORES_FILE,
    index=False
)


# ============================================================
# SAVE MODEL PARAMETERS
# ============================================================

scaler = (
    final_model
    .named_steps[
        "scaler"
    ]
)

classifier = (
    final_model
    .named_steps[
        "classifier"
    ]
)


model_data = {

    "model_type":
        "LogisticRegression",

    "feature_dimension":
        901,

    "label_mapping":
        label_map,

    "labels":
        {
            "0": "UNRELIABLE",
            "1": "UNCERTAIN",
            "2": "RELIABLE"
        },

    "decision_thresholds":
        {
            "unreliable":
                0.40,

            "reliable":
                0.70
        },

    "accuracy":
        float(
            accuracy
        ),

    "balanced_accuracy":
        float(
            balanced_accuracy
        ),

    "scaler_mean":
        scaler.mean_.tolist(),

    "scaler_scale":
        scaler.scale_.tolist(),

    "classifier_coef":
        classifier.coef_.tolist(),

    "classifier_intercept":
        classifier.intercept_.tolist()
}


with open(
    MODEL_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_data,
        f,
        indent=4
    )


# ============================================================
# TRUST SCORE SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRUST SCORE SUMMARY"
)

print(
    "=" * 60
)


print(
    "Minimum:",
    round(
        float(
            trust_scores.min()
        ),
        4
    )
)


print(
    "25th percentile:",
    round(
        float(
            np.quantile(
                trust_scores,
                0.25
            )
        ),
        4
    )
)


print(
    "Median:",
    round(
        float(
            np.median(
                trust_scores
            )
        ),
        4
    )
)


print(
    "75th percentile:",
    round(
        float(
            np.quantile(
                trust_scores,
                0.75
            )
        ),
        4
    )
)


print(
    "Maximum:",
    round(
        float(
            trust_scores.max()
        ),
        4
    )
)


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "PREDICTED TRUST CATEGORIES"
)

print(
    "=" * 60
)


category_counts = (
    scores_df[
        "category"
    ].value_counts()
)


for category in [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE"
]:

    print(
        category,
        ":",
        int(
            category_counts.get(
                category,
                0
            )
        )
    )


# ============================================================
# SAVE COMPLETE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRUSTGATE MODEL BUILD COMPLETE!"
)

print(
    "=" * 60
)

print(
    "Model:"
)

print(
    MODEL_FILE
)

print(
    "\nScores:"
)

print(
    SCORES_FILE
)