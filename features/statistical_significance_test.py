import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from scipy.stats import binomtest


# ============================================================
# SETTINGS
# ============================================================

ANNOTATION_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

FEATURE_FILE = (
    "outputs/trustgate_100_master_901.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_statistical_significance.json"
)

PREDICTIONS_FILE = (
    "outputs/trustgate_statistical_predictions.csv"
)


# ============================================================
# LOAD HUMAN ANNOTATIONS
# ============================================================

print("=" * 60)
print("TRUSTGATE STATISTICAL SIGNIFICANCE TEST")
print("=" * 60)

print("\nLoading independent human annotations...")

annotations = pd.read_csv(
    ANNOTATION_FILE
)

print(
    "Annotation records:",
    len(annotations)
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading TrustGate features...")

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
# ALIGN BY ID
# ============================================================

feature_by_id = {
    int(item["id"]): item
    for item in feature_data
}


annotations = annotations.sort_values(
    "id"
).reset_index(
    drop=True
)


aligned = []

for _, row in annotations.iterrows():

    sample_id = int(
        row["id"]
    )

    if sample_id not in feature_by_id:

        raise ValueError(
            f"Missing feature ID: {sample_id}"
        )

    item = feature_by_id[
        sample_id
    ]

    aligned.append(
        {
            "id": sample_id,
            "question": row["question"],
            "human_label": row["human_label"],
            "feature_vector": item[
                "feature_vector"
            ]
        }
    )


print(
    "\nAligned records:",
    len(aligned)
)


# ============================================================
# LABEL MAP
# ============================================================

label_map = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2,
}


# ============================================================
# BUILD REPRESENTATIONS
# ============================================================

X_hidden = []

X_fusion = []

y = []

ids = []

questions = []


for item in aligned:

    vector = np.asarray(
        item["feature_vector"],
        dtype=np.float32
    )

    if len(vector) != 901:

        raise ValueError(
            f"Expected 901 features, got {len(vector)}"
        )


    hidden = vector[:896]

    X_hidden.append(
        hidden
    )

    X_fusion.append(
        vector
    )

    y.append(
        label_map[
            item["human_label"]
        ]
    )

    ids.append(
        item["id"]
    )

    questions.append(
        item["question"]
    )


X_hidden = np.asarray(
    X_hidden,
    dtype=np.float32
)

X_fusion = np.asarray(
    X_fusion,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int64
)


print("\nRepresentation check:")

print(
    "Hidden-only:",
    X_hidden.shape
)

print(
    "TrustGate fusion:",
    X_fusion.shape
)


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# MODEL
#
# IMPORTANT:
# This matches the nonlinear evaluation experiment.
# ============================================================

def build_model():

    return Pipeline(
        steps=[

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )


# ============================================================
# GENERATE OOF PREDICTIONS
# ============================================================

def get_oof_predictions(
    X,
    name
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


    predictions = np.zeros(
        len(y),
        dtype=np.int64
    )


    for fold, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print(
            f"Fold {fold}/5"
        )


        model = build_model()


        model.fit(
            X[train_index],
            y[train_index]
        )


        fold_predictions = (
            model.predict(
                X[test_index]
            )
        )


        predictions[
            test_index
        ] = fold_predictions


    return predictions


hidden_predictions = (
    get_oof_predictions(
        X_hidden,
        "Hidden-only (896-D)"
    )
)


fusion_predictions = (
    get_oof_predictions(
        X_fusion,
        "TrustGate Fusion (901-D)"
    )
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions
):

    return {

        "accuracy":
            float(
                accuracy_score(
                    y,
                    predictions
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    predictions
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            )
    }


hidden_metrics = (
    calculate_metrics(
        hidden_predictions
    )
)


fusion_metrics = (
    calculate_metrics(
        fusion_predictions
    )
)


print(
    "\nHidden-only metrics:"
)

print(
    hidden_metrics
)


print(
    "\nFusion metrics:"
)

print(
    fusion_metrics
)


# ============================================================
# MCNEMAR CONTINGENCY TABLE
# ============================================================

hidden_correct = (
    hidden_predictions == y
)

fusion_correct = (
    fusion_predictions == y
)


# Both correct
both_correct = int(
    np.sum(
        hidden_correct &
        fusion_correct
    )
)


# Hidden correct, fusion wrong
hidden_only_correct = int(
    np.sum(
        hidden_correct &
        ~fusion_correct
    )
)


# Fusion correct, hidden wrong
fusion_only_correct = int(
    np.sum(
        ~hidden_correct &
        fusion_correct
    )
)


# Both wrong
both_wrong = int(
    np.sum(
        ~hidden_correct &
        ~fusion_correct
    )
)


print(
    "\n" + "=" * 60
)

print(
    "MCNEMAR CONTINGENCY TABLE"
)

print(
    "=" * 60
)


print(
    "\n                    Fusion Correct    Fusion Wrong"
)

print(
    "Hidden Correct     ",
    both_correct,
    "              ",
    hidden_only_correct
)

print(
    "Hidden Wrong       ",
    fusion_only_correct,
    "              ",
    both_wrong
)


# ============================================================
# EXACT MCNEMAR TEST
#
# Only discordant predictions are tested.
# ============================================================

discordant = (
    hidden_only_correct +
    fusion_only_correct
)


if discordant == 0:

    p_value = 1.0

else:

    result = binomtest(

        k=min(
            hidden_only_correct,
            fusion_only_correct
        ),

        n=discordant,

        p=0.5,

        alternative="two-sided"
    )

    p_value = float(
        result.pvalue
    )


print(
    "\nDiscordant samples:",
    discordant
)


print(
    "Exact McNemar p-value:",
    round(
        p_value,
        6
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

alpha = 0.05


if p_value < alpha:

    interpretation = (
        "STATISTICALLY_SIGNIFICANT"
    )

else:

    interpretation = (
        "NOT_STATISTICALLY_SIGNIFICANT"
    )


print(
    "\nResult:",
    interpretation
)


# ============================================================
# SAVE PER-SAMPLE PREDICTIONS
# ============================================================

reverse_map = {

    0: "UNRELIABLE",
    1: "UNCERTAIN",
    2: "RELIABLE"
}


prediction_rows = []


for i in range(
    len(y)
):

    prediction_rows.append(

        {

            "id":
                ids[i],

            "question":
                questions[i],

            "true_label":
                reverse_map[
                    int(y[i])
                ],

            "hidden_prediction":
                reverse_map[
                    int(
                        hidden_predictions[i]
                    )
                ],

            "fusion_prediction":
                reverse_map[
                    int(
                        fusion_predictions[i]
                    )
                ],

            "hidden_correct":
                bool(
                    hidden_correct[i]
                ),

            "fusion_correct":
                bool(
                    fusion_correct[i]
                )
        }
    )


predictions_df = pd.DataFrame(
    prediction_rows
)


predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "TrustGate statistical significance test",

    "records":
        int(
            len(y)
        ),

    "cross_validation":
        {

            "method":
                "StratifiedKFold",

            "splits":
                5,

            "shuffle":
                True,

            "random_state":
                42
        },

    "hidden_only_metrics":
        hidden_metrics,

    "trustgate_fusion_metrics":
        fusion_metrics,

    "performance_difference":
        {

            "accuracy":
                float(
                    fusion_metrics[
                        "accuracy"
                    ]
                    -
                    hidden_metrics[
                        "accuracy"
                    ]
                ),

            "balanced_accuracy":
                float(
                    fusion_metrics[
                        "balanced_accuracy"
                    ]
                    -
                    hidden_metrics[
                        "balanced_accuracy"
                    ]
                ),

            "macro_f1":
                float(
                    fusion_metrics[
                        "macro_f1"
                    ]
                    -
                    hidden_metrics[
                        "macro_f1"
                    ]
                )
        },

    "mcnemar_contingency_table":
        {

            "both_correct":
                both_correct,

            "hidden_only_correct":
                hidden_only_correct,

            "fusion_only_correct":
                fusion_only_correct,

            "both_wrong":
                both_wrong
        },

    "discordant_samples":
        discordant,

    "mcnemar_p_value":
        p_value,

    "alpha":
        alpha,

    "interpretation":
        interpretation
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
    "STATISTICAL SIGNIFICANCE TEST COMPLETE"
)

print(
    "=" * 60
)


print(
    "\nSaved results:"
)

print(
    OUTPUT_FILE
)


print(
    "\nSaved predictions:"
)

print(
    PREDICTIONS_FILE
)