import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_FEATURES = (
    "outputs/truthfulqa_100_features.json"
)

NEXT_FEATURES = (
    "outputs/truthfulqa_next_100_features.json"
)

ORIGINAL_LABELS = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

NEXT_LABELS = (
    "outputs/trustgate_human_evaluation_next_100_completed.csv"
)

OUTPUT_DIR = (
    "outputs/cross_validation_200"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LABEL ENCODING
# ============================================================

LABEL_MAP = {
    "RELIABLE": 0,
    "UNCERTAIN": 1,
    "UNRELIABLE": 2
}

LABEL_NAMES = [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE"
]


# ============================================================
# LOAD HUMAN LABELS
# ============================================================

print("=" * 70)
print("TRUSTGATE CLEAN 200-SAMPLE CROSS-VALIDATION")
print("=" * 70)

print("\nLoading human labels...")

labels_original = pd.read_csv(
    ORIGINAL_LABELS
)

labels_next = pd.read_csv(
    NEXT_LABELS
)

labels = pd.concat(
    [labels_original, labels_next],
    ignore_index=True
)

labels = labels[
    labels["human_label"].isin(
        LABEL_MAP.keys()
    )
].copy()

labels["target"] = labels[
    "human_label"
].map(
    LABEL_MAP
)

print(
    "Total labels:",
    len(labels)
)

print("\nLabel distribution:")

print(
    labels["human_label"].value_counts()
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature files...")

with open(
    ORIGINAL_FEATURES,
    "r",
    encoding="utf-8"
) as f:

    original_features = json.load(f)


with open(
    NEXT_FEATURES,
    "r",
    encoding="utf-8"
) as f:

    next_features = json.load(f)


features = (
    original_features +
    next_features
)

print(
    "Total feature records:",
    len(features)
)


# ============================================================
# BUILD FEATURE TABLE
# ============================================================

feature_rows = []

for item in features:

    feature_rows.append({

        "id":
            item["id"],

        "confidence":
            item["confidence"],

        "entropy":
            item["entropy"],

        "hidden_state":
            item["hidden_state"]

    })


feature_df = pd.DataFrame(
    feature_rows
)

print(
    "Feature records:",
    len(feature_df)
)


# ============================================================
# ALIGN FEATURES AND LABELS
# ============================================================

print("\nAligning features and labels...")

df = labels.merge(
    feature_df,
    on="id",
    how="inner"
)

print(
    "Aligned records:",
    len(df)
)

print(
    "Duplicate IDs:",
    df["id"].duplicated().sum()
)


# ============================================================
# BUILD MATRICES
# ============================================================

y = df["target"].values


# Signals

X_signals = df[
    [
        "confidence",
        "entropy"
    ]
].values


# Hidden states

X_hidden = np.array(
    df["hidden_state"].tolist()
)


# Fusion

X_fusion = np.concatenate(
    [
        X_signals,
        X_hidden
    ],
    axis=1
)


print("\nFeature shapes:")

print(
    "Signals:",
    X_signals.shape
)

print(
    "Hidden:",
    X_hidden.shape
)

print(
    "Fusion:",
    X_fusion.shape
)


# ============================================================
# CROSS-VALIDATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    X,
    model
):

    print("\n" + "=" * 70)

    print(name)

    print("=" * 70)


    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    predictions = np.zeros(
        len(y),
        dtype=int
    )


    probabilities = np.zeros(
        (
            len(y),
            3
        )
    )


    fold_results = []


    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X[
            train_idx
        ]

        X_test = X[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]


        # Fit model

        model.fit(
            X_train,
            y_train
        )


        # Predictions

        fold_predictions = (
            model.predict(
                X_test
            )
        )


        fold_probabilities = (
            model.predict_proba(
                X_test
            )
        )


        predictions[
            test_idx
        ] = fold_predictions


        probabilities[
            test_idx
        ] = fold_probabilities


        # Fold metrics

        accuracy = accuracy_score(
            y_test,
            fold_predictions
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                fold_predictions
            )
        )

        macro_f1 = f1_score(
            y_test,
            fold_predictions,
            average="macro",
            zero_division=0
        )


        print(
            f"\nFold {fold}"
        )

        print(
            "Accuracy:",
            round(
                accuracy,
                4
            )
        )

        print(
            "Balanced Accuracy:",
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


        fold_results.append({

            "fold":
                fold,

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "macro_f1":
                macro_f1

        })


    # ========================================================
    # FINAL OUT-OF-FOLD RESULTS
    # ========================================================

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

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
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


    predicted_labels = [
        LABEL_NAMES[x]
        for x in predictions
    ]


    prediction_counts = (
        pd.Series(
            predicted_labels
        )
        .value_counts()
    )


    dominant_percentage = (
        prediction_counts.max()
        /
        len(predictions)
        *
        100
    )


    collapse_warning = (
        dominant_percentage >= 70
    )


    print("\n" + "-" * 70)

    print(
        "FINAL OUT-OF-FOLD RESULTS"
    )

    print("-" * 70)

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "Balanced Accuracy:",
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

    print(
        "Weighted F1:",
        round(
            weighted_f1,
            4
        )
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        "Rows = Human Labels"
    )

    print(
        "Columns = Predictions"
    )

    print(
        "Labels:",
        LABEL_NAMES
    )

    print(cm)


    print(
        "\nPrediction Distribution:"
    )

    print(
        prediction_counts
    )


    print(
        "\nDominant prediction percentage:",
        round(
            dominant_percentage,
            2
        )
    )

    print(
        "Prediction collapse warning:",
        collapse_warning
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    result_df = df[
        [
            "id",
            "human_label"
        ]
    ].copy()


    result_df[
        "true_class"
    ] = y


    result_df[
        "predicted_class"
    ] = predictions


    result_df[
        "predicted_label"
    ] = predicted_labels


    result_df[
        "prediction_confidence"
    ] = probabilities.max(
        axis=1
    )


    result_df.to_csv(

        os.path.join(
            OUTPUT_DIR,

            name.lower()
            .replace(
                " ",
                "_"
            )
            .replace(
                "+",
                "plus"
            )
            + "_predictions.csv"
        ),

        index=False

    )


    return {

        "experiment":
            name,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "dominant_prediction_percentage":
            dominant_percentage,

        "prediction_collapse_warning":
            collapse_warning

    }


# ============================================================
# MODELS
# ============================================================

results = []


# ------------------------------------------------------------
# Signals Logistic Regression
# ------------------------------------------------------------

signals_lr = Pipeline([

    (
        "scaler",

        StandardScaler()
    ),

    (
        "classifier",

        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )

])


results.append(

    evaluate_model(

        "Signals Logistic Regression",

        X_signals,

        signals_lr

    )

)


# ------------------------------------------------------------
# Signals Random Forest
# ------------------------------------------------------------

signals_rf = RandomForestClassifier(

    n_estimators=300,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


results.append(

    evaluate_model(

        "Signals Random Forest",

        X_signals,

        signals_rf

    )

)


# ------------------------------------------------------------
# Hidden PCA20 Logistic Regression
# ------------------------------------------------------------

hidden_pca_lr = Pipeline([

    (
        "scaler",

        StandardScaler()
    ),

    (
        "pca",

        PCA(
            n_components=20,
            random_state=42
        )
    ),

    (
        "classifier",

        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )

])


results.append(

    evaluate_model(

        "Hidden PCA20 Logistic Regression",

        X_hidden,

        hidden_pca_lr

    )

)


# ------------------------------------------------------------
# Fusion PCA20 Logistic Regression
# ------------------------------------------------------------

fusion_pca_lr = Pipeline([

    (
        "scaler",

        StandardScaler()
    ),

    (
        "pca",

        PCA(
            n_components=20,
            random_state=42
        )
    ),

    (
        "classifier",

        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )

])


results.append(

    evaluate_model(

        "Fusion PCA20 Logistic Regression",

        X_fusion,

        fusion_pca_lr

    )

)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)

print(
    "FINAL CROSS-VALIDATION SUMMARY"
)

print("=" * 70)


print(

    results_df.sort_values(

        "macro_f1",

        ascending=False

    ).to_string(

        index=False

    )

)


results_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "cross_validation_200_results.csv"

    ),

    index=False

)


print(

    "\nResults saved to:"

)

print(

    os.path.join(

        OUTPUT_DIR,

        "cross_validation_200_results.csv"

    )

)


print(

    "\nEXPERIMENT COMPLETE."

)