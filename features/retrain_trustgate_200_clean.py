import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs/retrain_200"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FILES
# ============================================================

HUMAN_ORIGINAL = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

HUMAN_NEXT = (
    "outputs/trustgate_human_evaluation_next_100_completed.csv"
)

FEATURES_ORIGINAL = (
    "outputs/truthfulqa_100_features.json"
)

FEATURES_NEXT = (
    "outputs/truthfulqa_next_100_features.json"
)


# ============================================================
# LOAD HUMAN LABELS
# ============================================================

print("=" * 70)
print("TRUSTGATE CLEAN 200-SAMPLE RETRAINING EXPERIMENT")
print("=" * 70)

print("\nLoading human labels...")

human_original = pd.read_csv(HUMAN_ORIGINAL)
human_next = pd.read_csv(HUMAN_NEXT)

human_original = human_original[
    [
        "id",
        "human_label"
    ]
]

human_next = human_next[
    [
        "id",
        "human_label"
    ]
]

human_df = pd.concat(
    [
        human_original,
        human_next
    ],
    ignore_index=True
)

human_df["id"] = human_df["id"].astype(int)

print(
    "Total human labels:",
    len(human_df)
)

print("\nHuman label distribution:")

print(
    human_df[
        "human_label"
    ].value_counts()
)


# ============================================================
# LOAD FEATURE FILES
# ============================================================

print("\nLoading feature files...")

with open(
    FEATURES_ORIGINAL,
    "r",
    encoding="utf-8"
) as f:

    features_original = json.load(f)


with open(
    FEATURES_NEXT,
    "r",
    encoding="utf-8"
) as f:

    features_next = json.load(f)


features_all = (
    features_original +
    features_next
)

print(
    "Total feature records:",
    len(features_all)
)


# ============================================================
# BASIC FEATURE TABLE
# ============================================================

print("\nBuilding feature table...")

rows = []

for item in features_all:

    row = {

        "id":
            int(item["id"]),

        "confidence":
            item.get(
                "confidence",
                np.nan
            ),

        "entropy":
            item.get(
                "entropy",
                np.nan
            ),

        "hidden_state":
            item.get(
                "hidden_state",
                None
            )

    }

    rows.append(
        row
    )


feature_df = pd.DataFrame(
    rows
)

print(
    "Feature records:",
    len(feature_df)
)


# ============================================================
# ALIGN LABELS + FEATURES
# ============================================================

print("\nAligning labels and features...")

df = human_df.merge(
    feature_df,
    on="id",
    how="inner"
)

df = df.sort_values(
    "id"
).reset_index(
    drop=True
)

print(
    "Aligned records:",
    len(df)
)

assert len(df) == 200, (
    "ERROR: Expected exactly 200 aligned records."
)


# ============================================================
# CHECK FOR DUPLICATE IDS
# ============================================================

duplicates = df[
    "id"
].duplicated().sum()

print(
    "Duplicate IDs:",
    duplicates
)

assert duplicates == 0


# ============================================================
# PREPARE LABELS
# ============================================================

LABEL_ORDER = [

    "UNRELIABLE",
    "UNCERTAIN",
    "RELIABLE"

]

label_encoder = LabelEncoder()

label_encoder.fit(
    LABEL_ORDER
)

y = label_encoder.transform(
    df[
        "human_label"
    ]
)


print("\nEncoded classes:")

for label, code in zip(
    label_encoder.classes_,
    range(
        len(
            label_encoder.classes_
        )
    )
):

    print(
        f"{code} = {label}"
    )


# ============================================================
# SIGNAL FEATURES
# ============================================================

signal_columns = [

    "confidence",
    "entropy"

]

X_signals = df[
    signal_columns
].values


print(
    "\nSignal feature shape:",
    X_signals.shape
)


# ============================================================
# HIDDEN STATE FEATURES
# ============================================================

print(
    "\nBuilding hidden-state matrix..."
)

hidden_vectors = df[
    "hidden_state"
].tolist()


X_hidden = np.array(
    hidden_vectors,
    dtype=float
)


print(
    "Hidden-state shape:",
    X_hidden.shape
)


assert X_hidden.shape[0] == 200
assert X_hidden.shape[1] == 896


# ============================================================
# FUSION FEATURES
# ============================================================

X_fusion = np.concatenate(

    [

        X_signals,

        X_hidden

    ],

    axis=1

)


print(
    "Fusion feature shape:",
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
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(

    experiment_name,

    pipeline,

    X

):

    print("\n" + "=" * 70)

    print(
        experiment_name
    )

    print("=" * 70)


    predictions = cross_val_predict(

        pipeline,

        X,

        y,

        cv=cv,

        method="predict"

    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

        average="macro"

    )


    weighted_f1 = f1_score(

        y,

        predictions,

        average="weighted"

    )


    print(
        "\nAccuracy:",
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


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )


    print(

        classification_report(

            y,

            predictions,

            target_names=
                label_encoder.classes_,

            digits=4,

            zero_division=0

        )

    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(

        y,

        predictions

    )


    print(
        "Confusion Matrix:"
    )


    print(
        "Rows = Human Labels"
    )


    print(
        "Columns = Predictions"
    )


    print(
        cm
    )


    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    prediction_labels = (
        label_encoder.inverse_transform(
            predictions
        )
    )


    prediction_counts = (
        pd.Series(
            prediction_labels
        )
        .value_counts()
        .reindex(
            LABEL_ORDER,
            fill_value=0
        )
    )


    print(
        "\nPrediction Distribution:"
    )


    print(
        prediction_counts
    )


    # --------------------------------------------------------
    # COLLAPSE CHECK
    # --------------------------------------------------------

    dominant_count = (
        prediction_counts.max()
    )


    dominant_percentage = (

        dominant_count

        /

        len(predictions)

        *

        100

    )


    collapse_warning = (

        dominant_percentage >= 70

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


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(

        {

            "id":
                df["id"],

            "human_label":
                df["human_label"],

            "predicted_label":
                prediction_labels

        }

    )


    prediction_path = os.path.join(

        OUTPUT_DIR,

        experiment_name
        .lower()
        .replace(
            " ",
            "_"
        )
        .replace(
            "+",
            "plus"
        )
        .replace(
            "-",
            "_"
        )

        +

        "_predictions.csv"

    )


    prediction_df.to_csv(

        prediction_path,

        index=False

    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "experiment":
            experiment_name,

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
# RESULTS
# ============================================================

results = []


# ============================================================
# EXPERIMENT 1
# SIGNALS ONLY + LOGISTIC REGRESSION
# ============================================================

signals_logistic = Pipeline(

    [

        (

            "imputer",

            SimpleImputer(
                strategy="median"
            )

        ),

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


results.append(

    evaluate_model(

        "Signals Only Logistic Regression",

        signals_logistic,

        X_signals

    )

)


# ============================================================
# EXPERIMENT 2
# SIGNALS ONLY + RANDOM FOREST
# ============================================================

signals_rf = Pipeline(

    [

        (

            "imputer",

            SimpleImputer(
                strategy="median"
            )

        ),

        (

            "classifier",

            RandomForestClassifier(

                n_estimators=500,

                max_depth=5,

                min_samples_leaf=3,

                class_weight="balanced",

                random_state=42,

                n_jobs=-1

            )

        )

    ]

)


results.append(

    evaluate_model(

        "Signals Only Random Forest",

        signals_rf,

        X_signals

    )

)


# ============================================================
# EXPERIMENT 3
# HIDDEN PCA + LOGISTIC REGRESSION
# ============================================================

hidden_pca_logistic = Pipeline(

    [

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

                max_iter=5000,

                class_weight="balanced",

                random_state=42

            )

        )

    ]

)


results.append(

    evaluate_model(

        "Hidden PCA20 Logistic Regression",

        hidden_pca_logistic,

        X_hidden

    )

)


# ============================================================
# EXPERIMENT 4
# FUSION PCA + LOGISTIC REGRESSION
# ============================================================

fusion_pca_logistic = Pipeline(

    [

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

                max_iter=5000,

                class_weight="balanced",

                random_state=42

            )

        )

    ]

)


results.append(

    evaluate_model(

        "Fusion PCA20 Logistic Regression",

        fusion_pca_logistic,

        X_fusion

    )

)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_path = os.path.join(

    OUTPUT_DIR,

    "retrain_200_results.csv"

)

results_df.to_csv(

    results_path,

    index=False

)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL RESULTS SUMMARY"
)

print("=" * 70)


print(

    results_df.sort_values(

        "macro_f1",

        ascending=False

    )

    .to_string(

        index=False

    )

)


print("\nResults saved to:")

print(
    results_path
)


print("\nExperiment complete.")
