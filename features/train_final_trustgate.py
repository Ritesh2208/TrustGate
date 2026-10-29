import json
import os
import joblib

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_LABELS_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

NEXT_LABELS_FILE = (
    "outputs/trustgate_human_evaluation_next_100_completed.csv"
)

ORIGINAL_FEATURES_FILE = (
    "outputs/truthfulqa_100_features.json"
)

NEXT_FEATURES_FILE = (
    "outputs/truthfulqa_next_100_features.json"
)

MODEL_DIR = "models"

MODEL_FILE = (
    "models/trustgate_final_fusion_pca20_logreg.joblib"
)

LABEL_ENCODER_FILE = (
    "models/trustgate_label_encoder.joblib"
)

SUMMARY_FILE = (
    "outputs/trustgate_final_model_summary.json"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUSTGATE FINAL MODEL TRAINING")
print("=" * 70)


# ============================================================
# LOAD HUMAN LABELS
# ============================================================

print("\nLoading human labels...")

original_labels = pd.read_csv(
    ORIGINAL_LABELS_FILE
)

next_labels = pd.read_csv(
    NEXT_LABELS_FILE
)

labels_df = pd.concat(
    [
        original_labels,
        next_labels
    ],
    ignore_index=True
)


print(
    "Total human labels:",
    len(labels_df)
)

print("\nLabel distribution:")

print(
    labels_df[
        "human_label"
    ].value_counts()
)


# ============================================================
# LOAD FEATURE FILES
# ============================================================

print("\nLoading feature files...")

with open(
    ORIGINAL_FEATURES_FILE,
    "r",
    encoding="utf-8"
) as f:

    original_features = json.load(f)


with open(
    NEXT_FEATURES_FILE,
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
# CONVERT FEATURES TO DATAFRAME
# ============================================================

print("\nBuilding feature table...")

feature_rows = []


for record in features:

    feature_rows.append(
        {

            "id":
                record["id"],

            "confidence":
                record["confidence"],

            "entropy":
                record["entropy"],

            "hidden_state":
                record["hidden_state"]

        }
    )


features_df = pd.DataFrame(
    feature_rows
)


print(
    "Feature records:",
    len(features_df)
)


# ============================================================
# ALIGN LABELS AND FEATURES
# ============================================================

print("\nAligning labels and features...")


data = labels_df.merge(
    features_df,
    on="id",
    how="inner"
)


print(
    "Aligned records:",
    len(data)
)


duplicates = data[
    "id"
].duplicated().sum()


print(
    "Duplicate IDs:",
    duplicates
)


if len(data) != 200:

    raise ValueError(
        f"Expected 200 aligned records, got {len(data)}"
    )


if duplicates != 0:

    raise ValueError(
        "Duplicate IDs detected."
    )


# ============================================================
# LABEL ENCODING
# ============================================================

print("\nEncoding labels...")


LABEL_ORDER = [

    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE"

]


label_encoder = LabelEncoder()

label_encoder.fit(
    LABEL_ORDER
)


y = label_encoder.transform(
    data["human_label"]
)


print("\nEncoded classes:")

for index, label in enumerate(
    label_encoder.classes_
):

    print(
        f"{index} = {label}"
    )


# ============================================================
# BUILD HIDDEN MATRIX
# ============================================================

print("\nBuilding hidden-state matrix...")


X_hidden = np.array(

    data[
        "hidden_state"
    ].tolist(),

    dtype=np.float32

)


print(
    "Hidden-state shape:",
    X_hidden.shape
)


if X_hidden.shape[1] != 896:

    raise ValueError(
        "Expected hidden dimension of 896."
    )


# ============================================================
# BUILD SIGNAL MATRIX
# ============================================================

X_signals = np.column_stack(

    [

        data[
            "confidence"
        ].values,

        data[
            "entropy"
        ].values

    ]

)


print(
    "Signal feature shape:",
    X_signals.shape
)


# ============================================================
# BUILD FUSION MATRIX
# ============================================================

X_fusion = np.hstack(

    [

        X_hidden,

        X_signals

    ]

)


print(
    "Fusion feature shape:",
    X_fusion.shape
)


# ============================================================
# FINAL MODEL PIPELINE
# ============================================================

print("\nBuilding final pipeline...")

pipeline = Pipeline(

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


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\nTraining on all 200 human-labeled samples...")

pipeline.fit(

    X_fusion,

    y

)


print(
    "Training complete."
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving final model...")

joblib.dump(

    pipeline,

    MODEL_FILE

)


joblib.dump(

    label_encoder,

    LABEL_ENCODER_FILE

)


print(
    "Model saved to:",
    MODEL_FILE
)

print(
    "Label encoder saved to:",
    LABEL_ENCODER_FILE
)


# ============================================================
# CREATE SUMMARY
# ============================================================

summary = {

    "model_name":
        "TrustGate Final Fusion PCA20 Logistic Regression",

    "training_samples":
        int(
            len(data)
        ),

    "hidden_dimension":
        int(
            X_hidden.shape[1]
        ),

    "signal_features":
        [

            "confidence",

            "entropy"

        ],

    "fusion_dimension":
        int(
            X_fusion.shape[1]
        ),

    "pca_components":
        20,

    "classes":
        label_encoder.classes_.tolist(),

    "label_distribution":

        data[
            "human_label"
        ]

        .value_counts()

        .to_dict()

}


with open(

    SUMMARY_FILE,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        summary,

        f,

        indent=4

    )


print(
    "\nSummary saved to:",
    SUMMARY_FILE
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL TRUSTGATE MODEL TRAINING COMPLETE!"
)

print("=" * 70)

print(
    "\nTraining samples:",
    len(data)
)

print(
    "Fusion dimension:",
    X_fusion.shape[1]
)

print(
    "PCA components:",
    20
)

print(
    "\nFinal model:"
)

print(
    MODEL_FILE
)

print(
    "\nReady for inference."
)