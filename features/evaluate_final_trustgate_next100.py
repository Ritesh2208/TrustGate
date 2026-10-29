import json
import pandas as pd

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

HUMAN_FILE = (
    "outputs/trustgate_human_evaluation_next_100_completed.csv"
)

PREDICTION_FILE = (
    "outputs/trustgate_final_predictions.json"
)

OUTPUT_CSV = (
    "outputs/trustgate_final_next100_evaluation.csv"
)


LABEL_ORDER = [

    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE"

]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FINAL TRUSTGATE MODEL: NEXT 100 EVALUATION")
print("=" * 70)


# ============================================================
# LOAD HUMAN LABELS
# ============================================================

print("\nLoading human evaluation labels...")

human_df = pd.read_csv(
    HUMAN_FILE
)

print(
    "Human records:",
    len(human_df)
)

print("\nHuman label distribution:")

print(
    human_df[
        "human_label"
    ].value_counts()
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading TrustGate predictions...")

with open(
    PREDICTION_FILE,
    "r",
    encoding="utf-8"
) as f:

    predictions = json.load(f)


pred_df = pd.DataFrame(
    predictions
)


print(
    "Prediction records:",
    len(pred_df)
)

print("\nPrediction distribution:")

print(
    pred_df[
        "predicted_label"
    ].value_counts()
)


# ============================================================
# ALIGN BY ID
# ============================================================

print("\nAligning predictions with human labels...")

evaluation_df = human_df.merge(

    pred_df,

    on="id",

    how="inner",

    suffixes=(
        "_human",
        "_prediction"
    )

)


print(
    "Aligned records:",
    len(evaluation_df)
)


if len(evaluation_df) != 100:

    raise ValueError(
        f"Expected 100 aligned records, got {len(evaluation_df)}"
    )


# ============================================================
# EXTRACT LABELS
# ============================================================

y_true = evaluation_df[
    "human_label"
]

y_pred = evaluation_df[
    "predicted_label"
]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(

    classification_report(

        y_true,

        y_pred,

        labels=LABEL_ORDER,

        zero_division=0

    )

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=LABEL_ORDER

)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\nRows = Human Labels"
)

print(
    "Columns = TrustGate Predictions"
)

print(
    "\nLabels:",
    LABEL_ORDER
)

print(
    cm
)


# ============================================================
# CORRECTNESS
# ============================================================

evaluation_df[
    "correct_prediction"
] = (

    evaluation_df[
        "human_label"
    ]

    ==

    evaluation_df[
        "predicted_label"
    ]

)


# ============================================================
# SAVE EVALUATION CSV
# ============================================================

evaluation_df.to_csv(

    OUTPUT_CSV,

    index=False

)


print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nRecords evaluated:",
    len(evaluation_df)
)

print(
    f"\nFINAL Accuracy: {accuracy:.4f}"
)

print(
    f"FINAL Balanced Accuracy: {balanced_accuracy:.4f}"
)

print(
    f"FINAL Macro F1: {macro_f1:.4f}"
)

print(
    f"FINAL Weighted F1: {weighted_f1:.4f}"
)

print(
    "\nDetailed evaluation saved to:"
)

print(
    OUTPUT_CSV
)