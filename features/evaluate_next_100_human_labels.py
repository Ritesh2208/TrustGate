import json
import os

import pandas as pd
import numpy as np

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
    "outputs/trustgate_next_100_predictions.json"
)

OUTPUT_JSON = (
    "outputs/trustgate_next_100_human_evaluation_results.json"
)

OUTPUT_CSV = (
    "outputs/trustgate_next_100_human_vs_prediction.csv"
)


# ============================================================
# LABEL ORDER
# ============================================================

LABELS = [
    "UNRELIABLE",
    "UNCERTAIN",
    "RELIABLE"
]


# ============================================================
# START
# ============================================================

print("=" * 60)
print("TRUSTGATE NEXT 100 HUMAN EVALUATION")
print("=" * 60)


# ============================================================
# LOAD HUMAN LABELS
# ============================================================

print("\nLoading human evaluation file...")

human_df = pd.read_csv(
    HUMAN_FILE
)

print(
    "Human records:",
    len(human_df)
)

print(
    "\nHuman label distribution:"
)

print(
    human_df[
        "human_label"
    ].value_counts()
)


# ============================================================
# CLEAN HUMAN LABELS
# ============================================================

human_df[
    "human_label"
] = (
    human_df[
        "human_label"
    ]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# CHECK MISSING LABELS
# ============================================================

missing_labels = human_df[
    "human_label"
].isin(
    LABELS
) == False


if missing_labels.any():

    print(
        "\nWARNING: Invalid or missing labels found!"
    )

    print(
        human_df[
            missing_labels
        ][
            ["id", "human_label"]
        ]
    )

    raise ValueError(
        "Please fix human labels before evaluation."
    )


print(
    "\nHuman labels validated successfully."
)


# ============================================================
# LOAD TRUSTGATE PREDICTIONS
# ============================================================

print(
    "\nLoading TrustGate predictions..."
)

with open(
    PREDICTION_FILE,
    "r",
    encoding="utf-8"
) as f:

    predictions = json.load(
        f
    )


print(
    "Prediction records:",
    len(predictions)
)


# ============================================================
# INSPECT PREDICTION STRUCTURE
# ============================================================

if len(predictions) == 0:

    raise ValueError(
        "Prediction file is empty."
    )


print(
    "\nPrediction keys:"
)

print(
    predictions[0].keys()
)


# ============================================================
# CREATE PREDICTION DATAFRAME
# ============================================================

prediction_rows = []


for item in predictions:

    prediction_rows.append(
        {

            "id":
                item.get(
                    "id"
                ),

            "predicted_label":
                item.get(
                    "prediction",
                    item.get(
                        "predicted_label",
                        ""
                    )
                ),

            "classification_confidence":
                item.get(
                    "classification_confidence",
                    None
                ),

            "trust_score":
                item.get(
                    "trust_score",
                    None
                )

        }
    )


prediction_df = pd.DataFrame(
    prediction_rows
)


# ============================================================
# CLEAN PREDICTED LABELS
# ============================================================

prediction_df[
    "predicted_label"
] = (
    prediction_df[
        "predicted_label"
    ]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# VALIDATE PREDICTED LABELS
# ============================================================

invalid_predictions = (
    ~prediction_df[
        "predicted_label"
    ].isin(
        LABELS
    )
)


if invalid_predictions.any():

    print(
        "\nWARNING: Invalid predicted labels found:"
    )

    print(
        prediction_df[
            invalid_predictions
        ]
    )

    raise ValueError(
        "Invalid labels found in prediction file."
    )


# ============================================================
# ALIGN DATASETS
# ============================================================

print(
    "\nAligning human labels and predictions by ID..."
)


human_df[
    "id"
] = human_df[
    "id"
].astype(int)


prediction_df[
    "id"
] = prediction_df[
    "id"
].astype(int)


merged_df = human_df.merge(

    prediction_df,

    on="id",

    how="inner"

)


print(
    "Human records:",
    len(human_df)
)

print(
    "Prediction records:",
    len(prediction_df)
)

print(
    "Aligned records:",
    len(merged_df)
)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

if len(merged_df) == 0:

    raise ValueError(
        "No records aligned by ID."
    )


if len(merged_df) != len(human_df):

    print(
        "\nWARNING:"
    )

    print(
        "Not all human records aligned with predictions."
    )


# ============================================================
# DISPLAY LABEL DISTRIBUTIONS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "LABEL DISTRIBUTION COMPARISON"
)

print(
    "=" * 60
)


print(
    "\nHUMAN LABELS:"
)


for label in LABELS:

    count = sum(
        merged_df[
            "human_label"
        ] == label
    )

    print(
        f"{label}: {count}"
    )


print(
    "\nTRUSTGATE PREDICTIONS:"
)


for label in LABELS:

    count = sum(
        merged_df[
            "predicted_label"
        ] == label
    )

    print(
        f"{label}: {count}"
    )


# ============================================================
# PREPARE LABELS
# ============================================================

y_true = merged_df[
    "human_label"
]

y_pred = merged_df[
    "predicted_label"
]


# ============================================================
# CALCULATE METRICS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "CLASSIFICATION PERFORMANCE"
)

print(
    "=" * 60
)


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

    labels=LABELS,

    average="macro",

    zero_division=0

)


weighted_f1 = f1_score(

    y_true,

    y_pred,

    labels=LABELS,

    average="weighted",

    zero_division=0

)


print(
    "\nAccuracy:",
    round(
        accuracy,
        4
    )
)


print(
    "\nBalanced Accuracy:",
    round(
        balanced_accuracy,
        4
    )
)


print(
    "\nMacro F1:",
    round(
        macro_f1,
        4
    )
)


print(
    "\nWeighted F1:",
    round(
        weighted_f1,
        4
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 60
)


report = classification_report(

    y_true,

    y_pred,

    labels=LABELS,

    zero_division=0

)


print(
    report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "CONFUSION MATRIX"
)

print(
    "=" * 60
)


cm = confusion_matrix(

    y_true,

    y_pred,

    labels=LABELS

)


print(
    "\nRows = Human Labels"
)

print(
    "Columns = TrustGate Predictions\n"
)


print(
    "Labels:",
    LABELS
)


print(
    cm
)


# ============================================================
# TRUST SCORE ANALYSIS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRUST SCORE ANALYSIS"
)

print(
    "=" * 60
)


merged_df[
    "correct_prediction"
] = (

    merged_df[
        "human_label"
    ]

    ==

    merged_df[
        "predicted_label"
    ]

)


correct_scores = merged_df.loc[

    merged_df[
        "correct_prediction"
    ],

    "trust_score"

].dropna()


incorrect_scores = merged_df.loc[

    ~merged_df[
        "correct_prediction"
    ],

    "trust_score"

].dropna()


if len(correct_scores) > 0:

    print(
        "\nAverage trust score for CORRECT predictions:",
        round(
            float(
                correct_scores.mean()
            ),
            4
        )
    )


if len(incorrect_scores) > 0:

    print(
        "Average trust score for INCORRECT predictions:",
        round(
            float(
                incorrect_scores.mean()
            ),
            4
        )
    )


# ============================================================
# CLASSIFICATION CONFIDENCE ANALYSIS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "CLASSIFICATION CONFIDENCE ANALYSIS"
)

print(
    "=" * 60
)


correct_confidence = merged_df.loc[

    merged_df[
        "correct_prediction"
    ],

    "classification_confidence"

].dropna()


incorrect_confidence = merged_df.loc[

    ~merged_df[
        "correct_prediction"
    ],

    "classification_confidence"

].dropna()


if len(correct_confidence) > 0:

    print(
        "\nAverage confidence for CORRECT predictions:",
        round(
            float(
                correct_confidence.mean()
            ),
            4
        )
    )


if len(incorrect_confidence) > 0:

    print(
        "Average confidence for INCORRECT predictions:",
        round(
            float(
                incorrect_confidence.mean()
            ),
            4
        )
    )


# ============================================================
# SAVE DETAILED CSV
# ============================================================

print(
    "\nCreating detailed comparison CSV..."
)


merged_df.to_csv(

    OUTPUT_CSV,

    index=False,

    encoding="utf-8"

)


# ============================================================
# PREPARE JSON RESULTS
# ============================================================

results = {

    "experiment":

        "TrustGate independent evaluation on questions 101-200",


    "records":

        int(
            len(
                merged_df
            )
        ),


    "id_range":

        {

            "minimum":

                int(
                    merged_df[
                        "id"
                    ].min()
                ),

            "maximum":

                int(
                    merged_df[
                        "id"
                    ].max()
                )

        },


    "human_label_distribution":

        {

            label:

                int(
                    sum(
                        merged_df[
                            "human_label"
                        ] == label
                    )
                )

            for label in LABELS

        },


    "prediction_distribution":

        {

            label:

                int(
                    sum(
                        merged_df[
                            "predicted_label"
                        ] == label
                    )
                )

            for label in LABELS

        },


    "metrics":

        {

            "accuracy":

                float(
                    accuracy
                ),

            "balanced_accuracy":

                float(
                    balanced_accuracy
                ),

            "macro_f1":

                float(
                    macro_f1
                ),

            "weighted_f1":

                float(
                    weighted_f1
                )

        },


    "confusion_matrix":

        cm.tolist(),


    "labels":

        LABELS,


    "trust_score_analysis":

        {

            "correct_prediction_mean":

                float(
                    correct_scores.mean()
                )

                if len(correct_scores) > 0

                else None,


            "incorrect_prediction_mean":

                float(
                    incorrect_scores.mean()
                )

                if len(incorrect_scores) > 0

                else None

        },


    "classification_confidence_analysis":

        {

            "correct_prediction_mean":

                float(
                    correct_confidence.mean()
                )

                if len(correct_confidence) > 0

                else None,


            "incorrect_prediction_mean":

                float(
                    incorrect_confidence.mean()
                )

                if len(incorrect_confidence) > 0

                else None

        }

}


# ============================================================
# SAVE JSON
# ============================================================

with open(

    OUTPUT_JSON,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        results,

        f,

        indent=4

    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "NEXT 100 HUMAN EVALUATION COMPLETE!"
)

print(
    "=" * 60
)


print(
    "\nRecords evaluated:",
    len(
        merged_df
    )
)


print(
    "\nFINAL RESULTS"
)


print(
    f"Accuracy: {accuracy:.4f}"
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


print(
    "\nResults saved to:"
)


print(
    OUTPUT_JSON
)


print(
    "\nDetailed comparison saved to:"
)


print(
    OUTPUT_CSV
)