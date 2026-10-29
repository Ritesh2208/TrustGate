import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_LABEL_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

ORIGINAL_PREDICTION_FILE = (
    "outputs/trustgate_statistical_predictions.csv"
)

NEXT_LABEL_FILE = (
    "outputs/trustgate_ai_assisted_evaluation_next_100.csv"
)

NEXT_PREDICTION_FILE = (
    "outputs/trustgate_next_100_predictions.csv"
)


OUTPUT_JSON = (
    "outputs/trustgate_distribution_shift_analysis.json"
)

OUTPUT_CSV = (
    "outputs/trustgate_distribution_shift_summary.csv"
)


LABELS = [
    "UNRELIABLE",
    "UNCERTAIN",
    "RELIABLE"
]


# ============================================================
# HELPER: NORMALIZE LABEL
# ============================================================

def normalize_label(value):

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in LABELS:
        return value

    return None


# ============================================================
# FIND HUMAN LABEL COLUMN
# ============================================================

def find_label_column(df):

    candidates = [
        "human_label",
        "true_label",
        "label",
        "ground_truth"
    ]

    for column in candidates:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not find human label column. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# FIND PREDICTION COLUMN
# ============================================================

def find_prediction_column(df):

    # --------------------------------------------------------
    # IMPORTANT:
    # Your original statistical prediction file contains:
    #
    # hidden_prediction
    # fusion_prediction
    #
    # We want fusion_prediction because that represents
    # TrustGate Fusion.
    # --------------------------------------------------------

    priority_candidates = [

        "predicted_label",

        "fusion_prediction",

        "prediction",

        "trustgate_prediction",

        "predicted_class",

        "label_prediction"
    ]


    for column in priority_candidates:

        if column in df.columns:

            print(
                "Using prediction column:",
                column
            )

            return column


    raise ValueError(

        "Could not find prediction column. "
        f"Available columns: {list(df.columns)}"

    )


# ============================================================
# LOAD CSV SAFELY
# ============================================================

def load_csv(path, description):

    print("\n" + "=" * 60)

    print(
        f"Loading {description}..."
    )

    print("=" * 60)


    if not os.path.exists(path):

        raise FileNotFoundError(

            f"\nFile not found:\n{path}\n"

            "Please check the filename and location."

        )


    df = pd.read_csv(path)


    print(
        "Records:",
        len(df)
    )


    print(
        "Columns:",
        list(df.columns)
    )


    return df


# ============================================================
# ALIGN DATASETS BY ID
# ============================================================

def align_data(

    labels_df,
    predictions_df,
    dataset_name

):

    print("\nAligning records by ID...")


    if "id" not in labels_df.columns:

        raise ValueError(

            f"{dataset_name}: "
            "Label file does not contain an 'id' column."

        )


    if "id" not in predictions_df.columns:

        raise ValueError(

            f"{dataset_name}: "
            "Prediction file does not contain an 'id' column."

        )


    label_column = find_label_column(
        labels_df
    )


    prediction_column = find_prediction_column(
        predictions_df
    )


    labels = labels_df[
        [
            "id",
            label_column
        ]
    ].copy()


    predictions = predictions_df[
        [
            "id",
            prediction_column
        ]
    ].copy()


    labels.columns = [

        "id",

        "human_label"

    ]


    predictions.columns = [

        "id",

        "predicted_label"

    ]


    labels["human_label"] = (

        labels["human_label"]

        .apply(
            normalize_label
        )

    )


    predictions["predicted_label"] = (

        predictions["predicted_label"]

        .apply(
            normalize_label
        )

    )


    # --------------------------------------------------------
    # Remove invalid labels
    # --------------------------------------------------------

    labels = labels.dropna(
        subset=[
            "human_label"
        ]
    )


    predictions = predictions.dropna(
        subset=[
            "predicted_label"
        ]
    )


    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = pd.merge(

        labels,

        predictions,

        on="id",

        how="inner"

    )


    print(
        "Label records:",
        len(labels)
    )


    print(
        "Prediction records:",
        len(predictions)
    )


    print(
        "Aligned records:",
        len(merged)
    )


    if len(merged) == 0:

        raise ValueError(

            f"{dataset_name}: "

            "No matching records after alignment."

        )


    return merged


# ============================================================
# CALCULATE LABEL DISTRIBUTION
# ============================================================

def get_distribution(series):

    counts = (

        series

        .value_counts()

        .reindex(
            LABELS,
            fill_value=0
        )

    )


    total = len(series)


    percentages = {}

    for label in LABELS:

        if total > 0:

            percentage = (

                counts[label]

                / total

            ) * 100

        else:

            percentage = 0


        percentages[label] = float(
            percentage
        )


    return {

        "counts":

            {

                label:

                int(
                    counts[label]
                )

                for label in LABELS

            },


        "percentages":

            percentages

    }


# ============================================================
# CALCULATE PERFORMANCE
# ============================================================

def calculate_metrics(

    y_true,

    y_pred

):

    accuracy = accuracy_score(

        y_true,

        y_pred

    )


    balanced_accuracy = (

        balanced_accuracy_score(

            y_true,

            y_pred

        )

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


    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=LABELS

    )


    return {

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
            ),

        "confusion_matrix":

            matrix.tolist()

    }


# ============================================================
# EVALUATE DATASET
# ============================================================

def evaluate_dataset(

    label_file,

    prediction_file,

    dataset_name

):

    print("\n")

    print("=" * 70)

    print(
        f"EVALUATING: {dataset_name}"
    )

    print("=" * 70)


    labels_df = load_csv(

        label_file,

        f"{dataset_name} labels"

    )


    predictions_df = load_csv(

        prediction_file,

        f"{dataset_name} predictions"

    )


    merged = align_data(

        labels_df,

        predictions_df,

        dataset_name

    )


    y_true = (

        merged[
            "human_label"
        ]

        .tolist()

    )


    y_pred = (

        merged[
            "predicted_label"
        ]

        .tolist()

    )


    # --------------------------------------------------------
    # DISTRIBUTIONS
    # --------------------------------------------------------

    human_distribution = (

        get_distribution(

            merged[
                "human_label"
            ]

        )

    )


    prediction_distribution = (

        get_distribution(

            merged[
                "predicted_label"
            ]

        )

    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(

        y_true,

        y_pred

    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")

    print("=" * 60)

    print(
        "HUMAN LABEL DISTRIBUTION"
    )

    print("=" * 60)


    for label in LABELS:

        count = (

            human_distribution[
                "counts"
            ][
                label
            ]

        )


        percentage = (

            human_distribution[
                "percentages"
            ][
                label
            ]

        )


        print(

            f"{label}: "

            f"{count} "

            f"({percentage:.2f}%)"

        )


    print("\n")

    print("=" * 60)

    print(
        "TRUSTGATE PREDICTION DISTRIBUTION"
    )

    print("=" * 60)


    for label in LABELS:

        count = (

            prediction_distribution[
                "counts"
            ][
                label
            ]

        )


        percentage = (

            prediction_distribution[
                "percentages"
            ][
                label
            ]

        )


        print(

            f"{label}: "

            f"{count} "

            f"({percentage:.2f}%)"

        )


    print("\n")

    print("=" * 60)

    print(
        "PERFORMANCE"
    )

    print("=" * 60)


    print(

        "Accuracy:",

        round(
            metrics[
                "accuracy"
            ],
            4
        )

    )


    print(

        "Balanced Accuracy:",

        round(
            metrics[
                "balanced_accuracy"
            ],
            4
        )

    )


    print(

        "Macro F1:",

        round(
            metrics[
                "macro_f1"
            ],
            4
        )

    )


    print(

        "Weighted F1:",

        round(
            metrics[
                "weighted_f1"
            ],
            4
        )

    )


    print("\n")

    print(
        "Confusion Matrix"
    )


    print(

        "Rows = Human Labels"

    )


    print(

        "Columns = TrustGate Predictions"

    )


    print(

        metrics[
            "confusion_matrix"
        ]

    )


    return {

        "dataset_name":

            dataset_name,


        "records":

            int(
                len(
                    merged
                )
            ),


        "human_label_distribution":

            human_distribution,


        "prediction_distribution":

            prediction_distribution,


        "performance":

            metrics

    }


# ============================================================
# CALCULATE DISTRIBUTION SHIFT
# ============================================================

def calculate_distribution_shift(

    original,

    next_dataset

):

    print("\n")

    print("=" * 70)

    print(
        "DISTRIBUTION SHIFT ANALYSIS"
    )

    print("=" * 70)


    result = {

        "human_label_shift": {},

        "prediction_shift": {},

        "performance_difference": {}

    }


    # --------------------------------------------------------
    # HUMAN LABEL SHIFT
    # --------------------------------------------------------

    print("\n")

    print(
        "HUMAN LABEL DISTRIBUTION SHIFT"
    )


    for label in LABELS:

        original_percentage = (

            original[
                "human_label_distribution"
            ][
                "percentages"
            ][
                label
            ]

        )


        next_percentage = (

            next_dataset[
                "human_label_distribution"
            ][
                "percentages"
            ][
                label
            ]

        )


        difference = (

            next_percentage

            -

            original_percentage

        )


        result[

            "human_label_shift"

        ][

            label

        ] = {

            "original_percentage":

                float(
                    original_percentage
                ),


            "next_percentage":

                float(
                    next_percentage
                ),


            "difference":

                float(
                    difference
                )

        }


        print(

            f"{label}: "

            f"{original_percentage:.2f}% "

            "-> "

            f"{next_percentage:.2f}% "

            f"(Difference: {difference:+.2f}%)"

        )


    # --------------------------------------------------------
    # PREDICTION SHIFT
    # --------------------------------------------------------

    print("\n")

    print(
        "TRUSTGATE PREDICTION DISTRIBUTION SHIFT"
    )


    for label in LABELS:

        original_percentage = (

            original[
                "prediction_distribution"
            ][
                "percentages"
            ][
                label
            ]

        )


        next_percentage = (

            next_dataset[
                "prediction_distribution"
            ][
                "percentages"
            ][
                label
            ]

        )


        difference = (

            next_percentage

            -

            original_percentage

        )


        result[

            "prediction_shift"

        ][

            label

        ] = {

            "original_percentage":

                float(
                    original_percentage
                ),


            "next_percentage":

                float(
                    next_percentage
                ),


            "difference":

                float(
                    difference
                )

        }


        print(

            f"{label}: "

            f"{original_percentage:.2f}% "

            "-> "

            f"{next_percentage:.2f}% "

            f"(Difference: {difference:+.2f}%)"

        )


    # --------------------------------------------------------
    # PERFORMANCE DIFFERENCE
    # --------------------------------------------------------

    print("\n")

    print(
        "PERFORMANCE DIFFERENCE"
    )


    metric_names = [

        "accuracy",

        "balanced_accuracy",

        "macro_f1",

        "weighted_f1"

    ]


    for metric in metric_names:

        original_value = (

            original[
                "performance"
            ][
                metric
            ]

        )


        next_value = (

            next_dataset[
                "performance"
            ][
                metric
            ]

        )


        difference = (

            next_value

            -

            original_value

        )


        result[

            "performance_difference"

        ][

            metric

        ] = {

            "original":

                float(
                    original_value
                ),


            "next_100":

                float(
                    next_value
                ),


            "difference":

                float(
                    difference
                )

        }


        print(

            f"{metric}: "

            f"{original_value:.4f} "

            "-> "

            f"{next_value:.4f} "

            f"(Difference: {difference:+.4f})"

        )


    return result


# ============================================================
# CREATE SUMMARY CSV
# ============================================================

def create_summary_csv(

    original,

    next_dataset,

    shift

):

    rows = []


    # --------------------------------------------------------
    # Dataset-level performance
    # --------------------------------------------------------

    for metric in [

        "accuracy",

        "balanced_accuracy",

        "macro_f1",

        "weighted_f1"

    ]:

        rows.append(

            {

                "category":

                    "performance",


                "metric":

                    metric,


                "original_100":

                    original[
                        "performance"
                    ][
                        metric
                    ],


                "next_100":

                    next_dataset[
                        "performance"
                    ][
                        metric
                    ],


                "difference":

                    shift[
                        "performance_difference"
                    ][
                        metric
                    ][
                        "difference"
                    ]

            }

        )


    # --------------------------------------------------------
    # Human label distribution
    # --------------------------------------------------------

    for label in LABELS:

        rows.append(

            {

                "category":

                    "human_label_distribution",


                "metric":

                    label,


                "original_100":

                    original[
                        "human_label_distribution"
                    ][
                        "percentages"
                    ][
                        label
                    ],


                "next_100":

                    next_dataset[
                        "human_label_distribution"
                    ][
                        "percentages"
                    ][
                        label
                    ],


                "difference":

                    shift[
                        "human_label_shift"
                    ][
                        label
                    ][
                        "difference"
                    ]

            }

        )


    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    for label in LABELS:

        rows.append(

            {

                "category":

                    "prediction_distribution",


                "metric":

                    label,


                "original_100":

                    original[
                        "prediction_distribution"
                    ][
                        "percentages"
                    ][
                        label
                    ],


                "next_100":

                    next_dataset[
                        "prediction_distribution"
                    ][
                        "percentages"
                    ][
                        label
                    ],


                "difference":

                    shift[
                        "prediction_shift"
                    ][
                        label
                    ][
                        "difference"
                    ]

            }

        )


    df = pd.DataFrame(

        rows

    )


    df.to_csv(

        OUTPUT_CSV,

        index=False

    )


    print("\nSummary CSV saved to:")

    print(
        OUTPUT_CSV
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")

    print("=" * 70)

    print(
        "TRUSTGATE DISTRIBUTION SHIFT ANALYSIS"
    )

    print("=" * 70)


    # ========================================================
    # ORIGINAL 100
    # ========================================================

    original_results = evaluate_dataset(

        ORIGINAL_LABEL_FILE,

        ORIGINAL_PREDICTION_FILE,

        "Original 100 Human-Labeled Dataset"

    )


    # ========================================================
    # NEXT 100
    # ========================================================

    next_results = evaluate_dataset(

        NEXT_LABEL_FILE,

        NEXT_PREDICTION_FILE,

        "Next 100 Independent Dataset"

    )


    # ========================================================
    # DISTRIBUTION SHIFT
    # ========================================================

    shift_results = (

        calculate_distribution_shift(

            original_results,

            next_results

        )

    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    final_results = {

        "original_dataset":

            original_results,


        "next_dataset":

            next_results,


        "distribution_shift":

            shift_results

    }


    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(

        OUTPUT_JSON,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            final_results,

            f,

            indent=4,

            ensure_ascii=False

        )


    print("\n")

    print("=" * 70)

    print(
        "SAVING RESULTS"
    )

    print("=" * 70)


    print(

        "\nJSON saved to:"

    )


    print(

        OUTPUT_JSON

    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    create_summary_csv(

        original_results,

        next_results,

        shift_results

    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "DISTRIBUTION SHIFT ANALYSIS COMPLETE!"
    )

    print("=" * 70)

    print(

        "\nOriginal dataset records:",

        original_results[
            "records"
        ]

    )


    print(

        "Next dataset records:",

        next_results[
            "records"
        ]

    )


    print(

        "\nAnalysis complete successfully."

    )