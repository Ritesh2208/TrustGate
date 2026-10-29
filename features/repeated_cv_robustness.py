import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    RepeatedStratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
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

OUTPUT_FILE = (
    "outputs/trustgate_repeated_cv_robustness.json"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

N_SPLITS = 5
N_REPEATS = 10

RANDOM_STATE = 42


# ============================================================
# LABEL MAP
# ============================================================

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2
}


# ============================================================
# LOAD HUMAN ANNOTATIONS
# ============================================================

print("=" * 60)

print(
    "TRUSTGATE REPEATED CROSS-VALIDATION ROBUSTNESS STUDY"
)

print("=" * 60)


print(
    "\nLoading independent human annotations..."
)


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

print(
    "\nLoading TrustGate 901-D features..."
)


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
# ALIGN DATA BY ID
# ============================================================

print(
    "\nAligning datasets by ID..."
)


feature_map = {
    item["id"]: item
    for item in feature_data
}


records = []


for _, row in annotations.iterrows():

    record_id = int(
        row["id"]
    )


    if record_id not in feature_map:

        raise ValueError(
            f"Missing feature record for ID {record_id}"
        )


    human_label = str(
        row["human_label"]
    ).strip().upper()


    if human_label not in LABEL_MAP:

        raise ValueError(
            f"Unknown human label: {human_label}"
        )


    feature_record = feature_map[
        record_id
    ]


    records.append(
        {
            "id": record_id,

            "label":
                LABEL_MAP[
                    human_label
                ],

            "hidden_state":
                feature_record[
                    "hidden_state"
                ],

            "feature_vector":
                feature_record[
                    "feature_vector"
                ]
        }
    )


print(
    "Aligned records:",
    len(records)
)


# ============================================================
# BUILD REPRESENTATIONS
# ============================================================

hidden_matrix = np.asarray(
    [
        record[
            "hidden_state"
        ]
        for record in records
    ],
    dtype=np.float32
)


fusion_matrix = np.asarray(
    [
        record[
            "feature_vector"
        ]
        for record in records
    ],
    dtype=np.float32
)


y = np.asarray(
    [
        record[
            "label"
        ]
        for record in records
    ],
    dtype=np.int64
)


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
    hidden_matrix.shape
)


print(
    "TrustGate fusion:",
    fusion_matrix.shape
)


if hidden_matrix.shape[1] != 896:

    raise ValueError(
        "Expected hidden state dimension of 896."
    )


if fusion_matrix.shape[1] != 901:

    raise ValueError(
        "Expected TrustGate dimension of 901."
    )


print(
    "\nCONFIRMED:"
)

print(
    "896-D hidden representation"
)

print(
    "901-D TrustGate fusion representation"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "INDEPENDENT HUMAN LABEL DISTRIBUTION"
)

print(
    "=" * 60
)


reverse_map = {
    0: "UNRELIABLE",
    1: "UNCERTAIN",
    2: "RELIABLE"
}


unique, counts = np.unique(
    y,
    return_counts=True
)


for label, count in zip(
    unique,
    counts
):

    print(
        f"{reverse_map[int(label)]}: {count}"
    )


# ============================================================
# SAFETY CHECK
# ============================================================

minimum_class_count = int(
    counts.min()
)


if minimum_class_count < N_SPLITS:

    raise ValueError(
        "Smallest class must contain at least "
        f"{N_SPLITS} examples."
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)


print(
    "\n" + "=" * 60
)

print(
    "REPEATED CROSS-VALIDATION SETTINGS"
)

print(
    "=" * 60
)


print(
    "Splits:",
    N_SPLITS
)


print(
    "Repeats:",
    N_REPEATS
)


print(
    "Total evaluations:",
    N_SPLITS * N_REPEATS
)


# ============================================================
# MODEL
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

                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_representation(
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


    fold_results = []


    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(
            X,
            y
        ),
        start=1
    ):


        X_train = X[
            train_index
        ]

        X_test = X[
            test_index
        ]


        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]


        model = build_model()


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_test
        )


        accuracy = accuracy_score(
            y_test,
            predictions
        )


        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                predictions
            )
        )


        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )


        fold_results.append(
            {
                "fold":
                    fold_number,

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
                    )
            }
        )


        print(
            f"Evaluation {fold_number}/"
            f"{N_SPLITS * N_REPEATS}"
        )


    # --------------------------------------------------------
    # CONVERT TO ARRAYS
    # --------------------------------------------------------

    accuracy_scores = np.asarray(
        [
            item["accuracy"]
            for item in fold_results
        ]
    )


    balanced_scores = np.asarray(
        [
            item[
                "balanced_accuracy"
            ]
            for item in fold_results
        ]
    )


    f1_scores = np.asarray(
        [
            item["macro_f1"]
            for item in fold_results
        ]
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    results = {

        "representation":
            name,

        "dimension":
            int(
                X.shape[1]
            ),

        "evaluations":
            len(
                fold_results
            ),

        "accuracy": {

            "mean":
                float(
                    accuracy_scores.mean()
                ),

            "std":
                float(
                    accuracy_scores.std()
                ),

            "min":
                float(
                    accuracy_scores.min()
                ),

            "max":
                float(
                    accuracy_scores.max()
                )
        },

        "balanced_accuracy": {

            "mean":
                float(
                    balanced_scores.mean()
                ),

            "std":
                float(
                    balanced_scores.std()
                ),

            "min":
                float(
                    balanced_scores.min()
                ),

            "max":
                float(
                    balanced_scores.max()
                )
        },

        "macro_f1": {

            "mean":
                float(
                    f1_scores.mean()
                ),

            "std":
                float(
                    f1_scores.std()
                ),

            "min":
                float(
                    f1_scores.min()
                ),

            "max":
                float(
                    f1_scores.max()
                )
        },

        "fold_results":
            fold_results
    }


    print(
        "\nRESULT SUMMARY"
    )


    print(
        "\nAccuracy:"
    )


    print(
        f"{results['accuracy']['mean']:.4f} "
        f"± "
        f"{results['accuracy']['std']:.4f}"
    )


    print(
        "\nBalanced Accuracy:"
    )


    print(
        f"{results['balanced_accuracy']['mean']:.4f} "
        f"± "
        f"{results['balanced_accuracy']['std']:.4f}"
    )


    print(
        "\nMacro F1:"
    )


    print(
        f"{results['macro_f1']['mean']:.4f} "
        f"± "
        f"{results['macro_f1']['std']:.4f}"
    )


    return results


# ============================================================
# RUN EXPERIMENTS
# ============================================================

hidden_results = evaluate_representation(
    hidden_matrix,
    "Hidden-only (896-D)"
)


fusion_results = evaluate_representation(
    fusion_matrix,
    "TrustGate Fusion (901-D)"
)


# ============================================================
# CALCULATE DIFFERENCES
# ============================================================

accuracy_difference = (

    fusion_results[
        "accuracy"
    ][
        "mean"
    ]

    -

    hidden_results[
        "accuracy"
    ][
        "mean"
    ]
)


balanced_difference = (

    fusion_results[
        "balanced_accuracy"
    ][
        "mean"
    ]

    -

    hidden_results[
        "balanced_accuracy"
    ][
        "mean"
    ]
)


f1_difference = (

    fusion_results[
        "macro_f1"
    ][
        "mean"
    ]

    -

    hidden_results[
        "macro_f1"
    ][
        "mean"
    ]
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "ROBUSTNESS EXPERIMENT SUMMARY"
)

print(
    "=" * 60
)


print()

print(
    f"{'Model':<30}"
    f"{'Accuracy':>20}"
    f"{'Balanced Acc.':>20}"
    f"{'Macro F1':>20}"
)


print(
    "-" * 90
)


print(

    f"{'Hidden-only':<30}"

    f"{hidden_results['accuracy']['mean']:.4f}"

    f" ± {hidden_results['accuracy']['std']:.4f}"

    f"{hidden_results['balanced_accuracy']['mean']:.4f}"

    f" ± {hidden_results['balanced_accuracy']['std']:.4f}"

    f"{hidden_results['macro_f1']['mean']:.4f}"

    f" ± {hidden_results['macro_f1']['std']:.4f}"
)


print(

    f"{'TrustGate Fusion':<30}"

    f"{fusion_results['accuracy']['mean']:.4f}"

    f" ± {fusion_results['accuracy']['std']:.4f}"

    f"{fusion_results['balanced_accuracy']['mean']:.4f}"

    f" ± {fusion_results['balanced_accuracy']['std']:.4f}"

    f"{fusion_results['macro_f1']['mean']:.4f}"

    f" ± {fusion_results['macro_f1']['std']:.4f}"
)


print(
    "\n" + "=" * 60
)

print(
    "MEAN PERFORMANCE DIFFERENCE"
)

print(
    "=" * 60
)


print(
    f"Accuracy difference: "
    f"{accuracy_difference:+.4f}"
)


print(
    f"Balanced accuracy difference: "
    f"{balanced_difference:+.4f}"
)


print(
    f"Macro F1 difference: "
    f"{f1_difference:+.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_data = {

    "experiment":

        "TrustGate Repeated Cross-Validation "
        "Robustness Study",

    "annotation_file":

        ANNOTATION_FILE,

    "feature_file":

        FEATURE_FILE,

    "records":

        len(
            records
        ),

    "cross_validation": {

        "splits":
            N_SPLITS,

        "repeats":
            N_REPEATS,

        "total_evaluations":
            N_SPLITS * N_REPEATS,

        "random_state":
            RANDOM_STATE
    },

    "results": {

        "hidden_only":
            hidden_results,

        "trustgate_fusion":
            fusion_results
    },

    "fusion_difference": {

        "accuracy":
            float(
                accuracy_difference
            ),

        "balanced_accuracy":
            float(
                balanced_difference
            ),

        "macro_f1":
            float(
                f1_difference
            )
    }
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_data,
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
    "REPEATED CROSS-VALIDATION COMPLETE"
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
    "This experiment did NOT modify any existing "
    "TrustGate dataset or model."
)