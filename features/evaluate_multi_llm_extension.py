import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
    "outputs/trustgate_human_evaluation_completed.csv"
)

MASTER_FILE = (
    "outputs/trustgate_100_master_901.json"
)

MULTI_LLM_FILE = (
    "outputs/truthfulqa_100_multi_llm_agreement.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_multi_llm_extension_results.json"
)


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2
}

REVERSE_MAP = {
    0: "UNRELIABLE",
    1: "UNCERTAIN",
    2: "RELIABLE"
}


# ============================================================
# LOAD HUMAN ANNOTATIONS
# ============================================================

print("=" * 60)
print("TRUSTGATE MULTI-LLM EXTENSION EVALUATION")
print("=" * 60)

print("\nLoading independent human annotations...")

human_df = pd.read_csv(
    HUMAN_FILE
)

print(
    "Human annotation records:",
    len(human_df)
)


# Convert labels to NumPy array immediately.
# This avoids pandas / pyarrow indexing problems.

human_by_id = {}

for _, row in human_df.iterrows():

    question_id = int(
        row["id"]
    )

    label = str(
        row["human_label"]
    ).strip().upper()

    if label not in LABEL_MAP:

        raise ValueError(
            f"Invalid human label for ID {question_id}: {label}"
        )

    human_by_id[question_id] = (
        LABEL_MAP[label]
    )


# ============================================================
# LOAD ORIGINAL 901-D TRUSTGATE FEATURES
# ============================================================

print("\nLoading 901-D TrustGate features...")

with open(
    MASTER_FILE,
    "r",
    encoding="utf-8"
) as f:

    master_data = json.load(f)


print(
    "Master records:",
    len(master_data)
)


master_by_id = {}

for item in master_data:

    question_id = int(
        item["id"]
    )

    # Try common feature-vector field names
    if "feature_vector" in item:

        vector = item[
            "feature_vector"
        ]

    elif "trustgate_vector" in item:

        vector = item[
            "trustgate_vector"
        ]

    else:

        raise ValueError(
            f"No 901-D feature vector found for ID {question_id}. "
            f"Available keys: {list(item.keys())}"
        )


    if len(vector) != 901:

        raise ValueError(
            f"ID {question_id} has "
            f"{len(vector)} dimensions. "
            f"Expected 901."
        )


    master_by_id[
        question_id
    ] = vector


# ============================================================
# LOAD MULTI-LLM AGREEMENT
# ============================================================

print("\nLoading Multi-LLM agreement signals...")

with open(
    MULTI_LLM_FILE,
    "r",
    encoding="utf-8"
) as f:

    multi_llm_data = json.load(f)


print(
    "Multi-LLM records:",
    len(multi_llm_data)
)


agreement_by_id = {}

for item in multi_llm_data:

    question_id = int(
        item["id"]
    )

    agreement = float(
        item["multi_llm_agreement"]
    )

    agreement_by_id[
        question_id
    ] = agreement


# ============================================================
# ALIGN IDS
# ============================================================

print("\nAligning datasets by ID...")


human_ids = set(
    human_by_id.keys()
)

master_ids = set(
    master_by_id.keys()
)

agreement_ids = set(
    agreement_by_id.keys()
)


common_ids = sorted(

    human_ids
    &
    master_ids
    &
    agreement_ids

)


if len(common_ids) != 100:

    print(
        "WARNING: Expected 100 aligned records."
    )

    print(
        "Aligned records:",
        len(common_ids)
    )


if len(common_ids) < 20:

    raise ValueError(
        "Too few aligned records."
    )


print(
    "Aligned records:",
    len(common_ids)
)


# ============================================================
# BUILD MATRICES
# ============================================================

print("\nBuilding experiment representations...")


X_hidden = []

X_signals = []

X_original = []

X_extended = []

y = []


for question_id in common_ids:

    vector = np.asarray(
        master_by_id[
            question_id
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Original TrustGate structure:
    #
    # 896 hidden dimensions
    # +
    # 5 trust signals
    # =
    # 901 dimensions
    # --------------------------------------------------------

    hidden = vector[:896]

    signals = vector[896:]


    if len(hidden) != 896:

        raise ValueError(
            f"Hidden dimension error for ID {question_id}"
        )


    if len(signals) != 5:

        raise ValueError(
            f"Signal dimension error for ID {question_id}"
        )


    multi_llm_agreement = (
        agreement_by_id[
            question_id
        ]
    )


    # Extended representation:
    #
    # 896 hidden
    # +
    # 5 original signals
    # +
    # 1 Multi-LLM agreement signal
    # =
    # 902 dimensions

    extended = np.concatenate(

        [

            vector,

            np.array(
                [
                    multi_llm_agreement
                ],
                dtype=np.float32
            )

        ]

    )


    X_hidden.append(
        hidden
    )

    X_signals.append(
        signals
    )

    X_original.append(
        vector
    )

    X_extended.append(
        extended
    )

    y.append(
        human_by_id[
            question_id
        ]
    )


X_hidden = np.asarray(
    X_hidden,
    dtype=np.float32
)

X_signals = np.asarray(
    X_signals,
    dtype=np.float32
)

X_original = np.asarray(
    X_original,
    dtype=np.float32
)

X_extended = np.asarray(
    X_extended,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int64
)


print("\n" + "=" * 60)
print("REPRESENTATION CHECK")
print("=" * 60)

print(
    "Signals-only:",
    X_signals.shape
)

print(
    "Hidden-only:",
    X_hidden.shape
)

print(
    "Original TrustGate:",
    X_original.shape
)

print(
    "Extended TrustGate:",
    X_extended.shape
)


# ============================================================
# CHECK LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("INDEPENDENT HUMAN LABEL DISTRIBUTION")
print("=" * 60)


unique, counts = np.unique(
    y,
    return_counts=True
)


for label_id, count in zip(
    unique,
    counts
):

    print(
        f"{REVERSE_MAP[int(label_id)]:12s}: "
        f"{int(count)}"
    )


minimum_class_count = int(
    counts.min()
)


if minimum_class_count < 2:

    raise ValueError(
        "At least two examples per class are required."
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


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    X
):

    print("\n" + "=" * 60)

    print(
        f"EVALUATING: {name}"
    )

    print("=" * 60)


    model = Pipeline(

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


    print(
        "Generating out-of-fold predictions..."
    )


    predictions = cross_val_predict(

        model,

        X,

        y,

        cv=cv,

        method="predict",

        n_jobs=1

    )


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
        "\nAccuracy:",
        round(
            float(accuracy),
            4
        )
    )


    print(
        "Balanced accuracy:",
        round(
            float(balanced_accuracy),
            4
        )
    )


    print(
        "Macro F1:",
        round(
            float(macro_f1),
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


    return {

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "macro_f1":
            float(macro_f1),

        "confusion_matrix":
            cm.tolist(),

        "predictions":
            predictions.tolist()

    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = {}


results[
    "signals_only_5d"
] = evaluate_model(

    "SIGNALS ONLY (5-D)",

    X_signals

)


# ------------------------------------------------------------

multi_llm_only = np.column_stack(

    [

        X_signals,

        np.array(

            [

                agreement_by_id[
                    question_id
                ]

                for question_id in common_ids

            ],

            dtype=np.float32

        )

    ]

)


results[
    "signals_plus_multi_llm_6d"
] = evaluate_model(

    "SIGNALS + MULTI-LLM (6-D)",

    multi_llm_only

)


# ------------------------------------------------------------

results[
    "hidden_only_896d"
] = evaluate_model(

    "HIDDEN STATES ONLY (896-D)",

    X_hidden

)


# ------------------------------------------------------------

results[
    "original_trustgate_901d"
] = evaluate_model(

    "ORIGINAL TRUSTGATE (901-D)",

    X_original

)


# ------------------------------------------------------------

results[
    "extended_trustgate_902d"
] = evaluate_model(

    "EXTENDED TRUSTGATE + MULTI-LLM (902-D)",

    X_extended

)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MULTI-LLM EXTENSION EXPERIMENT SUMMARY")
print("=" * 60)


print(
    f"{'Model':35s}"
    f"{'Accuracy':>12s}"
    f"{'Balanced Acc.':>16s}"
    f"{'Macro F1':>12s}"
)

print(
    "-" * 75
)


summary_order = [

    (
        "Signals only (5-D)",
        "signals_only_5d"
    ),

    (
        "Signals + Multi-LLM (6-D)",
        "signals_plus_multi_llm_6d"
    ),

    (
        "Hidden only (896-D)",
        "hidden_only_896d"
    ),

    (
        "Original TrustGate (901-D)",
        "original_trustgate_901d"
    ),

    (
        "Extended TrustGate (902-D)",
        "extended_trustgate_902d"
    )

]


for display_name, key in summary_order:

    result = results[key]

    print(

        f"{display_name:35s}"

        f"{result['accuracy']:12.4f}"

        f"{result['balanced_accuracy']:16.4f}"

        f"{result['macro_f1']:12.4f}"

    )


# ============================================================
# IMPROVEMENT ANALYSIS
# ============================================================

original = results[
    "original_trustgate_901d"
]

extended = results[
    "extended_trustgate_902d"
]


print("\n" + "=" * 60)
print("MULTI-LLM CONTRIBUTION")
print("=" * 60)


print(
    "\nExtended vs Original TrustGate:"
)


print(

    "Accuracy difference:",

    f"{extended['accuracy'] - original['accuracy']:+.4f}"

)


print(

    "Balanced accuracy difference:",

    f"{extended['balanced_accuracy'] - original['balanced_accuracy']:+.4f}"

)


print(

    "Macro F1 difference:",

    f"{extended['macro_f1'] - original['macro_f1']:+.4f}"

)


# ============================================================
# SAVE RESULTS
# ============================================================

output_data = {

    "records":

        len(common_ids),

    "representation_dimensions": {

        "signals_only":

            int(
                X_signals.shape[1]
            ),

        "signals_plus_multi_llm":

            int(
                multi_llm_only.shape[1]
            ),

        "hidden_only":

            int(
                X_hidden.shape[1]
            ),

        "original_trustgate":

            int(
                X_original.shape[1]
            ),

        "extended_trustgate":

            int(
                X_extended.shape[1]
            )

    },

    "results":

        results,

    "multi_llm_contribution": {

        "accuracy_difference":

            float(
                extended["accuracy"]
                -
                original["accuracy"]
            ),

        "balanced_accuracy_difference":

            float(
                extended["balanced_accuracy"]
                -
                original["balanced_accuracy"]
            ),

        "macro_f1_difference":

            float(
                extended["macro_f1"]
                -
                original["macro_f1"]
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

print("\n" + "=" * 60)
print("MULTI-LLM EXTENSION EVALUATION COMPLETE")
print("=" * 60)

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
    "No existing TrustGate model or dataset was modified."
)

print(
    "This is a separate experimental evaluation."
)