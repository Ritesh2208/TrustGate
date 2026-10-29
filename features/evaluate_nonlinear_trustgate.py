import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# SETTINGS
# ============================================================

ANNOTATION_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

MASTER_FILE = (
    "outputs/trustgate_100_master_901.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_nonlinear_evaluation.json"
)


# ============================================================
# LABEL MAP
# ============================================================

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2,
}

REVERSE_MAP = {
    0: "UNRELIABLE",
    1: "UNCERTAIN",
    2: "RELIABLE",
}


# ============================================================
# LOAD INDEPENDENT HUMAN ANNOTATIONS
# ============================================================

print("=" * 60)
print("TRUSTGATE NONLINEAR EVALUATION")
print("=" * 60)

print("\nLoading independent human annotations...")

annotations_df = pd.read_csv(
    ANNOTATION_FILE
)

print(
    "Annotation records:",
    len(annotations_df)
)

print(
    "Annotation columns:"
)

print(
    annotations_df.columns.tolist()
)


# ============================================================
# LOAD TRUSTGATE FEATURES
# ============================================================

print("\nLoading TrustGate 901-D features...")

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


# ============================================================
# CREATE LOOKUP TABLES
# ============================================================

annotation_map = {

    int(row["id"]): row

    for _, row in annotations_df.iterrows()
}


master_map = {

    int(item["id"]): item

    for item in master_data
}


# ============================================================
# ALIGN BY ID
# ============================================================

common_ids = sorted(

    set(annotation_map.keys())

    &

    set(master_map.keys())
)


print(
    "\nAligned records:",
    len(common_ids)
)


if len(common_ids) == 0:

    raise ValueError(
        "No matching IDs found."
    )


# ============================================================
# BUILD REPRESENTATIONS
# ============================================================

hidden_features = []

signal_features = []

fusion_features = []

labels = []


for item_id in common_ids:

    annotation = annotation_map[
        item_id
    ]

    master = master_map[
        item_id
    ]


    # --------------------------------------------------------
    # HUMAN LABEL
    # --------------------------------------------------------

    human_label = str(

        annotation[
            "human_label"
        ]

    ).strip()


    if human_label not in LABEL_MAP:

        raise ValueError(

            f"Unknown human label "

            f"for ID {item_id}: "

            f"{human_label}"
        )


    labels.append(

        LABEL_MAP[
            human_label
        ]
    )


    # --------------------------------------------------------
    # HIDDEN STATE
    # --------------------------------------------------------

    hidden = master[
        "hidden_state"
    ]


    if len(hidden) != 896:

        raise ValueError(

            f"ID {item_id} has "

            f"{len(hidden)} hidden dimensions. "

            f"Expected 896."
        )


    # --------------------------------------------------------
    # TRUST SIGNALS
    # --------------------------------------------------------

    signals = [

        float(
            master["confidence"]
        ),

        float(
            master["entropy"]
        ),

        float(
            master["self_consistency"]
        ),

        float(
            master["retrieval_agreement"]
        ),

        float(
            master["verifier_score"]
        )
    ]


    # --------------------------------------------------------
    # STORE REPRESENTATIONS
    # --------------------------------------------------------

    hidden_features.append(
        hidden
    )

    signal_features.append(
        signals
    )

    fusion_features.append(

        hidden + signals
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_hidden = np.asarray(

    hidden_features,

    dtype=np.float32
)


X_signals = np.asarray(

    signal_features,

    dtype=np.float32
)


X_fusion = np.asarray(

    fusion_features,

    dtype=np.float32
)


y = np.asarray(

    labels,

    dtype=np.int64
)


# ============================================================
# REPRESENTATION CHECK
# ============================================================

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
    "TrustGate fusion:",
    X_fusion.shape
)


if X_hidden.shape[1] != 896:

    raise ValueError(
        "Hidden state dimension mismatch."
    )


if X_signals.shape[1] != 5:

    raise ValueError(
        "Signal dimension mismatch."
    )


if X_fusion.shape[1] != 901:

    raise ValueError(
        "Fusion dimension mismatch."
    )


print(
    "\nCONFIRMED:"
)

print(
    "896 hidden + 5 signals = 901 fusion"
)


# ============================================================
# LABEL DISTRIBUTION
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

        f"{REVERSE_MAP[int(label_id)]:<12}",

        ":",

        int(count)
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

minimum_class_count = int(
    counts.min()
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


print(
    "\nCross-validation folds:",
    n_splits
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_representation(

    name,

    X
):

    print("\n" + "=" * 60)

    print(
        f"EVALUATING: {name}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # RANDOM FOREST CLASSIFIER
    # --------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=8,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )


    print(
        "\nGenerating out-of-fold predictions..."
    )


    predictions = cross_val_predict(

        model,

        X,

        y,

        cv=cv,

        method="predict",

        n_jobs=1
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


    print()

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
        "Macro F1:",
        round(
            macro_f1,
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
            float(
                balanced_accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "confusion_matrix":
            cm.tolist()
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = {}


# ------------------------------------------------------------
# SIGNALS ONLY
# ------------------------------------------------------------

results[
    "signals_only"
] = evaluate_representation(

    "SIGNALS ONLY (5-D)",

    X_signals
)


# ------------------------------------------------------------
# HIDDEN STATES ONLY
# ------------------------------------------------------------

results[
    "hidden_only"
] = evaluate_representation(

    "HIDDEN STATES ONLY (896-D)",

    X_hidden
)


# ------------------------------------------------------------
# TRUSTGATE FUSION
# ------------------------------------------------------------

results[
    "trustgate_fusion"
] = evaluate_representation(

    "TRUSTGATE FUSION (901-D)",

    X_fusion
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)

print(
    "NONLINEAR TRUSTGATE COMPARISON"
)

print("=" * 60)


print()

print(

    f"{'Model':<30}"

    f"{'Accuracy':>12}"

    f"{'Balanced Acc.':>18}"

    f"{'Macro F1':>14}"
)


print(
    "-" * 76
)


display_names = {

    "signals_only":

        "Signals Only",

    "hidden_only":

        "Hidden Only",

    "trustgate_fusion":

        "TrustGate Fusion"
}


for key in [

    "signals_only",

    "hidden_only",

    "trustgate_fusion"
]:

    result = results[key]

    print(

        f"{display_names[key]:<30}"

        f"{result['accuracy']:>12.4f}"

        f"{result['balanced_accuracy']:>18.4f}"

        f"{result['macro_f1']:>14.4f}"
    )


# ============================================================
# FUSION CONTRIBUTION
# ============================================================

fusion = results[
    "trustgate_fusion"
]


hidden = results[
    "hidden_only"
]


print("\n" + "=" * 60)

print(
    "FUSION CONTRIBUTION"
)

print("=" * 60)


print(
    "\nFusion vs Hidden-only:"
)


print(

    "Accuracy difference:",

    f"{fusion['accuracy'] - hidden['accuracy']:+.4f}"
)


print(

    "Balanced accuracy difference:",

    f"{fusion['balanced_accuracy'] - hidden['balanced_accuracy']:+.4f}"
)


print(

    "Macro F1 difference:",

    f"{fusion['macro_f1'] - hidden['macro_f1']:+.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "experiment":

        "TrustGate nonlinear evaluation",

    "classifier":

        "RandomForestClassifier",

    "records":

        int(
            len(common_ids)
        ),

    "label_distribution": {

        REVERSE_MAP[
            int(label_id)
        ]:

        int(count)

        for label_id, count in zip(

            unique,

            counts
        )
    },

    "representation_dimensions": {

        "signals_only":

            int(
                X_signals.shape[1]
            ),

        "hidden_only":

            int(
                X_hidden.shape[1]
            ),

        "trustgate_fusion":

            int(
                X_fusion.shape[1]
            )
    },

    "cross_validation_folds":

        int(
            n_splits
        ),

    "results":

        results,

    "fusion_vs_hidden": {

        "accuracy_difference":

            float(

                fusion["accuracy"]

                -

                hidden["accuracy"]
            ),

        "balanced_accuracy_difference":

            float(

                fusion["balanced_accuracy"]

                -

                hidden["balanced_accuracy"]
            ),

        "macro_f1_difference":

            float(

                fusion["macro_f1"]

                -

                hidden["macro_f1"]
            )
    }
}


with open(

    OUTPUT_FILE,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        output,

        f,

        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)

print(
    "NONLINEAR TRUSTGATE EVALUATION COMPLETE"
)

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
    "This experiment did not modify "
    "any existing TrustGate dataset or model."
)