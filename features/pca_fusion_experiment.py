import json
import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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

MULTI_LLM_FILE = (
    "outputs/truthfulqa_100_multi_llm_agreement.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_pca_fusion_results.json"
)

# PCA dimensions to test
PCA_COMPONENTS = [
    20,
    50,
    75
]

RANDOM_STATE = 42


# ============================================================
# LABEL MAP
# ============================================================

LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2
}

REVERSE_LABEL_MAP = {
    0: "UNRELIABLE",
    1: "UNCERTAIN",
    2: "RELIABLE"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nERROR: File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def normalize_label(label):

    return str(label).strip().upper()


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_representation(
    name,
    X,
    y,
    cv,
    use_pca=False,
    pca_components=None
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

    print(
        "Input shape:",
        X.shape
    )


    # --------------------------------------------------------
    # BUILD MODEL PIPELINE
    #
    # IMPORTANT:
    # PCA is inside the pipeline.
    #
    # Therefore PCA is fitted ONLY on the training fold during
    # cross-validation.
    #
    # This prevents data leakage.
    # --------------------------------------------------------

    steps = []


    steps.append(
        (
            "scaler",
            StandardScaler()
        )
    )


    if use_pca:

        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    random_state=RANDOM_STATE
                )
            )
        )


    steps.append(
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    )


    model = Pipeline(
        steps=steps
    )


    # --------------------------------------------------------
    # CROSS-VALIDATION PREDICTIONS
    # --------------------------------------------------------

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
        average="macro",
        zero_division=0
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


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print(
        "\nClassification report:"
    )


    report = classification_report(
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


    print(
        report
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "name":
            name,

        "input_dimension":
            int(
                X.shape[1]
            ),

        "pca_used":
            bool(
                use_pca
            ),

        "pca_components":
            (
                int(pca_components)
                if use_pca
                else None
            ),

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

        "confusion_matrix":
            cm.tolist(),

        "predictions":
            [
                int(value)
                for value in predictions
            ]
    }


# ============================================================
# START
# ============================================================

print(
    "=" * 60
)

print(
    "TRUSTGATE PCA FUSION EXPERIMENT"
)

print(
    "=" * 60
)


# ============================================================
# LOAD HUMAN ANNOTATIONS
# ============================================================

print(
    "\nLoading independent human annotations..."
)


if not os.path.exists(
    ANNOTATION_FILE
):

    raise FileNotFoundError(
        f"\nERROR: Annotation file not found:\n"
        f"{ANNOTATION_FILE}"
    )


annotations_df = pd.read_csv(
    ANNOTATION_FILE
)


print(
    "Annotation records:",
    len(
        annotations_df
    )
)


# ------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_annotation_columns = [
    "id",
    "human_label"
]


for column in required_annotation_columns:

    if column not in annotations_df.columns:

        raise ValueError(
            f"\nMissing annotation column: {column}\n"
            f"Available columns:\n"
            f"{annotations_df.columns.tolist()}"
        )


# ============================================================
# LOAD TRUSTGATE MASTER DATA
# ============================================================

print(
    "\nLoading 901-D TrustGate data..."
)


master_data = load_json(
    MASTER_FILE
)


print(
    "Master records:",
    len(
        master_data
    )
)


# ============================================================
# LOAD MULTI-LLM DATA
# ============================================================

print(
    "\nLoading Multi-LLM agreement..."
)


multi_llm_data = load_json(
    MULTI_LLM_FILE
)


print(
    "Multi-LLM records:",
    len(
        multi_llm_data
    )
)


# ============================================================
# CREATE LOOKUPS
# ============================================================

master_by_id = {

    int(item["id"]): item

    for item in master_data
}


multi_llm_by_id = {

    int(item["id"]): item

    for item in multi_llm_data
}


annotations_by_id = {

    int(row["id"]): row

    for _,
    row in annotations_df.iterrows()
}


# ============================================================
# CHECK ID ALIGNMENT
# ============================================================

annotation_ids = set(
    annotations_by_id.keys()
)

master_ids = set(
    master_by_id.keys()
)

multi_llm_ids = set(
    multi_llm_by_id.keys()
)


common_ids = sorted(

    annotation_ids
    &
    master_ids
    &
    multi_llm_ids
)


print(
    "\n" + "=" * 60
)

print(
    "DATA ALIGNMENT"
)

print(
    "=" * 60
)


print(
    "Annotation IDs:",
    len(
        annotation_ids
    )
)

print(
    "Master IDs:",
    len(
        master_ids
    )
)

print(
    "Multi-LLM IDs:",
    len(
        multi_llm_ids
    )
)

print(
    "Common aligned IDs:",
    len(
        common_ids
    )
)


if len(common_ids) < 10:

    raise ValueError(
        "Too few aligned records."
    )


# ============================================================
# BUILD REPRESENTATIONS
# ============================================================

print(
    "\nBuilding aligned feature representations..."
)


hidden_features = []

trust_signals = []

multi_llm_signals = []

labels = []

record_ids = []


for record_id in common_ids:


    # --------------------------------------------------------
    # GET RECORDS
    # --------------------------------------------------------

    annotation = annotations_by_id[
        record_id
    ]

    master = master_by_id[
        record_id
    ]

    multi_llm = multi_llm_by_id[
        record_id
    ]


    # --------------------------------------------------------
    # HUMAN LABEL
    # --------------------------------------------------------

    human_label = normalize_label(
        annotation[
            "human_label"
        ]
    )


    if human_label not in LABEL_MAP:

        raise ValueError(
            f"Unknown human label for ID "
            f"{record_id}: {human_label}"
        )


    # --------------------------------------------------------
    # HIDDEN STATE
    # --------------------------------------------------------

    hidden = master.get(
        "hidden_state"
    )


    if hidden is None:

        raise ValueError(
            f"Missing hidden_state for ID "
            f"{record_id}"
        )


    if len(hidden) != 896:

        raise ValueError(
            f"ID {record_id} has "
            f"{len(hidden)} hidden dimensions. "
            f"Expected 896."
        )


    # --------------------------------------------------------
    # FIVE TRUST SIGNALS
    # --------------------------------------------------------

    signal_names = [

        "confidence",

        "entropy",

        "self_consistency",

        "retrieval_agreement",

        "verifier_score"
    ]


    signals = []


    for signal_name in signal_names:


        if signal_name not in master:

            raise ValueError(
                f"Missing signal "
                f"'{signal_name}' "
                f"for ID {record_id}"
            )


        value = master[
            signal_name
        ]


        signals.append(
            float(value)
        )


    # --------------------------------------------------------
    # MULTI-LLM AGREEMENT
    # --------------------------------------------------------

    if "multi_llm_agreement" not in multi_llm:

        raise ValueError(
            f"Missing multi_llm_agreement "
            f"for ID {record_id}"
        )


    agreement = float(

        multi_llm[
            "multi_llm_agreement"
        ]
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    hidden_features.append(
        hidden
    )


    trust_signals.append(
        signals
    )


    multi_llm_signals.append(
        [
            agreement
        ]
    )


    labels.append(

        LABEL_MAP[
            human_label
        ]
    )


    record_ids.append(
        record_id
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_hidden = np.asarray(
    hidden_features,
    dtype=np.float64
)


X_signals = np.asarray(
    trust_signals,
    dtype=np.float64
)


X_multi_llm = np.asarray(
    multi_llm_signals,
    dtype=np.float64
)


y = np.asarray(
    labels,
    dtype=np.int64
)


# ============================================================
# VALIDATION
# ============================================================

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
    "Hidden:",
    X_hidden.shape
)


print(
    "Trust signals:",
    X_signals.shape
)


print(
    "Multi-LLM:",
    X_multi_llm.shape
)


if X_hidden.shape[1] != 896:

    raise ValueError(
        "Hidden representation "
        "must have 896 dimensions."
    )


if X_signals.shape[1] != 5:

    raise ValueError(
        "Trust signal representation "
        "must have 5 dimensions."
    )


if X_multi_llm.shape[1] != 1:

    raise ValueError(
        "Multi-LLM representation "
        "must have 1 dimension."
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


unique_labels, counts = np.unique(

    y,

    return_counts=True
)


for label_id, count in zip(

    unique_labels,

    counts
):

    print(

        f"{REVERSE_LABEL_MAP[int(label_id)]}: ",

        int(count)
    )


minimum_class_count = int(
    counts.min()
)


if minimum_class_count < 2:

    raise ValueError(
        "At least two examples "
        "per class are required."
    )


# ============================================================
# CROSS-VALIDATION
# ============================================================

n_splits = min(

    5,

    minimum_class_count
)


cv = StratifiedKFold(

    n_splits=n_splits,

    shuffle=True,

    random_state=RANDOM_STATE
)


print(
    "\nCross-validation folds:",
    n_splits
)


# ============================================================
# EXPERIMENTS
# ============================================================

all_results = []


# ============================================================
# TEST EACH PCA DIMENSION
# ============================================================

for n_components in PCA_COMPONENTS:


    print(
        "\n"
        + "#"
        * 70
    )


    print(
        f"PCA COMPONENTS: "
        f"{n_components}"
    )


    print(
        "#"
        * 70
    )


    # --------------------------------------------------------
    # EXPERIMENT 1
    #
    # PCA HIDDEN ONLY
    # --------------------------------------------------------

    result_hidden = evaluate_representation(

        name=(
            f"PCA Hidden Only "
            f"({n_components}-D)"
        ),

        X=X_hidden,

        y=y,

        cv=cv,

        use_pca=True,

        pca_components=n_components
    )


    result_hidden[
        "experiment_type"
    ] = "pca_hidden_only"


    all_results.append(
        result_hidden
    )


    # --------------------------------------------------------
    # EXPERIMENT 2
    #
    # PCA HIDDEN + 5 TRUST SIGNALS
    #
    # IMPORTANT:
    #
    # We cannot simply put PCA on the entire combined matrix.
    #
    # PCA should only reduce the hidden-state representation.
    #
    # Therefore this experiment uses PCA outside the simple
    # pipeline structure via a custom representation strategy.
    #
    # To preserve correct CV behavior, we evaluate manually.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        f"EVALUATING: PCA Hidden + "
        f"5 Trust Signals "
        f"({n_components} + 5)"
    )

    print(
        "=" * 60
    )


    predictions = np.zeros(
        len(y),
        dtype=np.int64
    )


    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(
            X_hidden,
            y
        ),
        start=1
    ):


        print(
            f"Processing fold "
            f"{fold_number}/{n_splits}"
        )


        # Training data

        X_hidden_train = (
            X_hidden[
                train_index
            ]
        )

        X_hidden_test = (
            X_hidden[
                test_index
            ]
        )


        X_signal_train = (
            X_signals[
                train_index
            ]
        )

        X_signal_test = (
            X_signals[
                test_index
            ]
        )


        y_train = y[
            train_index
        ]


        # ----------------------------------------------
        # SCALE + PCA HIDDEN STATES
        # ----------------------------------------------

        hidden_scaler = StandardScaler()


        X_hidden_train_scaled = (
            hidden_scaler.fit_transform(
                X_hidden_train
            )
        )


        X_hidden_test_scaled = (
            hidden_scaler.transform(
                X_hidden_test
            )
        )


        pca = PCA(

            n_components=n_components,

            random_state=RANDOM_STATE
        )


        X_hidden_train_pca = (
            pca.fit_transform(
                X_hidden_train_scaled
            )
        )


        X_hidden_test_pca = (
            pca.transform(
                X_hidden_test_scaled
            )
        )


        # ----------------------------------------------
        # SCALE TRUST SIGNALS
        # ----------------------------------------------

        signal_scaler = StandardScaler()


        X_signal_train_scaled = (
            signal_scaler.fit_transform(
                X_signal_train
            )
        )


        X_signal_test_scaled = (
            signal_scaler.transform(
                X_signal_test
            )
        )


        # ----------------------------------------------
        # FUSE
        # ----------------------------------------------

        X_train_fused = np.hstack(

            [

                X_hidden_train_pca,

                X_signal_train_scaled
            ]
        )


        X_test_fused = np.hstack(

            [

                X_hidden_test_pca,

                X_signal_test_scaled
            ]
        )


        # ----------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------

        classifier = LogisticRegression(

            max_iter=5000,

            class_weight="balanced",

            random_state=RANDOM_STATE
        )


        classifier.fit(

            X_train_fused,

            y_train
        )


        predictions[
            test_index
        ] = classifier.predict(

            X_test_fused
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

        average="macro",

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


    result_fusion = {

        "name": (
            f"PCA Hidden + "
            f"5 Trust Signals "
            f"({n_components} + 5-D)"
        ),

        "experiment_type":
            "pca_hidden_plus_signals",

        "input_dimension":

            int(
                n_components + 5
            ),

        "pca_used":
            True,

        "pca_components":

            int(
                n_components
            ),

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

        "confusion_matrix":

            cm.tolist(),

        "predictions":

            [
                int(value)
                for value
                in predictions
            ]
    }


    all_results.append(
        result_fusion
    )


    # ========================================================
    # EXPERIMENT 3
    #
    # PCA HIDDEN + MULTI-LLM
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        f"EVALUATING: PCA Hidden + "
        f"Multi-LLM "
        f"({n_components} + 1)"
    )

    print(
        "=" * 60
    )


    predictions = np.zeros(
        len(y),
        dtype=np.int64
    )


    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(
            X_hidden,
            y
        ),
        start=1
    ):


        X_hidden_train = (
            X_hidden[
                train_index
            ]
        )

        X_hidden_test = (
            X_hidden[
                test_index
            ]
        )


        X_multi_train = (
            X_multi_llm[
                train_index
            ]
        )

        X_multi_test = (
            X_multi_llm[
                test_index
            ]
        )


        y_train = y[
            train_index
        ]


        hidden_scaler = (
            StandardScaler()
        )


        X_hidden_train_scaled = (
            hidden_scaler.fit_transform(
                X_hidden_train
            )
        )


        X_hidden_test_scaled = (
            hidden_scaler.transform(
                X_hidden_test
            )
        )


        pca = PCA(

            n_components=n_components,

            random_state=RANDOM_STATE
        )


        X_hidden_train_pca = (
            pca.fit_transform(
                X_hidden_train_scaled
            )
        )


        X_hidden_test_pca = (
            pca.transform(
                X_hidden_test_scaled
            )
        )


        multi_scaler = (
            StandardScaler()
        )


        X_multi_train_scaled = (
            multi_scaler.fit_transform(
                X_multi_train
            )
        )


        X_multi_test_scaled = (
            multi_scaler.transform(
                X_multi_test
            )
        )


        X_train_fused = np.hstack(

            [

                X_hidden_train_pca,

                X_multi_train_scaled
            ]
        )


        X_test_fused = np.hstack(

            [

                X_hidden_test_pca,

                X_multi_test_scaled
            ]
        )


        classifier = LogisticRegression(

            max_iter=5000,

            class_weight="balanced",

            random_state=RANDOM_STATE
        )


        classifier.fit(

            X_train_fused,

            y_train
        )


        predictions[
            test_index
        ] = classifier.predict(

            X_test_fused
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


    cm = confusion_matrix(

        y,

        predictions,

        labels=[
            0,
            1,
            2
        ]
    )


    result_multi = {

        "name": (
            f"PCA Hidden + "
            f"Multi-LLM "
            f"({n_components} + 1-D)"
        ),

        "experiment_type":
            "pca_hidden_plus_multi_llm",

        "input_dimension":

            int(
                n_components + 1
            ),

        "pca_used":
            True,

        "pca_components":

            int(
                n_components
            ),

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

        "confusion_matrix":

            cm.tolist(),

        "predictions":

            [
                int(value)
                for value
                in predictions
            ]
    }


    all_results.append(
        result_multi
    )


    # ========================================================
    # EXPERIMENT 4
    #
    # PCA HIDDEN + ALL SIGNALS + MULTI-LLM
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        f"EVALUATING: FULL PCA TRUSTGATE "
        f"({n_components} + 5 + 1)"
    )

    print(
        "=" * 60
    )


    predictions = np.zeros(
        len(y),
        dtype=np.int64
    )


    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(
            X_hidden,
            y
        ),
        start=1
    ):


        X_hidden_train = (
            X_hidden[
                train_index
            ]
        )

        X_hidden_test = (
            X_hidden[
                test_index
            ]
        )


        X_signal_train = (
            X_signals[
                train_index
            ]
        )

        X_signal_test = (
            X_signals[
                test_index
            ]
        )


        X_multi_train = (
            X_multi_llm[
                train_index
            ]
        )

        X_multi_test = (
            X_multi_llm[
                test_index
            ]
        )


        y_train = y[
            train_index
        ]


        # ----------------------------------------------
        # HIDDEN STATES
        # ----------------------------------------------

        hidden_scaler = (
            StandardScaler()
        )


        X_hidden_train_scaled = (
            hidden_scaler.fit_transform(
                X_hidden_train
            )
        )


        X_hidden_test_scaled = (
            hidden_scaler.transform(
                X_hidden_test
            )
        )


        pca = PCA(

            n_components=n_components,

            random_state=RANDOM_STATE
        )


        X_hidden_train_pca = (
            pca.fit_transform(
                X_hidden_train_scaled
            )
        )


        X_hidden_test_pca = (
            pca.transform(
                X_hidden_test_scaled
            )
        )


        # ----------------------------------------------
        # TRUST SIGNALS
        # ----------------------------------------------

        signal_scaler = (
            StandardScaler()
        )


        X_signal_train_scaled = (
            signal_scaler.fit_transform(
                X_signal_train
            )
        )


        X_signal_test_scaled = (
            signal_scaler.transform(
                X_signal_test
            )
        )


        # ----------------------------------------------
        # MULTI-LLM SIGNAL
        # ----------------------------------------------

        multi_scaler = (
            StandardScaler()
        )


        X_multi_train_scaled = (
            multi_scaler.fit_transform(
                X_multi_train
            )
        )


        X_multi_test_scaled = (
            multi_scaler.transform(
                X_multi_test
            )
        )


        # ----------------------------------------------
        # FULL FUSION
        # ----------------------------------------------

        X_train_fused = np.hstack(

            [

                X_hidden_train_pca,

                X_signal_train_scaled,

                X_multi_train_scaled
            ]
        )


        X_test_fused = np.hstack(

            [

                X_hidden_test_pca,

                X_signal_test_scaled,

                X_multi_test_scaled
            ]
        )


        # ----------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------

        classifier = LogisticRegression(

            max_iter=5000,

            class_weight="balanced",

            random_state=RANDOM_STATE
        )


        classifier.fit(

            X_train_fused,

            y_train
        )


        predictions[
            test_index
        ] = classifier.predict(

            X_test_fused
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


    result_full = {

        "name": (
            f"Full PCA TrustGate "
            f"({n_components} + 5 + 1-D)"
        ),

        "experiment_type":
            "full_pca_trustgate",

        "input_dimension":

            int(
                n_components + 6
            ),

        "pca_used":
            True,

        "pca_components":

            int(
                n_components
            ),

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

        "confusion_matrix":

            cm.tolist(),

        "predictions":

            [
                int(value)
                for value
                in predictions
            ]
    }


    all_results.append(
        result_full
    )


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PCA FUSION EXPERIMENT SUMMARY"
)

print(
    "=" * 70
)


summary_rows = []


for result in all_results:

    summary_rows.append(

        {

            "Model":
                result["name"],

            "Input Dimension":
                result[
                    "input_dimension"
                ],

            "PCA Components":
                result[
                    "pca_components"
                ],

            "Accuracy":
                result[
                    "accuracy"
                ],

            "Balanced Accuracy":
                result[
                    "balanced_accuracy"
                ],

            "Macro F1":
                result[
                    "macro_f1"
                ]
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


print()

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# FIND BEST MODEL
# ============================================================

best_result = max(

    all_results,

    key=lambda result:

        result[
            "macro_f1"
        ]
)


print(
    "\n" + "=" * 60
)

print(
    "BEST PCA FUSION RESULT"
)

print(
    "=" * 60
)


print(
    "Model:",
    best_result[
        "name"
    ]
)


print(
    "Accuracy:",
    round(

        best_result[
            "accuracy"
        ],

        4
    )
)


print(
    "Balanced Accuracy:",
    round(

        best_result[
            "balanced_accuracy"
        ],

        4
    )
)


print(
    "Macro F1:",
    round(

        best_result[
            "macro_f1"
        ],

        4
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_data = {

    "experiment":

        "TrustGate PCA Fusion Experiment",

    "records":

        int(
            len(y)
        ),

    "hidden_dimension":

        int(
            X_hidden.shape[1]
        ),

    "trust_signal_dimension":

        int(
            X_signals.shape[1]
        ),

    "multi_llm_dimension":

        int(
            X_multi_llm.shape[1]
        ),

    "pca_components_tested":

        PCA_COMPONENTS,

    "cross_validation_folds":

        int(
            n_splits
        ),

    "label_distribution":

        {

            REVERSE_LABEL_MAP[
                int(label_id)
            ]:

            int(count)

            for label_id,
            count

            in zip(

                unique_labels,

                counts
            )
        },

    "results":

        all_results,

    "best_result":

        best_result
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
    "PCA FUSION EXPERIMENT COMPLETE"
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
    "No existing TrustGate model or dataset was modified."
)


print(
    "PCA was fitted separately inside each "
    "cross-validation training fold."
)