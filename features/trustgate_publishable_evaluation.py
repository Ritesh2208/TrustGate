import json
import os
import random
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# TRUSTGATE PUBLICATION-QUALITY EVALUATION
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_LABEL_FILE = (
    "outputs/trustgate_human_evaluation_completed.csv"
)

NEXT_LABEL_FILE = (
    "outputs/trustgate_ai_assisted_evaluation_next_100.csv"
)

ORIGINAL_FEATURE_FILE = (
    "outputs/truthfulqa_100_combined_features.json"
)

NEXT_FEATURE_FILE = (
    "outputs/trustgate_next_100_master_901.json"
)


OUTPUT_JSON = (
    "outputs/trustgate_publishable_evaluation.json"
)

OUTPUT_CSV = (
    "outputs/trustgate_publishable_results.csv"
)


RANDOM_SEED = 42

NUM_FOLDS = 5

NUM_REPEATS = 10

BOOTSTRAP_ITERATIONS = 1000


LABEL_MAP = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2
}


LABEL_NAMES = [
    "UNRELIABLE",
    "UNCERTAIN",
    "RELIABLE"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)


set_seed(RANDOM_SEED)


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_files():

    print()

    print("=" * 70)

    print("VALIDATING INPUT FILES")

    print("=" * 70)

    required_files = [

        ORIGINAL_LABEL_FILE,

        NEXT_LABEL_FILE,

        ORIGINAL_FEATURE_FILE,

        NEXT_FEATURE_FILE

    ]

    for path in required_files:

        exists = os.path.exists(path)

        print()

        print("File:", path)

        print("Exists:", exists)

        if not exists:

            raise FileNotFoundError(

                f"Required file not found:\n{path}"

            )

    output_directory = os.path.dirname(OUTPUT_JSON)

    if output_directory:

        os.makedirs(

            output_directory,

            exist_ok=True

        )

    print()

    print("All required files found.")


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    print()

    print(f"Loading feature file: {path}")

    with open(

        path,

        "r",

        encoding="utf-8"

    ) as f:

        data = json.load(f)

    if not isinstance(data, list):

        raise ValueError(

            f"Expected JSON list in {path}, "

            f"but found {type(data)}"

        )

    print(

        "Records loaded:",

        len(data)

    )

    return data


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels(path, dataset_name):

    print()

    print(

        f"Loading labels: {dataset_name}"

    )

    df = pd.read_csv(path)

    print(

        "Records:",

        len(df)

    )

    required_columns = [

        "id",

        "human_label"

    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(

                f"{dataset_name} is missing "

                f"required column: {column}"

            )

    df["id"] = (

        df["id"]
        .astype(str)
        .str.strip()

    )

    df["human_label"] = (

        df["human_label"]
        .astype(str)
        .str.strip()
        .str.upper()

    )

    invalid = df[

        ~df["human_label"].isin(

            LABEL_MAP.keys()

        )

    ]

    if len(invalid) > 0:

        print()

        print("INVALID LABELS:")

        print(invalid)

        raise ValueError(

            f"Invalid labels found in "

            f"{dataset_name}"

        )

    print(

        "Validated labels:",

        len(df)

    )

    print()

    print(

        "Label distribution:"

    )

    print(

        df["human_label"]
        .value_counts()
        .sort_index()

    )

    return df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(item):


    # ========================================================
    # CASE 1:
    # fusion_features
    # ========================================================

    if "fusion_features" in item:

        vector = np.array(

            item["fusion_features"],

            dtype=np.float32

        )

        return vector


    # ========================================================
    # CASE 2:
    # hidden_features + signals dictionary
    # ========================================================

    if (

        "hidden_features" in item

        and

        "signals" in item

    ):

        hidden = np.array(

            item["hidden_features"],

            dtype=np.float32

        )

        signals_dict = item["signals"]


        signal_order = [

            "confidence",

            "entropy",

            "self_consistency",

            "retrieval_agreement",

            "verifier_score"

        ]


        signals = np.array(

            [

                signals_dict.get(

                    key,

                    0.0

                )

                for key in signal_order

            ],

            dtype=np.float32

        )

        return np.concatenate(

            [

                hidden,

                signals

            ]

        )


    # ========================================================
    # CASE 3:
    # hidden_state / hidden_features + individual signals
    # ========================================================

    hidden = item.get(

        "hidden_state",

        item.get(

            "hidden_features",

            None

        )

    )


    if hidden is None:

        raise ValueError(

            "Could not locate hidden features."

        )


    hidden = np.array(

        hidden,

        dtype=np.float32

    )


    signals = np.array(

        [

            item.get(

                "confidence",

                0.0

            ),

            item.get(

                "entropy",

                0.0

            ),

            item.get(

                "self_consistency",

                0.0

            ),

            item.get(

                "retrieval_agreement",

                0.0

            ),

            item.get(

                "verifier_score",

                0.0

            )

        ],

        dtype=np.float32

    )


    return np.concatenate(

        [

            hidden,

            signals

        ]

    )


# ============================================================
# ALIGN DATASET
# ============================================================

def align_dataset(

    labels_df,

    feature_data,

    dataset_name

):

    print()

    print(

        "=" * 70

    )

    print(

        f"ALIGNING {dataset_name.upper()} DATASET"

    )

    print(

        "=" * 70

    )


    feature_map = {}


    for item in feature_data:

        if "id" not in item:

            raise ValueError(

                f"Feature record in "

                f"{dataset_name} "

                f"does not contain an id."

            )

        item_id = str(

            item["id"]

        ).strip()


        feature_map[item_id] = item


    X_list = []

    y_list = []

    ids = []

    missing = []


    for _, row in labels_df.iterrows():

        item_id = str(

            row["id"]

        ).strip()


        if item_id not in feature_map:

            missing.append(

                item_id

            )

            continue


        vector = extract_features(

            feature_map[item_id]

        )


        if len(vector) != 901:

            raise ValueError(

                f"ID {item_id} has "

                f"{len(vector)} features "

                f"instead of 901."

            )


        if not np.all(

            np.isfinite(vector)

        ):

            raise ValueError(

                f"ID {item_id} contains "

                f"NaN or infinite values."

            )


        X_list.append(

            vector

        )


        y_list.append(

            LABEL_MAP[

                row["human_label"]

            ]

        )


        ids.append(

            item_id

        )


    if len(X_list) == 0:

        raise ValueError(

            f"No records aligned for "

            f"{dataset_name}"

        )


    X = np.array(

        X_list,

        dtype=np.float32

    )


    y = np.array(

        y_list,

        dtype=np.int64

    )


    print()

    print(

        f"{dataset_name} aligned records:",

        len(X)

    )

    print(

        "Feature dimension:",

        X.shape[1]

    )


    if missing:

        print()

        print(

            "WARNING: Missing feature IDs:",

            len(missing)

        )

        print(

            missing[:20]

        )


    label_counts = {

        LABEL_NAMES[class_id]:

            int(

                np.sum(

                    y == class_id

                )

            )

        for class_id in range(3)

    }


    print()

    print(

        "Aligned label distribution:"

    )

    print(

        label_counts

    )


    return (

        X,

        y,

        ids

    )


# ============================================================
# REPRESENTATIONS
# ============================================================

def get_representation(

    X,

    representation

):

    if representation == "signals_only":

        return X[:, 896:]


    if representation == "hidden_only":

        return X[:, :896]


    if representation == "fusion":

        return X


    raise ValueError(

        f"Unknown representation: "

        f"{representation}"

    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(

    y_true,

    y_pred

):

    return {

        "accuracy":

            float(

                accuracy_score(

                    y_true,

                    y_pred

                )

            ),


        "balanced_accuracy":

            float(

                balanced_accuracy_score(

                    y_true,

                    y_pred

                )

            ),


        "macro_f1":

            float(

                f1_score(

                    y_true,

                    y_pred,

                    average="macro",

                    zero_division=0

                )

            ),


        "weighted_f1":

            float(

                f1_score(

                    y_true,

                    y_pred,

                    average="weighted",

                    zero_division=0

                )

            ),


        "confusion_matrix":

            confusion_matrix(

                y_true,

                y_pred,

                labels=[

                    0,

                    1,

                    2

                ]

            ).tolist()

    }


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_confidence_interval(

    y_true,

    y_pred,

    metric_function,

    iterations=1000,

    confidence=0.95

):

    rng = np.random.default_rng(

        RANDOM_SEED

    )


    y_true = np.asarray(

        y_true

    )


    y_pred = np.asarray(

        y_pred

    )


    n = len(y_true)


    scores = []


    for _ in range(iterations):

        indices = rng.integers(

            low=0,

            high=n,

            size=n

        )


        sample_true = y_true[

            indices

        ]


        sample_pred = y_pred[

            indices

        ]


        try:

            score = metric_function(

                sample_true,

                sample_pred

            )


            if np.isfinite(score):

                scores.append(

                    score

                )


        except Exception:

            continue


    if len(scores) == 0:

        return {

            "lower": None,

            "upper": None

        }


    alpha = (

        1.0

        -

        confidence

    ) / 2.0


    lower = np.percentile(

        scores,

        100 * alpha

    )


    upper = np.percentile(

        scores,

        100 * (

            1 - alpha

        )

    )


    return {

        "lower":

            float(lower),


        "upper":

            float(upper)

    }


# ============================================================
# BOOTSTRAP METRICS
# ============================================================

def calculate_bootstrap_metrics(

    y_true,

    y_pred

):

    return {

        "accuracy":

            bootstrap_confidence_interval(

                y_true,

                y_pred,

                accuracy_score,

                BOOTSTRAP_ITERATIONS

            ),


        "balanced_accuracy":

            bootstrap_confidence_interval(

                y_true,

                y_pred,

                balanced_accuracy_score,

                BOOTSTRAP_ITERATIONS

            ),


        "macro_f1":

            bootstrap_confidence_interval(

                y_true,

                y_pred,

                lambda a, b:

                    f1_score(

                        a,

                        b,

                        average="macro",

                        zero_division=0

                    ),

                BOOTSTRAP_ITERATIONS

            )

    }


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(

    model_name,

    use_pca=False,

    pca_components=None

):

    steps = [

        (

            "scaler",

            StandardScaler()

        )

    ]


    if use_pca:

        steps.append(

            (

                "pca",

                PCA(

                    n_components=pca_components,

                    random_state=RANDOM_SEED

                )

            )

        )


    if model_name == "logistic_regression":

        classifier = LogisticRegression(

            max_iter=10000,

            class_weight="balanced",

            solver="lbfgs",

            random_state=RANDOM_SEED

        )


    elif model_name == "random_forest":

        classifier = RandomForestClassifier(

            n_estimators=500,

            class_weight="balanced",

            random_state=RANDOM_SEED,

            n_jobs=-1

        )


    else:

        raise ValueError(

            f"Unknown model: "

            f"{model_name}"

        )


    steps.append(

        (

            "classifier",

            classifier

        )

    )


    return Pipeline(

        steps

    )


# ============================================================
# SAFE PCA VALIDATION
# ============================================================

def validate_pca_components(

    X_train,

    use_pca,

    pca_components

):

    if not use_pca:

        return


    maximum_allowed = min(

        X_train.shape[0],

        X_train.shape[1]

    )


    if pca_components >= maximum_allowed:

        raise ValueError(

            f"PCA components ({pca_components}) "

            f"must be less than "

            f"min(train_samples, features) "

            f"({maximum_allowed})"

        )


# ============================================================
# LEAKAGE-FREE CROSS VALIDATION
# ============================================================

def evaluate_cross_validation(

    X,

    y,

    representation,

    model_name,

    use_pca=False,

    pca_components=None,

    folds=5

):

    X_representation = get_representation(

        X,

        representation

    )


    skf = StratifiedKFold(

        n_splits=folds,

        shuffle=True,

        random_state=RANDOM_SEED

    )


    all_true = []

    all_pred = []

    fold_metrics = []


    for fold_number, (

        train_index,

        test_index

    ) in enumerate(

        skf.split(

            X_representation,

            y

        ),

        start=1

    ):


        X_train = X_representation[

            train_index

        ]


        X_test = X_representation[

            test_index

        ]


        y_train = y[

            train_index

        ]


        y_test = y[

            test_index

        ]


        validate_pca_components(

            X_train,

            use_pca,

            pca_components

        )


        model = create_model(

            model_name,

            use_pca,

            pca_components

        )


        model.fit(

            X_train,

            y_train

        )


        predictions = model.predict(

            X_test

        )


        metrics = calculate_metrics(

            y_test,

            predictions

        )


        metrics[

            "fold"

        ] = fold_number


        fold_metrics.append(

            metrics

        )


        all_true.extend(

            y_test.tolist()

        )


        all_pred.extend(

            predictions.tolist()

        )


    overall_metrics = calculate_metrics(

        all_true,

        all_pred

    )


    bootstrap = calculate_bootstrap_metrics(

        all_true,

        all_pred

    )


    return {

        "records":

            int(

                len(y)

            ),


        "representation":

            representation,


        "model":

            model_name,


        "pca_used":

            bool(use_pca),


        "pca_components":

            pca_components,


        "metrics":

            overall_metrics,


        "bootstrap_confidence_intervals":

            bootstrap,


        "fold_results":

            fold_metrics,


        "predictions":

            [

                int(x)

                for x in all_pred

            ]

    }


# ============================================================
# REPEATED CROSS VALIDATION
# ============================================================

def evaluate_repeated_cv(

    X,

    y,

    representation,

    model_name,

    use_pca=False,

    pca_components=None

):

    X_representation = get_representation(

        X,

        representation

    )


    rskf = RepeatedStratifiedKFold(

        n_splits=NUM_FOLDS,

        n_repeats=NUM_REPEATS,

        random_state=RANDOM_SEED

    )


    accuracy_scores = []

    balanced_scores = []

    macro_f1_scores = []


    for train_index, test_index in rskf.split(

        X_representation,

        y

    ):


        X_train = X_representation[

            train_index

        ]


        X_test = X_representation[

            test_index

        ]


        y_train = y[

            train_index

        ]


        y_test = y[

            test_index

        ]


        validate_pca_components(

            X_train,

            use_pca,

            pca_components

        )


        model = create_model(

            model_name,

            use_pca,

            pca_components

        )


        model.fit(

            X_train,

            y_train

        )


        predictions = model.predict(

            X_test

        )


        accuracy_scores.append(

            accuracy_score(

                y_test,

                predictions

            )

        )


        balanced_scores.append(

            balanced_accuracy_score(

                y_test,

                predictions

            )

        )


        macro_f1_scores.append(

            f1_score(

                y_test,

                predictions,

                average="macro",

                zero_division=0

            )

        )


    return {

        "folds":

            NUM_FOLDS,


        "repeats":

            NUM_REPEATS,


        "total_runs":

            NUM_FOLDS

            *

            NUM_REPEATS,


        "accuracy":

            {

                "mean":

                    float(

                        np.mean(

                            accuracy_scores

                        )

                    ),


                "std":

                    float(

                        np.std(

                            accuracy_scores

                        )

                    )

            },


        "balanced_accuracy":

            {

                "mean":

                    float(

                        np.mean(

                            balanced_scores

                        )

                    ),


                "std":

                    float(

                        np.std(

                            balanced_scores

                        )

                    )

            },


        "macro_f1":

            {

                "mean":

                    float(

                        np.mean(

                            macro_f1_scores

                        )

                    ),


                "std":

                    float(

                        np.std(

                            macro_f1_scores

                        )

                    )

            }

    }


# ============================================================
# CROSS-DATASET EVALUATION
# ============================================================

def evaluate_cross_dataset(

    X_train,

    y_train,

    X_test,

    y_test,

    representation,

    model_name,

    use_pca=False,

    pca_components=None

):

    X_train_rep = get_representation(

        X_train,

        representation

    )


    X_test_rep = get_representation(

        X_test,

        representation

    )


    validate_pca_components(

        X_train_rep,

        use_pca,

        pca_components

    )


    model = create_model(

        model_name,

        use_pca,

        pca_components

    )


    model.fit(

        X_train_rep,

        y_train

    )


    predictions = model.predict(

        X_test_rep

    )


    metrics = calculate_metrics(

        y_test,

        predictions

    )


    bootstrap = calculate_bootstrap_metrics(

        y_test,

        predictions

    )


    return {

        "train_records":

            int(

                len(y_train)

            ),


        "test_records":

            int(

                len(y_test)

            ),


        "representation":

            representation,


        "model":

            model_name,


        "pca_used":

            bool(use_pca),


        "pca_components":

            pca_components,


        "metrics":

            metrics,


        "bootstrap_confidence_intervals":

            bootstrap,


        "predictions":

            [

                int(x)

                for x in predictions

            ]

    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(

    title,

    result

):

    print()

    print("-" * 70)

    print(title)

    print("-" * 70)


    metrics = result[

        "metrics"

    ]


    print(

        "Accuracy:",

        round(

            metrics["accuracy"],

            4

        )

    )


    print(

        "Balanced Accuracy:",

        round(

            metrics["balanced_accuracy"],

            4

        )

    )


    print(

        "Macro F1:",

        round(

            metrics["macro_f1"],

            4

        )

    )


    print(

        "Weighted F1:",

        round(

            metrics["weighted_f1"],

            4

        )

    )


# ============================================================
# ADD CSV ROW
# ============================================================

def add_csv_row(

    csv_rows,

    experiment,

    evaluation_name,

    result

):

    metrics = result[

        "metrics"

    ]


    row = {

        "experiment":

            experiment["name"],


        "evaluation":

            evaluation_name,


        "representation":

            experiment["representation"],


        "model":

            experiment["model"],


        "pca":

            experiment["use_pca"],


        "pca_components":

            experiment["pca_components"],


        "accuracy":

            metrics["accuracy"],


        "balanced_accuracy":

            metrics["balanced_accuracy"],


        "macro_f1":

            metrics["macro_f1"],


        "weighted_f1":

            metrics["weighted_f1"]

    }


    if (

        "bootstrap_confidence_intervals"

        in

        result

    ):

        ci = result[

            "bootstrap_confidence_intervals"

        ]


        row[

            "accuracy_ci_lower"

        ] = ci["accuracy"]["lower"]


        row[

            "accuracy_ci_upper"

        ] = ci["accuracy"]["upper"]


        row[

            "balanced_accuracy_ci_lower"

        ] = ci[

            "balanced_accuracy"

        ]["lower"]


        row[

            "balanced_accuracy_ci_upper"

        ] = ci[

            "balanced_accuracy"

        ]["upper"]


        row[

            "macro_f1_ci_lower"

        ] = ci["macro_f1"]["lower"]


        row[

            "macro_f1_ci_upper"

        ] = ci["macro_f1"]["upper"]


    csv_rows.append(

        row

    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print("TRUSTGATE PUBLICATION-QUALITY EVALUATION")

    print("=" * 70)

    print()

    print(

        "Random seed:",

        RANDOM_SEED

    )


    # ========================================================
    # VALIDATE FILES
    # ========================================================

    validate_files()


    # ========================================================
    # LOAD DATA
    # ========================================================

    print()

    print("=" * 70)

    print("LOADING DATA")

    print("=" * 70)


    original_features = load_json(

        ORIGINAL_FEATURE_FILE

    )


    next_features = load_json(

        NEXT_FEATURE_FILE

    )


    original_labels = load_labels(

        ORIGINAL_LABEL_FILE,

        "Original 100"

    )


    next_labels = load_labels(

        NEXT_LABEL_FILE,

        "Next 100"

    )


    # ========================================================
    # ALIGN DATASETS
    # ========================================================

    X_original, y_original, ids_original = (

        align_dataset(

            original_labels,

            original_features,

            "Original"

        )

    )


    X_next, y_next, ids_next = (

        align_dataset(

            next_labels,

            next_features,

            "Next"

        )

    )


    # ========================================================
    # COMBINED DATASET
    # ========================================================

    X_combined = np.vstack(

        [

            X_original,

            X_next

        ]

    )


    y_combined = np.concatenate(

        [

            y_original,

            y_next

        ]

    )


    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print("DATASET SUMMARY")

    print("=" * 70)


    print()

    print(

        "Original records:",

        len(X_original)

    )


    print(

        "Next records:",

        len(X_next)

    )


    print(

        "Combined records:",

        len(X_combined)

    )


    print(

        "Feature dimension:",

        X_combined.shape[1]

    )


    print()

    print(

        "Combined label distribution:"

    )


    for class_id, class_name in enumerate(

        LABEL_NAMES

    ):

        count = int(

            np.sum(

                y_combined == class_id

            )

        )


        print(

            f"{class_name}: {count}"

        )


    # ========================================================
    # EXPERIMENT CONFIGURATIONS
    # ========================================================

    experiments = [

        {

            "name":

                "Signals + Logistic Regression",


            "representation":

                "signals_only",


            "model":

                "logistic_regression",


            "use_pca":

                False,


            "pca_components":

                None

        },


        {

            "name":

                "Hidden + Logistic Regression",


            "representation":

                "hidden_only",


            "model":

                "logistic_regression",


            "use_pca":

                False,


            "pca_components":

                None

        },


        {

            "name":

                "Fusion + Logistic Regression",


            "representation":

                "fusion",


            "model":

                "logistic_regression",


            "use_pca":

                False,


            "pca_components":

                None

        },


        {

            "name":

                "Hidden PCA-20 + Logistic Regression",


            "representation":

                "hidden_only",


            "model":

                "logistic_regression",


            "use_pca":

                True,


            "pca_components":

                20

        },


        {

            "name":

                "Fusion PCA-20 + Logistic Regression",


            "representation":

                "fusion",


            "model":

                "logistic_regression",


            "use_pca":

                True,


            "pca_components":

                20

        },


        {

            "name":

                "Signals + Random Forest",


            "representation":

                "signals_only",


            "model":

                "random_forest",


            "use_pca":

                False,


            "pca_components":

                None

        },


        {

            "name":

                "Hidden + Random Forest",


            "representation":

                "hidden_only",


            "model":

                "random_forest",


            "use_pca":

                False,


            "pca_components":

                None

        },


        {

            "name":

                "Fusion + Random Forest",


            "representation":

                "fusion",


            "model":

                "random_forest",


            "use_pca":

                False,


            "pca_components":

                None

        }

    ]


    # ========================================================
    # RESULTS STRUCTURE
    # ========================================================

    results = {

        "experiment":

            "TrustGate Publication Quality Evaluation",


        "random_seed":

            RANDOM_SEED,


        "methodology": {

            "cross_validation":

                "StratifiedKFold",


            "folds":

                NUM_FOLDS,


            "repeated_cv":

                "RepeatedStratifiedKFold",


            "repeats":

                NUM_REPEATS,


            "bootstrap_iterations":

                BOOTSTRAP_ITERATIONS,


            "feature_dimension":

                int(

                    X_combined.shape[1]

                )

        },


        "datasets": {

            "original_records":

                int(

                    len(y_original)

                ),


            "next_records":

                int(

                    len(y_next)

                ),


            "combined_records":

                int(

                    len(y_combined)

                )

        },


        "within_dataset_cv":

            {},


        "combined_cv":

            {},


        "repeated_cv":

            {},


        "cross_dataset":

            {}

    }


    csv_rows = []


    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================

    for experiment_number, experiment in enumerate(

        experiments,

        start=1

    ):


        name = experiment["name"]


        print()

        print()

        print("=" * 70)

        print(

            f"EXPERIMENT {experiment_number}/"

            f"{len(experiments)}"

        )

        print(name)

        print("=" * 70)


        # ====================================================
        # ORIGINAL CROSS VALIDATION
        # ====================================================

        print()

        print(

            "Running Original Dataset CV..."

        )


        result_original = (

            evaluate_cross_validation(

                X_original,

                y_original,

                experiment["representation"],

                experiment["model"],

                experiment["use_pca"],

                experiment["pca_components"]

            )

        )


        print_result(

            f"ORIGINAL CV: {name}",

            result_original

        )


        results[

            "within_dataset_cv"

        ][

            f"original__{name}"

        ] = result_original


        add_csv_row(

            csv_rows,

            experiment,

            "original_cv",

            result_original

        )


        # ====================================================
        # NEXT CROSS VALIDATION
        # ====================================================

        print()

        print(

            "Running Next Dataset CV..."

        )


        result_next = (

            evaluate_cross_validation(

                X_next,

                y_next,

                experiment["representation"],

                experiment["model"],

                experiment["use_pca"],

                experiment["pca_components"]

            )

        )


        print_result(

            f"NEXT CV: {name}",

            result_next

        )


        results[

            "within_dataset_cv"

        ][

            f"next__{name}"

        ] = result_next


        add_csv_row(

            csv_rows,

            experiment,

            "next_cv",

            result_next

        )


        # ====================================================
        # COMBINED 200 CV
        # ====================================================

        print()

        print(

            "Running Combined 200 CV..."

        )


        result_combined = (

            evaluate_cross_validation(

                X_combined,

                y_combined,

                experiment["representation"],

                experiment["model"],

                experiment["use_pca"],

                experiment["pca_components"]

            )

        )


        print_result(

            f"COMBINED 200 CV: {name}",

            result_combined

        )


        results[

            "combined_cv"

        ][

            f"combined__{name}"

        ] = result_combined


        add_csv_row(

            csv_rows,

            experiment,

            "combined_200_cv",

            result_combined

        )


        # ====================================================
        # REPEATED CV ORIGINAL
        # ====================================================

        print()

        print(

            "Running Repeated CV on Original..."

        )


        repeated_original = (

            evaluate_repeated_cv(

                X_original,

                y_original,

                experiment["representation"],

                experiment["model"],

                experiment["use_pca"],

                experiment["pca_components"]

            )

        )


        results[

            "repeated_cv"

        ][

            f"original__{name}"

        ] = repeated_original


        print()

        print(

            "Repeated CV Original:"

        )


        print(

            repeated_original

        )


        # ====================================================
        # ORIGINAL -> NEXT
        # ====================================================

        print()

        print(

            "Running Original -> Next..."

        )


        original_to_next = (

            evaluate_cross_dataset(

                X_original,

                y_original,

                X_next,

                y_next,

                experiment["representation"],

                experiment["model"],

                experiment["use_pca"],

                experiment["pca_components"]

            )

        )


        print_result(

            f"ORIGINAL -> NEXT: {name}",

            original_to_next

        )


        results[

            "cross_dataset"

        ][

            f"original_to_next__{name}"

        ] = original_to_next


        add_csv_row(

            csv_rows,

            experiment,

            "original_to_next",

            original_to_next

        )


        # ====================================================
        # NEXT -> ORIGINAL
        # ====================================================

        print()

        print(

            "Running Next -> Original..."

        )


        next_to_original = (

            evaluate_cross_dataset(

                X_next,

                y_next,

                X_original,

                y_original,

                experiment["representation"],

                experiment["model"],

                experiment["use_pca"],

                experiment["pca_components"]

            )

        )


        print_result(

            f"NEXT -> ORIGINAL: {name}",

            next_to_original

        )


        results[

            "cross_dataset"

        ][

            f"next_to_original__{name}"

        ] = next_to_original


        add_csv_row(

            csv_rows,

            experiment,

            "next_to_original",

            next_to_original

        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()

    print()

    print("=" * 70)

    print("SAVING RESULTS")

    print("=" * 70)


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


    results_df = pd.DataFrame(

        csv_rows

    )


    results_df.to_csv(

        OUTPUT_CSV,

        index=False

    )


    print()

    print(

        "JSON saved:"

    )

    print(

        OUTPUT_JSON

    )


    print()

    print(

        "CSV saved:"

    )

    print(

        OUTPUT_CSV

    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print("FINAL SUMMARY")

    print("=" * 70)


    print()

    summary_df = (

        results_df[

            [

                "experiment",

                "evaluation",

                "accuracy",

                "balanced_accuracy",

                "macro_f1"

            ]

        ]

        .sort_values(

            [

                "evaluation",

                "macro_f1"

            ],

            ascending=[

                True,

                False

            ]

        )

    )


    print(

        summary_df.to_string(

            index=False

        )

    )


    print()

    print("=" * 70)

    print(

        "TRUSTGATE PUBLICATION-QUALITY "

        "EVALUATION COMPLETE"

    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        print()

        print("=" * 70)

        print("EVALUATION FAILED")

        print("=" * 70)

        print()

        traceback.print_exc()

        print()

        raise