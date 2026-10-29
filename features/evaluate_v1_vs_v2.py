import json
import os

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_FILE = "outputs/trustgate_standardized_500_labeled.json"

EXPERIENCE_FILE = "outputs/trustgate_v7_1_verified_experiences.json"

OUTPUT_FILE = "outputs/trustgate_v1_vs_v2_evaluation.json"

LABELS = [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE",
]

RANDOM_STATE = 42
N_SPLITS = 5
PCA_COMPONENTS = 20


# ============================================================
# UTILITY
# ============================================================

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("TRUSTGATE V1 vs V2")
print("5-FOLD SELF-TRAINING VALIDATION")
print("=" * 75)

check_file(BASE_FILE)
check_file(EXPERIENCE_FILE)

with open(BASE_FILE, "r", encoding="utf-8") as f:
    base_data = json.load(f)

with open(EXPERIENCE_FILE, "r", encoding="utf-8") as f:
    experiences = json.load(f)

print()
print("Human-labeled examples :", len(base_data))
print("Verified experiences   :", len(experiences))


# ============================================================
# VALIDATE HUMAN DATA
# ============================================================

required_base_fields = [
    "hidden_state",
    "confidence",
    "entropy",
    "label",
]

for i, item in enumerate(base_data):

    for field in required_base_fields:

        if field not in item:
            raise KeyError(
                f"Human example {i} is missing field: {field}"
            )

    if item["label"] not in LABELS:
        raise ValueError(
            f"Invalid human label at index {i}: "
            f"{item['label']}"
        )


# ============================================================
# VALIDATE EXPERIENCE DATA
# ============================================================

required_experience_fields = [
    "hidden_state",
    "confidence",
    "entropy",
    "pseudo_label",
]

for i, item in enumerate(experiences):

    for field in required_experience_fields:

        if field not in item:
            raise KeyError(
                f"Experience {i} is missing field: {field}"
            )

    if item["pseudo_label"] not in LABELS:
        raise ValueError(
            f"Invalid pseudo-label at experience {i}: "
            f"{item['pseudo_label']}"
        )


# ============================================================
# EXTRACT HUMAN FEATURES
# ============================================================

hidden = np.asarray(
    [
        item["hidden_state"]
        for item in base_data
    ],
    dtype=np.float32,
)

confidence = np.asarray(
    [
        float(item["confidence"])
        for item in base_data
    ],
    dtype=np.float32,
)

entropy = np.asarray(
    [
        float(item["entropy"])
        for item in base_data
    ],
    dtype=np.float32,
)

labels = np.asarray(
    [
        item["label"]
        for item in base_data
    ],
    dtype=object,
)


# ============================================================
# EXTRACT VERIFIED EXPERIENCE FEATURES
# ============================================================

experience_hidden = np.asarray(
    [
        item["hidden_state"]
        for item in experiences
    ],
    dtype=np.float32,
)

experience_confidence = np.asarray(
    [
        float(item["confidence"])
        for item in experiences
    ],
    dtype=np.float32,
)

experience_entropy = np.asarray(
    [
        float(item["entropy"])
        for item in experiences
    ],
    dtype=np.float32,
)

experience_labels = np.asarray(
    [
        item["pseudo_label"]
        for item in experiences
    ],
    dtype=object,
)


# ============================================================
# DIMENSION CHECKS
# ============================================================

if len(hidden) == 0:
    raise ValueError("No human examples loaded.")

if hidden.ndim != 2:
    raise ValueError(
        f"Human hidden states must be 2-D. "
        f"Found shape: {hidden.shape}"
    )

if hidden.shape[1] != 896:
    raise ValueError(
        f"Expected 896-D hidden states. "
        f"Found: {hidden.shape[1]}"
    )

if len(experience_hidden) > 0:

    if experience_hidden.ndim != 2:
        raise ValueError(
            "Experience hidden states must be 2-D."
        )

    if experience_hidden.shape[1] != 896:
        raise ValueError(
            "Experience hidden states must be 896-D. "
            f"Found: {experience_hidden.shape[1]}"
        )


# ============================================================
# PRINT CLASS DISTRIBUTIONS
# ============================================================

print()
print("=" * 75)
print("HUMAN LABEL DISTRIBUTION")
print("=" * 75)

for label in LABELS:

    count = int(
        np.sum(labels == label)
    )

    print(
        f"{label:<15}: {count}"
    )


print()
print("=" * 75)
print("VERIFIED EXPERIENCE DISTRIBUTION")
print("=" * 75)

for label in LABELS:

    count = int(
        np.sum(experience_labels == label)
    )

    print(
        f"{label:<15}: {count}"
    )


# ============================================================
# BUILD MODEL
# ============================================================

def train_model(
    train_hidden,
    train_confidence,
    train_entropy,
    train_labels,
    extra_hidden=None,
    extra_confidence=None,
    extra_entropy=None,
    extra_labels=None,
):
    """
    Train one TrustGate model.

    Representation:

        896-D hidden state
              |
        StandardScaler
              |
        PCA(20)
              |
        + confidence
        + entropy
              |
        22-D representation
              |
        StandardScaler
              |
        Balanced LinearSVC
    """

    train_hidden = np.asarray(
        train_hidden,
        dtype=np.float32,
    )

    train_confidence = np.asarray(
        train_confidence,
        dtype=np.float32,
    )

    train_entropy = np.asarray(
        train_entropy,
        dtype=np.float32,
    )

    train_labels = np.asarray(
        train_labels,
        dtype=object,
    )

    # --------------------------------------------------------
    # Add self-training experiences if supplied
    # --------------------------------------------------------

    if extra_hidden is not None:

        if len(extra_hidden) > 0:

            train_hidden = np.vstack(
                [
                    train_hidden,
                    extra_hidden,
                ]
            )

            train_confidence = np.concatenate(
                [
                    train_confidence,
                    extra_confidence,
                ]
            )

            train_entropy = np.concatenate(
                [
                    train_entropy,
                    extra_entropy,
                ]
            )

            train_labels = np.concatenate(
                [
                    train_labels,
                    extra_labels,
                ]
            )

    # --------------------------------------------------------
    # Hidden-state scaling
    # --------------------------------------------------------

    hidden_scaler = StandardScaler()

    hidden_scaled = (
        hidden_scaler.fit_transform(
            train_hidden
        )
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=RANDOM_STATE,
    )

    hidden_pca = (
        pca.fit_transform(
            hidden_scaled
        )
    )

    # --------------------------------------------------------
    # 22-D Trust Representation
    #
    # 20 PCA dimensions
    # + confidence
    # + entropy
    # --------------------------------------------------------

    representation = np.hstack(
        [
            hidden_pca,
            train_confidence.reshape(-1, 1),
            train_entropy.reshape(-1, 1),
        ]
    )

    # --------------------------------------------------------
    # Final representation scaling
    # --------------------------------------------------------

    representation_scaler = StandardScaler()

    representation_scaled = (
        representation_scaler.fit_transform(
            representation
        )
    )

    # --------------------------------------------------------
    # Balanced Linear SVM
    # --------------------------------------------------------

    classifier = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=10000,
    )

    classifier.fit(
        representation_scaled,
        train_labels,
    )

    return {
        "hidden_scaler": hidden_scaler,
        "pca": pca,
        "representation_scaler": representation_scaler,
        "classifier": classifier,
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_model(
    model,
    hidden_data,
    confidence_data,
    entropy_data,
):
    """
    Generate TrustGate predictions.
    """

    hidden_scaled = (
        model["hidden_scaler"].transform(
            hidden_data
        )
    )

    hidden_pca = (
        model["pca"].transform(
            hidden_scaled
        )
    )

    representation = np.hstack(
        [
            hidden_pca,
            confidence_data.reshape(-1, 1),
            entropy_data.reshape(-1, 1),
        ]
    )

    representation_scaled = (
        model["representation_scaler"].transform(
            representation
        )
    )

    predictions = (
        model["classifier"].predict(
            representation_scaled
        )
    )

    return predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    return {

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

print()
print("=" * 75)
print("STARTING 5-FOLD CROSS-VALIDATION")
print("=" * 75)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


v1_predictions = np.empty(
    len(labels),
    dtype=object,
)

v2_predictions = np.empty(
    len(labels),
    dtype=object,
)


fold_results = []


# ============================================================
# FOLD LOOP
# ============================================================

for fold_number, (
    train_indices,
    test_indices,
) in enumerate(
    cv.split(
        hidden,
        labels,
    ),
    start=1,
):

    print()
    print("-" * 75)
    print(
        f"FOLD {fold_number}/{N_SPLITS}"
    )
    print("-" * 75)

    print(
        "Human training examples:",
        len(train_indices),
    )

    print(
        "Human validation examples:",
        len(test_indices),
    )

    print(
        "Self-training experiences:",
        len(experiences),
    )

    # ========================================================
    # V1
    # ========================================================

    print()
    print("Training V1...")

    v1_model = train_model(
        train_hidden=hidden[train_indices],
        train_confidence=confidence[train_indices],
        train_entropy=entropy[train_indices],
        train_labels=labels[train_indices],
    )

    v1_fold_predictions = predict_model(
        model=v1_model,
        hidden_data=hidden[test_indices],
        confidence_data=confidence[test_indices],
        entropy_data=entropy[test_indices],
    )

    v1_predictions[test_indices] = (
        v1_fold_predictions
    )

    v1_fold_metrics = calculate_metrics(
        labels[test_indices],
        v1_fold_predictions,
    )

    # ========================================================
    # V2
    # ========================================================

    print()
    print("Training V2 with verified experiences...")

    v2_model = train_model(
        train_hidden=hidden[train_indices],
        train_confidence=confidence[train_indices],
        train_entropy=entropy[train_indices],
        train_labels=labels[train_indices],
        extra_hidden=experience_hidden,
        extra_confidence=experience_confidence,
        extra_entropy=experience_entropy,
        extra_labels=experience_labels,
    )

    v2_fold_predictions = predict_model(
        model=v2_model,
        hidden_data=hidden[test_indices],
        confidence_data=confidence[test_indices],
        entropy_data=entropy[test_indices],
    )

    v2_predictions[test_indices] = (
        v2_fold_predictions
    )

    v2_fold_metrics = calculate_metrics(
        labels[test_indices],
        v2_fold_predictions,
    )

    # ========================================================
    # FOLD OUTPUT
    # ========================================================

    print()
    print("V1:")
    print(
        f"  Accuracy          : "
        f"{v1_fold_metrics['accuracy']:.4f}"
    )
    print(
        f"  Balanced Accuracy : "
        f"{v1_fold_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"  Macro F1          : "
        f"{v1_fold_metrics['macro_f1']:.4f}"
    )

    print()
    print("V2:")
    print(
        f"  Accuracy          : "
        f"{v2_fold_metrics['accuracy']:.4f}"
    )
    print(
        f"  Balanced Accuracy : "
        f"{v2_fold_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"  Macro F1          : "
        f"{v2_fold_metrics['macro_f1']:.4f}"
    )

    print()
    print("V2 - V1:")
    print(
        f"  Accuracy          : "
        f"{v2_fold_metrics['accuracy'] - v1_fold_metrics['accuracy']:+.4f}"
    )
    print(
        f"  Balanced Accuracy : "
        f"{v2_fold_metrics['balanced_accuracy'] - v1_fold_metrics['balanced_accuracy']:+.4f}"
    )
    print(
        f"  Macro F1          : "
        f"{v2_fold_metrics['macro_f1'] - v1_fold_metrics['macro_f1']:+.4f}"
    )

    fold_results.append(
        {
            "fold": fold_number,
            "v1": v1_fold_metrics,
            "v2": v2_fold_metrics,
            "delta": {
                "accuracy":
                    v2_fold_metrics["accuracy"]
                    - v1_fold_metrics["accuracy"],

                "balanced_accuracy":
                    v2_fold_metrics["balanced_accuracy"]
                    - v1_fold_metrics["balanced_accuracy"],

                "macro_f1":
                    v2_fold_metrics["macro_f1"]
                    - v1_fold_metrics["macro_f1"],

                "weighted_f1":
                    v2_fold_metrics["weighted_f1"]
                    - v1_fold_metrics["weighted_f1"],
            },
        }
    )


# ============================================================
# OVERALL METRICS
# ============================================================

v1_metrics = calculate_metrics(
    labels,
    v1_predictions,
)

v2_metrics = calculate_metrics(
    labels,
    v2_predictions,
)


# ============================================================
# DELTAS
# ============================================================

metric_deltas = {

    "accuracy":
        v2_metrics["accuracy"]
        - v1_metrics["accuracy"],

    "balanced_accuracy":
        v2_metrics["balanced_accuracy"]
        - v1_metrics["balanced_accuracy"],

    "macro_f1":
        v2_metrics["macro_f1"]
        - v1_metrics["macro_f1"],

    "weighted_f1":
        v2_metrics["weighted_f1"]
        - v1_metrics["weighted_f1"],
}


# ============================================================
# PREDICTION CHANGES
# ============================================================

changed_mask = (
    v1_predictions
    != v2_predictions
)

changed_count = int(
    np.sum(changed_mask)
)

change_rate = (
    changed_count / len(labels)
)


# ============================================================
# CLASS-WISE METRICS
# ============================================================

def class_report_dict(
    y_true,
    y_pred,
):

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    output = {}

    for label in LABELS:

        output[label] = {
            "precision":
                float(
                    report[label]["precision"]
                ),

            "recall":
                float(
                    report[label]["recall"]
                ),

            "f1":
                float(
                    report[label]["f1-score"]
                ),

            "support":
                int(
                    report[label]["support"]
                ),
        }

    return output


v1_class_metrics = class_report_dict(
    labels,
    v1_predictions,
)

v2_class_metrics = class_report_dict(
    labels,
    v2_predictions,
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

v1_confusion = confusion_matrix(
    labels,
    v1_predictions,
    labels=LABELS,
)

v2_confusion = confusion_matrix(
    labels,
    v2_predictions,
    labels=LABELS,
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print()
print("=" * 75)
print("FINAL V1 vs V2 RESULTS")
print("=" * 75)

print()
print("V1 — ORIGINAL TRUSTGATE")
print(
    f"Accuracy          : {v1_metrics['accuracy']:.4f}"
)
print(
    f"Balanced Accuracy : {v1_metrics['balanced_accuracy']:.4f}"
)
print(
    f"Macro F1          : {v1_metrics['macro_f1']:.4f}"
)
print(
    f"Weighted F1       : {v1_metrics['weighted_f1']:.4f}"
)


print()
print("V2 — SELF-TRAINED TRUSTGATE")
print(
    f"Accuracy          : {v2_metrics['accuracy']:.4f}"
)
print(
    f"Balanced Accuracy : {v2_metrics['balanced_accuracy']:.4f}"
)
print(
    f"Macro F1          : {v2_metrics['macro_f1']:.4f}"
)
print(
    f"Weighted F1       : {v2_metrics['weighted_f1']:.4f}"
)


print()
print("=" * 75)
print("V2 - V1 DELTAS")
print("=" * 75)

print(
    f"Accuracy          : "
    f"{metric_deltas['accuracy']:+.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{metric_deltas['balanced_accuracy']:+.4f}"
)

print(
    f"Macro F1          : "
    f"{metric_deltas['macro_f1']:+.4f}"
)

print(
    f"Weighted F1       : "
    f"{metric_deltas['weighted_f1']:+.4f}"
)


print()
print("=" * 75)
print("PREDICTION CHANGES")
print("=" * 75)

print(
    f"Changed predictions : "
    f"{changed_count}/{len(labels)}"
)

print(
    f"Change rate         : "
    f"{change_rate:.4f}"
)


# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

print()
print("=" * 75)
print("V1 CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        labels,
        v1_predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
    )
)


print()
print("=" * 75)
print("V2 CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        labels,
        v2_predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print()
print("=" * 75)
print("V1 CONFUSION MATRIX")
print("=" * 75)

print(
    "Rows = actual"
)

print(
    "Columns = predicted"
)

print(
    "Labels:",
    LABELS
)

print(v1_confusion)


print()
print("=" * 75)
print("V2 CONFUSION MATRIX")
print("=" * 75)

print(
    "Rows = actual"
)

print(
    "Columns = predicted"
)

print(
    "Labels:",
    LABELS
)

print(v2_confusion)


# ============================================================
# PREDICTION CHANGE BREAKDOWN
# ============================================================

print()
print("=" * 75)
print("PREDICTION CHANGE BREAKDOWN")
print("=" * 75)

change_pairs = {}

for old, new in zip(
    v1_predictions,
    v2_predictions,
):

    if old != new:

        key = f"{old} -> {new}"

        change_pairs[key] = (
            change_pairs.get(key, 0) + 1
        )


if len(change_pairs) == 0:

    print(
        "No predictions changed."
    )

else:

    for key, count in sorted(
        change_pairs.items()
    ):

        print(
            f"{key:<35}: {count}"
        )


# ============================================================
# FOLD SUMMARY
# ============================================================

print()
print("=" * 75)
print("FOLD-BY-FOLD SUMMARY")
print("=" * 75)

for fold in fold_results:

    print(
        f"\nFold {fold['fold']}"
    )

    print(
        f"  V1 Macro F1 : "
        f"{fold['v1']['macro_f1']:.4f}"
    )

    print(
        f"  V2 Macro F1 : "
        f"{fold['v2']['macro_f1']:.4f}"
    )

    print(
        f"  Delta       : "
        f"{fold['delta']['macro_f1']:+.4f}"
    )


# ============================================================
# INTERPRETATION
# ============================================================

if (
    metric_deltas["macro_f1"] > 0
    and metric_deltas["balanced_accuracy"] > 0
):

    interpretation = (
        "The self-trained TrustGate shows improvement "
        "over the human-only baseline on both macro F1 "
        "and balanced accuracy under the 5-fold evaluation."
    )

elif (
    metric_deltas["macro_f1"] < 0
    and metric_deltas["balanced_accuracy"] < 0
):

    interpretation = (
        "The self-trained TrustGate does not improve "
        "over the human-only baseline under this "
        "5-fold evaluation. The automatically verified "
        "experiences may require stronger quality control."
    )

else:

    interpretation = (
        "The self-trained TrustGate produces mixed results "
        "relative to the human-only baseline. The effect "
        "should be reported using the individual metrics "
        "rather than summarized as a universal improvement."
    )


print()
print("=" * 75)
print("INTERPRETATION")
print("=" * 75)

print(
    interpretation
)


# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

results = {

    "experiment":
        "TrustGate V1 vs V2 self-training validation",

    "random_state":
        RANDOM_STATE,

    "n_splits":
        N_SPLITS,

    "representation":
        {
            "hidden_dimension": 896,
            "pca_components": PCA_COMPONENTS,
            "confidence": True,
            "entropy": True,
            "final_dimension": 22,
        },

    "human_examples":
        len(base_data),

    "verified_experiences":
        len(experiences),

    "human_label_distribution":
        {
            label:
                int(
                    np.sum(
                        labels == label
                    )
                )
            for label in LABELS
        },

    "verified_experience_distribution":
        {
            label:
                int(
                    np.sum(
                        experience_labels == label
                    )
                )
            for label in LABELS
        },

    "v1_metrics":
        v1_metrics,

    "v2_metrics":
        v2_metrics,

    "v2_minus_v1":
        metric_deltas,

    "v1_class_metrics":
        v1_class_metrics,

    "v2_class_metrics":
        v2_class_metrics,

    "v1_confusion_matrix":
        v1_confusion.tolist(),

    "v2_confusion_matrix":
        v2_confusion.tolist(),

    "labels":
        LABELS,

    "changed_predictions":
        changed_count,

    "change_rate":
        float(change_rate),

    "prediction_change_breakdown":
        change_pairs,

    "fold_results":
        fold_results,

    "interpretation":
        interpretation,

    "methodology_note":
        (
            "Five-fold stratified cross-validation was performed "
            "on the 500 human-labeled TrustGate examples. "
            "For V1, each training fold contained only its "
            "human-labeled training examples. For V2, the same "
            "human training fold was augmented with the fixed "
            "set of automatically verified self-training "
            "experiences. The held-out human validation fold "
            "was not used for training."
        ),

    "pseudo_label_note":
        (
            "The verified experiences are automatically generated "
            "pseudo-labeled examples and are not human ground truth."
        ),
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 75)
print("EVALUATION COMPLETE")
print("=" * 75)

print()
print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "Use the V2 - V1 deltas above for the project report."
)

print()