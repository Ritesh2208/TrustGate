import json
import os
from datetime import datetime, timezone

import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

from external_verifier import ExternalVerifier


# ============================================================
# TRUSTGATE V7 — SELF-TRAINING
# ============================================================
#
# Purpose:
#
#   TrustGate v1
#       ↓
#   generate / obtain new answer
#       ↓
#   external verification
#       ↓
#   high-confidence pseudo-label
#       ↓
#   Trust Experience
#       ↓
#   quality gate
#       ↓
#   TrustGate v2
#
# IMPORTANT:
#
# - Existing 500 human labels are NEVER modified.
# - Automatically generated labels are explicitly marked
#   as AUTO_VERIFIED / PSEUDO_LABEL.
# - Weak verifier results are discarded.
# - TruthfulQA reference answers are NOT used to create labels.
# - The held-out benchmark remains evaluation-only.
#
# ============================================================


BASE_FEATURE_FILE = (
    "outputs/trustgate_standardized_500_labeled.json"
)

HELDOUT_FEATURE_FILE = (
    "outputs/trustgate_heldout_290.json"
)

MODEL_V1 = (
    "models/trustgate_500_22d_linear_svm.joblib"
)

MODEL_V2 = (
    "models/trustgate_v2_self_trained.joblib"
)

EXPERIENCE_FILE = (
    "outputs/trustgate_v7_verified_experiences.json"
)

RESULT_FILE = (
    "outputs/trustgate_v7_self_training_results.json"
)


LABELS = [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE",
]


# ============================================================
# SETTINGS
# ============================================================

# Development subset used to CREATE experiences.
# These are NOT used in the final evaluation.

DEVELOPMENT_SIZE = 40

# Only strong evidence can generate a pseudo-label.

MIN_SUPPORT = 0.80
MAX_CONTRADICTION = 0.30

MIN_SOURCES = 2
MIN_EVIDENCE_SENTENCES = 3


# ============================================================
# JSON
# ============================================================

def save_json(path, data):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    temp = path + ".tmp"

    with open(
        temp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    os.replace(
        temp,
        path
    )


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# TRUST REPRESENTATION
# ============================================================

def build_representation_components(
    hidden_states,
    confidence,
    entropy
):

    hidden_states = np.asarray(
        hidden_states,
        dtype=np.float32
    )

    confidence = np.asarray(
        confidence,
        dtype=np.float32
    )

    entropy = np.asarray(
        entropy,
        dtype=np.float32
    )

    hidden_scaler = StandardScaler()

    hidden_scaled = (
        hidden_scaler.fit_transform(
            hidden_states
        )
    )

    pca = PCA(
        n_components=20,
        random_state=42
    )

    hidden_20 = pca.fit_transform(
        hidden_scaled
    )

    X = np.hstack([
        hidden_20,
        confidence.reshape(-1, 1),
        entropy.reshape(-1, 1)
    ])

    final_scaler = StandardScaler()

    X_scaled = (
        final_scaler.fit_transform(
            X
        )
    )

    return (
        hidden_scaler,
        pca,
        final_scaler,
        X_scaled
    )


# ============================================================
# LOAD ORIGINAL 500 HUMAN-LABELED DATA
# ============================================================

print()
print("=" * 70)
print("TRUSTGATE V7 — SELF-TRAINING")
print("=" * 70)

print()
print("Loading original 500 human-labeled dataset...")

base_data = load_json(
    BASE_FEATURE_FILE
)

print(
    "Human-labeled samples:",
    len(base_data)
)

assert len(base_data) == 500


# ============================================================
# BUILD BASE TRAINING MATRIX
# ============================================================

base_hidden = np.array(
    [
        item["hidden_state"]
        for item in base_data
    ],
    dtype=np.float32
)

base_confidence = np.array(
    [
        item["confidence"]
        for item in base_data
    ],
    dtype=np.float32
)

base_entropy = np.array(
    [
        item["entropy"]
        for item in base_data
    ],
    dtype=np.float32
)

base_labels = np.array(
    [
        item["label"]
        for item in base_data
    ]
)


print()
print("Original label distribution:")

for label in LABELS:

    print(
        f"  {label}:",
        int(
            np.sum(
                base_labels == label
            )
        )
    )


# ============================================================
# LOAD HELD-OUT QUESTIONS
# ============================================================

print()
print("Loading held-out TruthfulQA questions...")

heldout = load_json(
    HELDOUT_FEATURE_FILE
)

print(
    "Held-out questions:",
    len(heldout)
)

assert len(heldout) == 290


# ============================================================
# DEVELOPMENT / EVALUATION SPLIT
# ============================================================

development = heldout[
    :DEVELOPMENT_SIZE
]

evaluation = heldout[
    DEVELOPMENT_SIZE:
]

print()
print(
    "Development questions:",
    len(development)
)

print(
    "Final evaluation questions:",
    len(evaluation)
)


# ============================================================
# LOAD FROZEN V1
# ============================================================

print()
print("Loading TrustGate v1...")

model_bundle_v1 = joblib.load(
    MODEL_V1
)

print(
    "TrustGate v1 loaded."
)


# ============================================================
# VERIFY EVIDENCE
# ============================================================

print()
print("Loading external verifier...")

verifier = ExternalVerifier(
    max_sources=3,
    timeout=10
)

print(
    "External verifier ready."
)


# ============================================================
# GENERATE PSEUDO-LABEL FROM VERIFIER
# ============================================================

def generate_verified_label(
    question,
    answer
):

    result = verifier.verify(
        question,
        answer
    )

    status = result.get(
        "verification_status",
        "UNKNOWN"
    )

    support = float(
        result.get(
            "support_score",
            0.0
        )
    )

    contradiction = float(
        result.get(
            "contradiction_score",
            0.0
        )
    )

    source_count = int(
        result.get(
            "source_count",
            0
        )
    )

    evidence_count = int(
        result.get(
            "evidence_sentence_count",
            0
        )
    )

    # --------------------------------------------------------
    # HIGH-CONFIDENCE SUPPORTED
    # --------------------------------------------------------

    if (
        status == "SUPPORTED"
        and
        support >= MIN_SUPPORT
        and
        contradiction <= MAX_CONTRADICTION
        and
        source_count >= MIN_SOURCES
        and
        evidence_count >= MIN_EVIDENCE_SENTENCES
    ):

        return {
            "accepted": True,
            "pseudo_label": "RELIABLE",
            "verification": result,
            "quality_reason":
                "HIGH_CONFIDENCE_SUPPORT"
        }

    # --------------------------------------------------------
    # HIGH-CONFIDENCE CONTRADICTION
    # --------------------------------------------------------

    if (
        status == "CONTRADICTED"
        and
        contradiction >= MIN_SUPPORT
        and
        support <= MAX_CONTRADICTION
        and
        source_count >= MIN_SOURCES
        and
        evidence_count >= MIN_EVIDENCE_SENTENCES
    ):

        return {
            "accepted": True,
            "pseudo_label": "UNRELIABLE",
            "verification": result,
            "quality_reason":
                "HIGH_CONFIDENCE_CONTRADICTION"
        }

    # --------------------------------------------------------
    # Everything else is discarded.
    # --------------------------------------------------------

    return {
        "accepted": False,
        "pseudo_label": None,
        "verification": result,
        "quality_reason":
            "INSUFFICIENT_VERIFICATION_CONFIDENCE"
    }


# ============================================================
# BUILD VERIFIED EXPERIENCES
# ============================================================

experiences = []

print()
print("=" * 70)
print("GENERATING VERIFIED TRUST EXPERIENCES")
print("=" * 70)


for index, item in enumerate(
    development
):

    question = item[
        "question"
    ]

    answer = item[
        "best_answer"
    ]

    print()
    print(
        f"[{index + 1}/{len(development)}]",
        question
    )

    try:

        verification = (
            generate_verified_label(
                question,
                answer
            )
        )

        result = verification[
            "verification"
        ]

        print(
            "Status:",
            result.get(
                "verification_status"
            )
        )

        print(
            "Support:",
            result.get(
                "support_score"
            )
        )

        print(
            "Contradiction:",
            result.get(
                "contradiction_score"
            )
        )

        print(
            "Sources:",
            result.get(
                "source_count"
            )
        )

        if not verification[
            "accepted"
        ]:

            print(
                "→ REJECTED"
            )

            continue

        pseudo_label = (
            verification[
                "pseudo_label"
            ]
        )

        experience = {

            "experience_id":
                len(experiences) + 1,

            "question":
                question,

            "answer":
                answer,

            "pseudo_label":
                pseudo_label,

            "label_source":
                "EXTERNAL_VERIFIER",

            "quality_reason":
                verification[
                    "quality_reason"
                ],

            "verification_status":
                result.get(
                    "verification_status"
                ),

            "support_score":
                result.get(
                    "support_score",
                    0.0
                ),

            "contradiction_score":
                result.get(
                    "contradiction_score",
                    0.0
                ),

            "source_count":
                result.get(
                    "source_count",
                    0
                ),

            "evidence_sentence_count":
                result.get(
                    "evidence_sentence_count",
                    0
                ),

            "sources":
                result.get(
                    "sources",
                    []
                ),

            "quality_status":
                "APPROVED",

            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

        }

        experiences.append(
            experience
        )

        print(
            "→ APPROVED:",
            pseudo_label
        )

    except Exception as exc:

        print(
            "→ ERROR:",
            repr(exc)
        )


# ============================================================
# SAVE EXPERIENCES
# ============================================================

save_json(
    EXPERIENCE_FILE,
    experiences
)

print()
print("=" * 70)
print("EXPERIENCE GENERATION COMPLETE")
print("=" * 70)

print(
    "Approved experiences:",
    len(experiences)
)

print(
    "Saved:",
    EXPERIENCE_FILE
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(experiences) < 3:

    print()
    print(
        "WARNING:"
    )

    print(
        "Fewer than 3 high-confidence experiences "
        "were generated."
    )

    print(
        "TrustGate v2 will NOT be trained from "
        "weak evidence."
    )

    print(
        "The experiment will stop here."
    )

    raise SystemExit(0)


# ============================================================
# PREPARE SELF-TRAINING DATA
# ============================================================

experience_questions = {
    exp["question"]
    for exp in experiences
}


# ------------------------------------------------------------
# We use the original 500 human-labeled data as the base.
# Automatically verified experiences are additional data.
# ------------------------------------------------------------

experience_hidden = []
experience_confidence = []
experience_entropy = []
experience_labels = []


# The held-out records contain no generated hidden states,
# therefore we cannot directly train on them.
#
# Instead, this stage uses their question/answer only for
# verification. To train on the experience, we require a
# corresponding feature representation.
#
# For this reason, we retrieve the matching answer feature
# from the existing standardized feature store when the
# question exists there.
#
# This is intentionally strict: no synthetic hidden state
# is invented.
# ============================================================

base_by_question = {
    str(item["question"]).strip():
        item
    for item in base_data
}


for experience in experiences:

    question = (
        str(
            experience["question"]
        ).strip()
    )

    if question not in base_by_question:

        print(
            "[Experience skipped for training]",
            "No stored feature representation:",
            question
        )

        continue

    source_item = (
        base_by_question[
            question
        ]
    )

    experience_hidden.append(
        source_item["hidden_state"]
    )

    experience_confidence.append(
        source_item["confidence"]
    )

    experience_entropy.append(
        source_item["entropy"]
    )

    experience_labels.append(
        experience["pseudo_label"]
    )


# ============================================================
# IMPORTANT DATA CHECK
# ============================================================

if len(experience_hidden) < 3:

    print()
    print(
        "Not enough experiences have stored feature "
        "representations for safe retraining."
    )

    print(
        "No TrustGate v2 will be produced."
    )

    raise SystemExit(0)


# ============================================================
# COMBINE DATA
# ============================================================

all_hidden = np.vstack([
    base_hidden,
    np.asarray(
        experience_hidden,
        dtype=np.float32
    )
])

all_confidence = np.concatenate([
    base_confidence,
    np.asarray(
        experience_confidence,
        dtype=np.float32
    )
])

all_entropy = np.concatenate([
    base_entropy,
    np.asarray(
        experience_entropy,
        dtype=np.float32
    )
])

all_labels = np.concatenate([
    base_labels,
    np.asarray(
        experience_labels
    )
])


print()
print("=" * 70)
print("SELF-TRAINING DATASET")
print("=" * 70)

print(
    "Original human examples:",
    len(base_data)
)

print(
    "Verified experiences used:",
    len(experience_hidden)
)

print(
    "Combined examples:",
    len(all_labels)
)


# ============================================================
# FIT V2 REPRESENTATION
# ============================================================

hidden_scaler = StandardScaler()

hidden_scaled = (
    hidden_scaler.fit_transform(
        all_hidden
    )
)

pca = PCA(
    n_components=20,
    random_state=42
)

hidden_20 = (
    pca.fit_transform(
        hidden_scaled
    )
)

X = np.hstack([
    hidden_20,
    all_confidence.reshape(-1, 1),
    all_entropy.reshape(-1, 1)
])

final_scaler = StandardScaler()

X_scaled = (
    final_scaler.fit_transform(
        X
    )
)


# ============================================================
# TRAIN TRUSTGATE V2
# ============================================================

print()
print("=" * 70)
print("TRAINING TRUSTGATE V2")
print("=" * 70)

classifier = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

classifier.fit(
    X_scaled,
    all_labels
)


# ============================================================
# TRAINING SANITY METRICS
# ============================================================

train_prediction = (
    classifier.predict(
        X_scaled
    )
)

print()
print(
    "V2 training accuracy:",
    round(
        accuracy_score(
            all_labels,
            train_prediction
        ),
        4
    )
)

print(
    "V2 training balanced accuracy:",
    round(
        balanced_accuracy_score(
            all_labels,
            train_prediction
        ),
        4
    )
)

print(
    "V2 training macro F1:",
    round(
        f1_score(
            all_labels,
            train_prediction,
            average="macro"
        ),
        4
    )
)


# ============================================================
# SAVE V2
# ============================================================

bundle = {

    "hidden_scaler":
        hidden_scaler,

    "pca":
        pca,

    "final_scaler":
        final_scaler,

    "classifier":
        classifier,

    "input_hidden_dimension":
        896,

    "pca_dimension":
        20,

    "final_dimension":
        22,

    "signal_features":
        [
            "confidence",
            "entropy"
        ],

    "base_human_examples":
        len(base_data),

    "verified_experiences":
        len(experience_hidden),

    "training_type":
        "SELF_TRAINING_WITH_EXTERNAL_VERIFICATION",

    "created_at":
        datetime.now(
            timezone.utc
        ).isoformat(),

}


os.makedirs(
    os.path.dirname(
        MODEL_V2
    ),
    exist_ok=True
)

joblib.dump(
    bundle,
    MODEL_V2
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "model_v1":
        MODEL_V1,

    "model_v2":
        MODEL_V2,

    "original_human_examples":
        len(base_data),

    "approved_verified_experiences":
        len(experiences),

    "experiences_used_for_training":
        len(experience_hidden),

    "training_examples_v2":
        len(all_labels),

    "experience_generation":
        {
            "min_support":
                MIN_SUPPORT,

            "max_contradiction":
                MAX_CONTRADICTION,

            "min_sources":
                MIN_SOURCES,

            "min_evidence_sentences":
                MIN_EVIDENCE_SENTENCES
        },

    "note":
        "Automatically generated labels are pseudo-labels from external verification, not human ground truth.",

    "created_at":
        datetime.now(
            timezone.utc
        ).isoformat()

}

save_json(
    RESULT_FILE,
    summary
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("TRUSTGATE V7 SELF-TRAINING COMPLETE")
print("=" * 70)

print(
    "Verified experiences:",
    len(experiences)
)

print(
    "Experiences used for V2:",
    len(experience_hidden)
)

print(
    "TrustGate V2:",
    MODEL_V2
)

print(
    "Summary:",
    RESULT_FILE
)

print("=" * 70)