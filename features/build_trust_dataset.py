import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/truthfulqa_with_consistency_v2.json"
)

OUTPUT_FILE = (
    "outputs/trust_features.json"
)

CSV_OUTPUT_FILE = (
    "outputs/trust_features_summary.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading TrustGate feature data...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "Records loaded:",
    len(data)
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:

            return default

        return float(value)

    except:

        return default


def normalize_self_feedback(text):

    text = str(text).lower()

    if "incorrect" in text:

        return 0

    if "correct" in text:

        return 1

    if "uncertain" in text:

        return None

    return None


# ============================================================
# BUILD RAW FEATURE TABLE
# ============================================================

records = []


for item in data:

    question = item.get(
        "question",
        ""
    )

    answer = item.get(
        "answer",
        ""
    )

    reference_answer = item.get(
        "reference_answer",
        ""
    )


    # --------------------------------------------------------
    # Scalar reliability features
    # --------------------------------------------------------

    confidence = safe_float(
        item.get(
            "confidence"
        )
    )

    entropy = safe_float(
        item.get(
            "entropy"
        )
    )

    self_consistency = safe_float(
        item.get(
            "self_consistency_v2"
        )
    )

    retrieval_agreement = safe_float(
        item.get(
            "retrieval_agreement"
        )
    )

    verifier_score = safe_float(
        item.get(
            "verifier_score"
        )
    )


    # --------------------------------------------------------
    # Hidden state
    # --------------------------------------------------------

    hidden_state = item.get(
        "hidden_state",
        []
    )


    hidden_state = [
        safe_float(x)
        for x in hidden_state
    ]


    # --------------------------------------------------------
    # Self-feedback
    # --------------------------------------------------------

    self_feedback = item.get(
        "self_feedback",
        ""
    )


    self_feedback_label = (
        normalize_self_feedback(
            self_feedback
        )
    )


    # --------------------------------------------------------
    # Verifier label
    # --------------------------------------------------------

    verifier_label = item.get(
        "verifier_label",
        "UNKNOWN"
    )


    # --------------------------------------------------------
    # Build record
    # --------------------------------------------------------

    record = {

        "id": item.get(
            "id"
        ),

        "question": question,

        "answer": answer,

        "reference_answer":
            reference_answer,

        "confidence":
            confidence,

        "entropy":
            entropy,

        "self_consistency":
            self_consistency,

        "retrieval_agreement":
            retrieval_agreement,

        "verifier_score":
            verifier_score,

        "verifier_label":
            verifier_label,

        "self_feedback_label":
            self_feedback_label,

        "hidden_state":
            hidden_state
    }


    records.append(
        record
    )


# ============================================================
# CHECK HIDDEN-STATE DIMENSION
# ============================================================

hidden_dimensions = [
    len(x["hidden_state"])
    for x in records
]


print(
    "\nHidden-state dimensions:"
)

print(
    hidden_dimensions
)


if len(
    set(hidden_dimensions)
) != 1:

    raise ValueError(
        "Hidden-state dimensions are inconsistent."
    )


hidden_dimension = (
    hidden_dimensions[0]
)


print(
    "Hidden-state dimension:",
    hidden_dimension
)


# ============================================================
# CREATE NUMERICAL MATRIX
# ============================================================

scalar_features = []

for record in records:

    scalar_features.append([

        record[
            "confidence"
        ],

        record[
            "entropy"
        ],

        record[
            "self_consistency"
        ],

        record[
            "retrieval_agreement"
        ],

        record[
            "verifier_score"
        ]
    ])


scalar_features = np.array(
    scalar_features,
    dtype=np.float32
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

column_means = np.nanmean(
    scalar_features,
    axis=0
)


for column in range(
    scalar_features.shape[1]
):

    mask = np.isnan(
        scalar_features[:, column]
    )

    scalar_features[
        mask,
        column
    ] = column_means[
        column
    ]


# ============================================================
# STANDARDIZE SCALAR FEATURES
# ============================================================

scaler = StandardScaler()

scaled_scalar_features = (
    scaler.fit_transform(
        scalar_features
    )
)


# ============================================================
# BUILD FINAL FEATURE VECTOR
# ============================================================

for index, record in enumerate(
    records
):

    hidden = np.array(
        record[
            "hidden_state"
        ],
        dtype=np.float32
    )


    scalar = np.array(
        scaled_scalar_features[
            index
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Combined feature vector
    # --------------------------------------------------------

    combined = np.concatenate([
        scalar,
        hidden
    ])


    record[
        "scaled_confidence"
    ] = float(
        scalar[0]
    )

    record[
        "scaled_entropy"
    ] = float(
        scalar[1]
    )

    record[
        "scaled_self_consistency"
    ] = float(
        scalar[2]
    )

    record[
        "scaled_retrieval_agreement"
    ] = float(
        scalar[3]
    )

    record[
        "scaled_verifier_score"
    ] = float(
        scalar[4]
    )


    record[
        "combined_feature"
    ] = combined.tolist()


    record[
        "feature_dimension"
    ] = len(
        combined
    )


# ============================================================
# CREATE SUMMARY DATAFRAME
# ============================================================

summary_rows = []


for record in records:

    summary_rows.append({

        "id":
            record["id"],

        "confidence":
            record["confidence"],

        "entropy":
            record["entropy"],

        "self_consistency":
            record["self_consistency"],

        "retrieval_agreement":
            record["retrieval_agreement"],

        "verifier_score":
            record["verifier_score"],

        "verifier_label":
            record["verifier_label"],

        "self_feedback_label":
            record["self_feedback_label"],

        "hidden_state_dimension":
            len(
                record[
                    "hidden_state"
                ]
            ),

        "combined_feature_dimension":
            record[
                "feature_dimension"
            ]
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FEATURE SUMMARY"
)

print(
    "=" * 60
)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE MASTER JSON
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        records,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_df.to_csv(
    CSV_OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MASTER TRUST DATASET CREATED!"
)

print(
    "\nJSON:"
)

print(
    OUTPUT_FILE
)

print(
    "\nCSV:"
)

print(
    CSV_OUTPUT_FILE
)

print(
    "\nFinal feature dimension:"
)

print(
    5 + hidden_dimension
)