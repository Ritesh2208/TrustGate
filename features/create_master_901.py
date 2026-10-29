import json
import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = (
    "outputs/truthfulqa_100_combined_features.json"
)

VERIFIER_FILE = (
    "outputs/truthfulqa_100_verified.json"
)

RETRIEVAL_FILE = (
    "outputs/truthfulqa_100_retrieval_v3.json"
)

OUTPUT_JSON = (
    "outputs/trustgate_100_master_901.json"
)

OUTPUT_CSV = (
    "outputs/trustgate_100_master_summary.csv"
)


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 60)
print("LOADING TRUSTGATE DATA")
print("=" * 60)


with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_data = json.load(f)


with open(
    VERIFIER_FILE,
    "r",
    encoding="utf-8"
) as f:

    verifier_data = json.load(f)


with open(
    RETRIEVAL_FILE,
    "r",
    encoding="utf-8"
) as f:

    retrieval_data = json.load(f)


print(
    "Feature records:",
    len(feature_data)
)

print(
    "Verifier records:",
    len(verifier_data)
)

print(
    "Retrieval records:",
    len(retrieval_data)
)


# ============================================================
# CREATE LOOKUP TABLES
# ============================================================

verifier_lookup = {
    item["id"]: item
    for item in verifier_data
}


retrieval_lookup = {
    item["id"]: item
    for item in retrieval_data
}


# ============================================================
# BUILD MASTER DATASET
# ============================================================

master_records = []


print(
    "\n" + "=" * 60
)

print(
    "BUILDING 901-D FEATURE VECTORS"
)

print(
    "=" * 60
)


for item in feature_data:

    item_id = item["id"]


    if item_id not in verifier_lookup:

        print(
            "WARNING: Missing verifier:",
            item_id
        )

        continue


    if item_id not in retrieval_lookup:

        print(
            "WARNING: Missing retrieval:",
            item_id
        )

        continue


    verifier = verifier_lookup[
        item_id
    ]

    retrieval = retrieval_lookup[
        item_id
    ]


    # --------------------------------------------------------
    # Hidden state
    # --------------------------------------------------------

    hidden_state = item.get(
        "hidden_state",
        []
    )


    hidden_state = [
        float(x)
        for x in hidden_state
    ]


    if len(hidden_state) != 896:

        raise ValueError(
            f"ID {item_id}: "
            f"hidden state has "
            f"{len(hidden_state)} dimensions "
            f"instead of 896."
        )


    # --------------------------------------------------------
    # Scalar features
    # --------------------------------------------------------

    confidence = float(
        item.get(
            "confidence",
            0.0
        )
    )


    entropy = float(
        item.get(
            "entropy",
            0.0
        )
    )


    self_consistency = float(
        item.get(
            "self_consistency",
            0.0
        )
    )


    retrieval_agreement = float(
        retrieval.get(
            "retrieval_agreement",
            0.0
        )
    )


    verifier_score = float(
        verifier.get(
            "verifier_score",
            0.0
        )
    )


    # --------------------------------------------------------
    # 901-D vector
    # --------------------------------------------------------

    feature_vector = (
        hidden_state
        + [
            confidence,
            entropy,
            self_consistency,
            retrieval_agreement,
            verifier_score
        ]
    )


    if len(feature_vector) != 901:

        raise ValueError(
            f"ID {item_id}: "
            f"feature dimension is "
            f"{len(feature_vector)}, "
            f"expected 901."
        )


    # --------------------------------------------------------
    # Create record
    # --------------------------------------------------------

    record = {

        "id":
            item_id,

        "question":
            item.get(
                "question",
                ""
            ),

        "answer":
            item.get(
                "answer",
                ""
            ),

        "reference_answer":
            item.get(
                "reference_answer",
                ""
            ),

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
            verifier.get(
                "verifier_label",
                "UNKNOWN"
            ),

        "hidden_state":
            hidden_state,

        "feature_vector":
            feature_vector,

        "feature_dimension":
            901
    }


    master_records.append(
        record
    )


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "VALIDATING MASTER DATASET"
)

print(
    "=" * 60
)


print(
    "Master records:",
    len(master_records)
)


dimensions = [
    len(
        item["feature_vector"]
    )
    for item in master_records
]


print(
    "Feature dimensions:",
    sorted(
        set(dimensions)
    )
)


if not all(
    dimension == 901
    for dimension in dimensions
):

    raise ValueError(
        "Feature dimension validation failed."
    )


# ------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------

for item in master_records:

    vector = np.array(
        item["feature_vector"],
        dtype=np.float32
    )


    if np.isnan(
        vector
    ).any():

        raise ValueError(
            f"NaN found in ID {item['id']}"
        )


    if np.isinf(
        vector
    ).any():

        raise ValueError(
            f"Infinity found in ID {item['id']}"
        )


print(
    "No NaN values found."
)

print(
    "No infinite values found."
)


# ============================================================
# SUMMARY DATAFRAME
# ============================================================

summary_rows = []


for item in master_records:

    summary_rows.append(
        {
            "id":
                item["id"],

            "confidence":
                item["confidence"],

            "entropy":
                item["entropy"],

            "self_consistency":
                item[
                    "self_consistency"
                ],

            "retrieval_agreement":
                item[
                    "retrieval_agreement"
                ],

            "verifier_score":
                item[
                    "verifier_score"
                ],

            "verifier_label":
                item[
                    "verifier_label"
                ],

            "hidden_state_dimension":
                len(
                    item[
                        "hidden_state"
                    ]
                ),

            "feature_dimension":
                item[
                    "feature_dimension"
                ]
        }
    )


df = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "SCALAR FEATURE SUMMARY"
)

print(
    "=" * 60
)


print(
    df.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 60
)

print(
    "VERIFIER DISTRIBUTION"
)

print(
    "=" * 60
)


print(
    df[
        "verifier_label"
    ].value_counts()
)


# ============================================================
# STATISTICS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FEATURE STATISTICS"
)

print(
    "=" * 60
)


for column in [
    "confidence",
    "entropy",
    "self_consistency",
    "retrieval_agreement",
    "verifier_score"
]:

    print(
        f"\n{column}"
    )

    print(
        "  min:",
        round(
            df[column].min(),
            4
        )
    )

    print(
        "  max:",
        round(
            df[column].max(),
            4
        )
    )

    print(
        "  mean:",
        round(
            df[column].mean(),
            4
        )
    )


# ============================================================
# SAVE MASTER JSON
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        master_records,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SAVE CSV SUMMARY
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRUSTGATE 901-D MASTER DATASET COMPLETE!"
)

print(
    "=" * 60
)

print(
    "Records:",
    len(master_records)
)

print(
    "Feature dimension:",
    901
)

print(
    "\nJSON:"
)

print(
    OUTPUT_JSON
)

print(
    "\nCSV:"
)

print(
    OUTPUT_CSV
)