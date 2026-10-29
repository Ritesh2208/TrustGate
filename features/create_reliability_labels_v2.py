import json
import os


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/trustgate_100_master_901.json"
)

REFERENCE_ANALYSIS = (
    "outputs/reference_agreement_analysis.csv"
)

OUTPUT_FILE = (
    "outputs/trustgate_100_labeled_v2.json"
)


# ============================================================
# LOAD DATA
# ============================================================

import pandas as pd


print("=" * 60)
print("Loading TrustGate master dataset...")
print("=" * 60)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


analysis_df = pd.read_csv(
    REFERENCE_ANALYSIS
)


print(
    "Records loaded:",
    len(data)
)

print(
    "Reference analysis records:",
    len(analysis_df)
)


# ============================================================
# MERGE REFERENCE SCORES
# ============================================================

reference_lookup = {
    int(row["id"]): row
    for _, row in analysis_df.iterrows()
}


# ============================================================
# LABEL FUNCTION
# ============================================================

def assign_label(
    confidence,
    entropy,
    self_consistency,
    retrieval_agreement,
    verifier_score,
    semantic_similarity,
    lexical_overlap
):

    # --------------------------------------------------------
    # 1. Very strong contradiction
    # --------------------------------------------------------

    if (
        verifier_score <= -1.0
        and semantic_similarity < 0.55
    ):

        return "UNRELIABLE"


    # --------------------------------------------------------
    # 2. Strong reference agreement
    # --------------------------------------------------------

    if (
        semantic_similarity >= 0.75
        and lexical_overlap >= 0.55
        and retrieval_agreement >= 0.75
        and self_consistency >= 0.88
    ):

        return "RELIABLE"


    # --------------------------------------------------------
    # 3. Strong semantic + retrieval support
    # --------------------------------------------------------

    if (
        semantic_similarity >= 0.80
        and retrieval_agreement >= 0.80
        and self_consistency >= 0.85
    ):

        return "RELIABLE"


    # --------------------------------------------------------
    # 4. Very weak reference agreement
    # --------------------------------------------------------

    if (
        semantic_similarity < 0.35
        and lexical_overlap < 0.30
    ):

        return "UNCERTAIN"


    # --------------------------------------------------------
    # 5. Low generation confidence
    # --------------------------------------------------------

    if (
        confidence < 0.47
        and entropy > 1.75
    ):

        return "UNCERTAIN"


    # --------------------------------------------------------
    # 6. Low retrieval + low consistency
    # --------------------------------------------------------

    if (
        retrieval_agreement < 0.65
        and self_consistency < 0.88
    ):

        return "UNCERTAIN"


    # --------------------------------------------------------
    # 7. Strong contradiction
    # --------------------------------------------------------

    if (
        verifier_score <= -1.0
        and retrieval_agreement < 0.75
    ):

        return "UNRELIABLE"


    # --------------------------------------------------------
    # 8. High-quality middle/high agreement
    # --------------------------------------------------------

    if (
        semantic_similarity >= 0.65
        and lexical_overlap >= 0.45
        and retrieval_agreement >= 0.75
        and self_consistency >= 0.88
        and confidence >= 0.50
    ):

        return "RELIABLE"


    # --------------------------------------------------------
    # 9. Otherwise uncertain
    # --------------------------------------------------------

    return "UNCERTAIN"


# ============================================================
# GENERATE LABELS
# ============================================================

results = []


print(
    "\nCreating improved reliability labels..."
)


for item in data:

    item_id = int(
        item["id"]
    )


    if item_id not in reference_lookup:

        raise ValueError(
            f"Missing reference analysis for ID {item_id}"
        )


    ref = reference_lookup[
        item_id
    ]


    semantic_similarity = float(
        ref[
            "semantic_similarity"
        ]
    )


    lexical_overlap = float(
        ref[
            "lexical_overlap"
        ]
    )


    confidence = float(
        item[
            "confidence"
        ]
    )


    entropy = float(
        item[
            "entropy"
        ]
    )


    self_consistency = float(
        item[
            "self_consistency"
        ]
    )


    retrieval_agreement = float(
        item[
            "retrieval_agreement"
        ]
    )


    verifier_score = float(
        item[
            "verifier_score"
        ]
    )


    label = assign_label(

        confidence,

        entropy,

        self_consistency,

        retrieval_agreement,

        verifier_score,

        semantic_similarity,

        lexical_overlap
    )


    new_item = dict(
        item
    )


    new_item[
        "reference_semantic_similarity"
    ] = semantic_similarity


    new_item[
        "reference_lexical_overlap"
    ] = lexical_overlap


    new_item[
        "reliability_label"
    ] = label


    results.append(
        new_item
    )


# ============================================================
# DISTRIBUTION
# ============================================================

counts = {

    "RELIABLE":
        0,

    "UNCERTAIN":
        0,

    "UNRELIABLE":
        0
}


for item in results:

    counts[
        item[
            "reliability_label"
        ]
    ] += 1


print(
    "\n" + "=" * 60
)

print(
    "IMPROVED LABEL DISTRIBUTION"
)

print(
    "=" * 60
)


for label in [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE"
]:

    print(
        f"{label}:",
        counts[label]
    )


# ============================================================
# QUESTION LEVEL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "QUESTION-LEVEL LABELS"
)

print(
    "=" * 60
)


for item in results:

    print(
        "\nID:",
        item["id"]
    )

    print(
        "Question:",
        item["question"]
    )

    print(
        "Reference similarity:",
        round(
            item[
                "reference_semantic_similarity"
            ],
            4
        )
    )

    print(
        "Lexical overlap:",
        round(
            item[
                "reference_lexical_overlap"
            ],
            4
        )
    )

    print(
        "Retrieval:",
        round(
            item[
                "retrieval_agreement"
            ],
            4
        )
    )

    print(
        "Verifier:",
        item[
            "verifier_label"
        ]
    )

    print(
        "Reliability:",
        item[
            "reliability_label"
        ]
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )


print(
    "\n" + "=" * 60
)

print(
    "IMPROVED RELIABILITY LABELING COMPLETE!"
)

print(
    "=" * 60
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)