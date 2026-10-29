import json
import os
import pandas as pd


INPUT_FILE = "outputs/trustgate_100_master_901.json"

REFERENCE_FILE = "outputs/reference_agreement_analysis.csv"

OUTPUT_FILE = "outputs/trustgate_100_labeled_v3.json"


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("Loading TrustGate data...")
print("=" * 60)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


reference_df = pd.read_csv(
    REFERENCE_FILE
)


reference_lookup = {
    int(row["id"]): row
    for _, row in reference_df.iterrows()
}


print(
    "Records:",
    len(data)
)


# ============================================================
# SCORE
# ============================================================

def calculate_trust_score(
    confidence,
    entropy,
    self_consistency,
    retrieval,
    verifier,
    semantic_similarity,
    lexical_overlap
):

    # --------------------------------------------------------
    # Normalize entropy.
    #
    # Our observed entropy is roughly 1.1 - 2.1.
    # Lower entropy = better.
    # --------------------------------------------------------

    entropy_score = 1.0 - (
        (entropy - 1.0) / 1.2
    )

    entropy_score = max(
        0.0,
        min(
            1.0,
            entropy_score
        )
    )


    # --------------------------------------------------------
    # Convert verifier from [-1,1] to [0,1]
    # --------------------------------------------------------

    verifier_normalized = (
        verifier + 1.0
    ) / 2.0


    # --------------------------------------------------------
    # Combined evidence score
    # --------------------------------------------------------

    score = (

        0.15 * confidence

        + 0.10 * entropy_score

        + 0.15 * self_consistency

        + 0.25 * retrieval

        + 0.15 * verifier_normalized

        + 0.15 * semantic_similarity

        + 0.05 * lexical_overlap

    )


    # --------------------------------------------------------
    # Explicit contradiction penalty
    # --------------------------------------------------------

    if verifier <= -1.0:

        score -= 0.20


    score = max(
        0.0,
        min(
            1.0,
            score
        )
    )


    return score


# ============================================================
# PROCESS
# ============================================================

results = []


print(
    "\nCalculating continuous trust scores..."
)


for item in data:

    item_id = int(
        item["id"]
    )


    ref = reference_lookup[
        item_id
    ]


    confidence = float(
        item["confidence"]
    )

    entropy = float(
        item["entropy"]
    )

    consistency = float(
        item["self_consistency"]
    )

    retrieval = float(
        item["retrieval_agreement"]
    )

    verifier = float(
        item["verifier_score"]
    )

    semantic = float(
        ref[
            "semantic_similarity"
        ]
    )

    lexical = float(
        ref[
            "lexical_overlap"
        ]
    )


    trust_score = calculate_trust_score(

        confidence,

        entropy,

        consistency,

        retrieval,

        verifier,

        semantic,

        lexical
    )


    # ========================================================
    # DATA-DRIVEN CLASSIFICATION
    # ========================================================

    # Very low score
    if trust_score < 0.50:

        label = "UNRELIABLE"


    # High score
    elif trust_score >= 0.70:

        label = "RELIABLE"


    # Middle
    else:

        label = "UNCERTAIN"


    new_item = dict(
        item
    )


    new_item[
        "reference_semantic_similarity"
    ] = semantic


    new_item[
        "reference_lexical_overlap"
    ] = lexical


    new_item[
        "silver_trust_score"
    ] = trust_score


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

    "RELIABLE": 0,

    "UNCERTAIN": 0,

    "UNRELIABLE": 0
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
    "V3 LABEL DISTRIBUTION"
)

print(
    "=" * 60
)


for label in [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE"
]:

    count = counts[
        label
    ]

    percentage = (
        count
        /
        len(results)
        *
        100
    )


    print(
        f"{label}: "
        f"{count} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

scores = [
    item[
        "silver_trust_score"
    ]
    for item in results
]


print(
    "\n" + "=" * 60
)

print(
    "TRUST SCORE DISTRIBUTION"
)

print(
    "=" * 60
)


print(
    "Minimum:",
    round(
        min(scores),
        4
    )
)

print(
    "25%:",
    round(
        pd.Series(
            scores
        ).quantile(
            0.25
        ),
        4
    )
)

print(
    "Median:",
    round(
        pd.Series(
            scores
        ).median(),
        4
    )
)

print(
    "75%:",
    round(
        pd.Series(
            scores
        ).quantile(
            0.75
        ),
        4
    )
)

print(
    "Maximum:",
    round(
        max(scores),
        4
    )
)


# ============================================================
# SHOW ALL UNRELIABLE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "UNRELIABLE EXAMPLES"
)

print(
    "=" * 60
)


for item in results:

    if item[
        "reliability_label"
    ] == "UNRELIABLE":

        print(
            "\nID:",
            item["id"]
        )

        print(
            "Question:",
            item["question"]
        )

        print(
            "Trust score:",
            round(
                item[
                    "silver_trust_score"
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
    "V3 LABELING COMPLETE!"
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