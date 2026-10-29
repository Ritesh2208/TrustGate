import json
import os


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/trustgate_100_master_901.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_100_labeled.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading TrustGate 901-D master dataset...")
print("=" * 60)


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
# RELIABILITY LABEL FUNCTION
# ============================================================

def assign_reliability(
    confidence,
    entropy,
    self_consistency,
    retrieval_agreement,
    verifier_score
):

    # --------------------------------------------------------
    # Strong negative evidence
    # --------------------------------------------------------

    if verifier_score <= -1.0:

        return "UNRELIABLE"


    # --------------------------------------------------------
    # Strong positive evidence
    # --------------------------------------------------------

    if (
        verifier_score >= 1.0
        and retrieval_agreement >= 0.75
        and self_consistency >= 0.85
    ):

        return "RELIABLE"


    # --------------------------------------------------------
    # Strong retrieval + stable generation
    # --------------------------------------------------------

    if (
        retrieval_agreement >= 0.85
        and self_consistency >= 0.90
        and confidence >= 0.50
        and entropy <= 1.70
        and verifier_score >= 0
    ):

        return "RELIABLE"


    # --------------------------------------------------------
    # Very low confidence / high uncertainty
    # --------------------------------------------------------

    if (
        confidence < 0.47
        and entropy > 1.75
        and self_consistency < 0.88
    ):

        return "UNCERTAIN"


    # --------------------------------------------------------
    # Weak retrieval
    # --------------------------------------------------------

    if (
        retrieval_agreement < 0.65
        and self_consistency < 0.88
    ):

        return "UNCERTAIN"


    # --------------------------------------------------------
    # Moderate uncertainty
    # --------------------------------------------------------

    if (
        confidence < 0.50
        and entropy > 1.70
    ):

        return "UNCERTAIN"


    # --------------------------------------------------------
    # Stable answer with reasonable evidence
    # --------------------------------------------------------

    if (
        retrieval_agreement >= 0.75
        and self_consistency >= 0.88
        and verifier_score >= 0
    ):

        return "RELIABLE"


    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return "UNCERTAIN"


# ============================================================
# CREATE LABELS
# ============================================================

results = []


print(
    "\nCreating reliability labels..."
)


for item in data:

    label = assign_reliability(

        confidence=float(
            item[
                "confidence"
            ]
        ),

        entropy=float(
            item[
                "entropy"
            ]
        ),

        self_consistency=float(
            item[
                "self_consistency"
            ]
        ),

        retrieval_agreement=float(
            item[
                "retrieval_agreement"
            ]
        ),

        verifier_score=float(
            item[
                "verifier_score"
            ]
        )
    )


    new_item = dict(
        item
    )


    new_item[
        "reliability_label"
    ] = label


    results.append(
        new_item
    )


# ============================================================
# DISTRIBUTION
# ============================================================

label_counts = {

    "RELIABLE":
        0,

    "UNCERTAIN":
        0,

    "UNRELIABLE":
        0
}


for item in results:

    label = item[
        "reliability_label"
    ]

    label_counts[
        label
    ] += 1


print(
    "\n" + "=" * 60
)

print(
    "RELIABILITY LABEL DISTRIBUTION"
)

print(
    "=" * 60
)


print(
    "Reliable:",
    label_counts[
        "RELIABLE"
    ]
)

print(
    "Uncertain:",
    label_counts[
        "UNCERTAIN"
    ]
)

print(
    "Unreliable:",
    label_counts[
        "UNRELIABLE"
    ]
)


# ============================================================
# QUESTION-LEVEL RESULTS
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
        "Confidence:",
        round(
            item["confidence"],
            4
        )
    )

    print(
        "Entropy:",
        round(
            item["entropy"],
            4
        )
    )

    print(
        "Self-consistency:",
        round(
            item[
                "self_consistency"
            ],
            4
        )
    )

    print(
        "Retrieval agreement:",
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
        "Verifier score:",
        item[
            "verifier_score"
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


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "RELIABILITY LABELING COMPLETE!"
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