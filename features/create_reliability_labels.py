import json
import re


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/trust_features.json"
)

OUTPUT_FILE = (
    "outputs/trust_features_labeled.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading TrustGate feature dataset...")

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
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    return text.strip()


# ============================================================
# REFERENCE ANSWER OVERLAP
# ============================================================

def calculate_reference_overlap(
    answer,
    reference
):

    answer_words = set(
        normalize_text(
            answer
        ).split()
    )

    reference_words = set(
        normalize_text(
            reference
        ).split()
    )

    if not reference_words:

        return 0.0

    overlap = (
        len(
            answer_words
            & reference_words
        )
        /
        len(reference_words)
    )

    return float(
        overlap
    )


# ============================================================
# CREATE RELIABILITY LABEL
# ============================================================

def create_label(item):

    answer = item.get(
        "answer",
        ""
    )

    reference = item.get(
        "reference_answer",
        ""
    )

    verifier_label = str(
        item.get(
            "verifier_label",
            "UNKNOWN"
        )
    ).upper()

    verifier_score = float(
        item.get(
            "verifier_score",
            0.0
        )
    )


    # --------------------------------------------------------
    # Calculate reference overlap
    # --------------------------------------------------------

    overlap = calculate_reference_overlap(
        answer,
        reference
    )


    # --------------------------------------------------------
    # RULE 1:
    # Strong external contradiction
    #
    # This is our strongest negative signal.
    # --------------------------------------------------------

    if (
        verifier_label == "CONTRADICTED"
        and verifier_score <= -0.5
    ):

        return 0


    # --------------------------------------------------------
    # RULE 2:
    # Strong external support
    # --------------------------------------------------------

    if (
        verifier_label == "SUPPORTED"
        and verifier_score >= 0.5
    ):

        return 1


    # --------------------------------------------------------
    # RULE 3:
    # Reference agreement + no contradiction
    #
    # We require reasonably strong overlap.
    # --------------------------------------------------------

    if (
        overlap >= 0.50
        and verifier_label
        not in {
            "CONTRADICTED"
        }
    ):

        return 1


    # --------------------------------------------------------
    # RULE 4:
    # Strong reference disagreement
    #
    # We only mark this unreliable when there is also
    # some evidence that the answer is wrong.
    # --------------------------------------------------------

    if (
        overlap < 0.20
        and verifier_label == "CONTRADICTED"
    ):

        return 0


    # --------------------------------------------------------
    # Otherwise:
    # insufficient evidence
    # --------------------------------------------------------

    return -1


# ============================================================
# CREATE LABELS
# ============================================================

print(
    "\nCreating reliability labels..."
)


label_counts = {
    1: 0,
    0: 0,
    -1: 0
}


for item in data:

    label = create_label(
        item
    )


    # Save numeric label

    item[
        "reliability_label"
    ] = label


    # Save readable label

    if label == 1:

        label_name = "RELIABLE"

    elif label == 0:

        label_name = "UNRELIABLE"

    else:

        label_name = "UNCERTAIN"


    item[
        "reliability_label_name"
    ] = label_name


    # Save useful diagnostic information

    item[
        "reference_overlap"
    ] = calculate_reference_overlap(
        item.get(
            "answer",
            ""
        ),
        item.get(
            "reference_answer",
            ""
        )
    )


    label_counts[
        label
    ] += 1


# ============================================================
# DISPLAY DISTRIBUTION
# ============================================================

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
    label_counts[1]
)

print(
    "Unreliable:",
    label_counts[0]
)

print(
    "Uncertain:",
    label_counts[-1]
)


# ============================================================
# DISPLAY EACH QUESTION'S LABEL
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


for item in data:

    print(
        f"\nID: {item.get('id')}"
    )

    print(
        "Question:",
        item.get(
            "question",
            ""
        )
    )

    print(
        "Verifier:",
        item.get(
            "verifier_label",
            "UNKNOWN"
        )
    )

    print(
        "Verifier score:",
        item.get(
            "verifier_score",
            0.0
        )
    )

    print(
        "Reference overlap:",
        round(
            item.get(
                "reference_overlap",
                0.0
            ),
            4
        )
    )

    print(
        "Reliability:",
        item.get(
            "reliability_label_name"
        )
    )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
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
    "Saved to:"
)

print(
    OUTPUT_FILE
)