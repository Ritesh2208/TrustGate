import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "outputs/trustgate_human_evaluation_completed.csv"

EXPECTED_ROWS = 100

REQUIRED_COLUMNS = [
    "id",
    "question",
    "qwen_answer",
    "reference_answer",
    "human_label",
    "human_confidence",
    "human_notes",
]

VALID_LABELS = {
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE",
}

VALID_CONFIDENCE = {
    1,
    2,
    3,
}


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("TRUSTGATE HUMAN EVALUATION VALIDATION")
print("=" * 60)


# ============================================================
# CHECK FILE EXISTS
# ============================================================

if not os.path.exists(INPUT_FILE):

    print("\nERROR: File not found.")
    print("\nExpected file:")
    print(INPUT_FILE)

    print("\nCurrent directory:")
    print(os.getcwd())

    raise SystemExit(1)


print("\nLoading:")
print(INPUT_FILE)


# ============================================================
# LOAD CSV
# ============================================================

try:

    df = pd.read_csv(
        INPUT_FILE
    )

except Exception as e:

    print("\nERROR: Could not read CSV.")
    print(e)

    raise SystemExit(1)


print(
    "\nRecords loaded:",
    len(df)
)


# ============================================================
# CHECK RECORD COUNT
# ============================================================

print("\n" + "-" * 60)
print("CHECKING RECORD COUNT")
print("-" * 60)


if len(df) != EXPECTED_ROWS:

    print(
        f"FAIL: Expected {EXPECTED_ROWS} rows, "
        f"but found {len(df)}."
    )

    raise SystemExit(1)


print(
    f"PASS: Exactly {EXPECTED_ROWS} rows found."
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print("\n" + "-" * 60)
print("CHECKING REQUIRED COLUMNS")
print("-" * 60)


missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]


if missing_columns:

    print("FAIL: Missing columns:")

    for column in missing_columns:
        print(" -", column)

    raise SystemExit(1)


print(
    "PASS: All required columns are present."
)


# ============================================================
# CHECK IDS
# ============================================================

print("\n" + "-" * 60)
print("CHECKING QUESTION IDS")
print("-" * 60)


if df["id"].isna().any():

    print("FAIL: Missing IDs detected.")

    raise SystemExit(1)


try:

    ids = pd.to_numeric(
        df["id"]
    ).astype(int)

except Exception:

    print(
        "FAIL: IDs must be numeric."
    )

    raise SystemExit(1)


if ids.duplicated().any():

    duplicates = ids[
        ids.duplicated()
    ].tolist()

    print(
        "FAIL: Duplicate IDs found:"
    )

    print(
        duplicates
    )

    raise SystemExit(1)


expected_ids = set(
    range(
        EXPECTED_ROWS
    )
)

actual_ids = set(
    ids.tolist()
)


if actual_ids != expected_ids:

    missing_ids = sorted(
        expected_ids - actual_ids
    )

    extra_ids = sorted(
        actual_ids - expected_ids
    )

    print(
        "FAIL: IDs do not match 0-99."
    )

    if missing_ids:

        print(
            "Missing IDs:",
            missing_ids
        )

    if extra_ids:

        print(
            "Unexpected IDs:",
            extra_ids
        )

    raise SystemExit(1)


print(
    "PASS: IDs 0-99 are present exactly once."
)


# ============================================================
# CHECK QUESTIONS
# ============================================================

print("\n" + "-" * 60)
print("CHECKING QUESTIONS")
print("-" * 60)


question_missing = (
    df["question"]
    .isna()
)


question_empty = (
    df["question"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
)


question_errors = (
    question_missing
    |
    question_empty
)


if question_errors.any():

    bad_ids = ids[
        question_errors
    ].tolist()

    print(
        "FAIL: Missing/empty questions."
    )

    print(
        "Affected IDs:",
        bad_ids
    )

    raise SystemExit(1)


print(
    "PASS: All questions are present."
)


# ============================================================
# CHECK QWEN ANSWERS
# ============================================================

print("\n" + "-" * 60)
print("CHECKING QWEN ANSWERS")
print("-" * 60)


answer_missing = (
    df["qwen_answer"]
    .isna()
)


answer_empty = (
    df["qwen_answer"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
)


answer_errors = (
    answer_missing
    |
    answer_empty
)


if answer_errors.any():

    bad_ids = ids[
        answer_errors
    ].tolist()

    print(
        "FAIL: Missing/empty Qwen answers."
    )

    print(
        "Affected IDs:",
        bad_ids
    )

    raise SystemExit(1)


print(
    "PASS: All Qwen answers are present."
)


# ============================================================
# CHECK HUMAN LABELS
# ============================================================

print("\n" + "-" * 60)
print("CHECKING HUMAN LABELS")
print("-" * 60)


labels = (
    df["human_label"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


missing_labels = (
    labels == ""
)


if missing_labels.any():

    bad_ids = ids[
        missing_labels
    ].tolist()

    print(
        "FAIL: Missing human labels."
    )

    print(
        "Affected IDs:",
        bad_ids
    )

    raise SystemExit(1)


invalid_labels = sorted(
    set(labels)
    -
    VALID_LABELS
)


if invalid_labels:

    print(
        "FAIL: Invalid human labels:"
    )

    for label in invalid_labels:

        print(
            " -",
            label
        )

    print(
        "\nAllowed:"
    )

    for label in sorted(
        VALID_LABELS
    ):

        print(
            " -",
            label
        )

    raise SystemExit(1)


print(
    "PASS: All human labels are valid."
)


# ============================================================
# HUMAN LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("HUMAN LABEL DISTRIBUTION")
print("=" * 60)


label_counts = (
    labels
    .value_counts()
)


for label in [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE",
]:

    count = int(
        label_counts.get(
            label,
            0
        )
    )

    percentage = (
        count
        /
        len(df)
        *
        100
    )

    print(
        f"{label:<12}: "
        f"{count:>3} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# CHECK HUMAN CONFIDENCE
# ============================================================

print("\n" + "-" * 60)
print("CHECKING HUMAN CONFIDENCE")
print("-" * 60)


if df["human_confidence"].isna().any():

    bad_ids = ids[
        df["human_confidence"].isna()
    ].tolist()

    print(
        "FAIL: Missing confidence values."
    )

    print(
        "Affected IDs:",
        bad_ids
    )

    raise SystemExit(1)


try:

    confidence = pd.to_numeric(
        df["human_confidence"]
    )

except Exception:

    print(
        "FAIL: Human confidence must be numeric."
    )

    raise SystemExit(1)


invalid_confidence = sorted(
    set(
        confidence.tolist()
    )
    -
    VALID_CONFIDENCE
)


if invalid_confidence:

    print(
        "FAIL: Invalid confidence values:"
    )

    print(
        invalid_confidence
    )

    print(
        "\nAllowed values: 1, 2, 3"
    )

    raise SystemExit(1)


print(
    "PASS: All human confidence values are valid."
)


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("HUMAN CONFIDENCE DISTRIBUTION")
print("=" * 60)


confidence_counts = (
    confidence
    .value_counts()
    .sort_index()
)


for value in [
    1,
    2,
    3,
]:

    count = int(
        confidence_counts.get(
            value,
            0
        )
    )

    percentage = (
        count
        /
        len(df)
        *
        100
    )

    print(
        f"Confidence {value}: "
        f"{count} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# NOTES CHECK
# ============================================================

print("\n" + "-" * 60)
print("CHECKING HUMAN NOTES")
print("-" * 60)


notes_missing = (
    df["human_notes"]
    .isna()
    |
    df["human_notes"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
)


notes_count = int(
    notes_missing.sum()
)


print(
    "Rows without notes:",
    notes_count
)


if notes_count == 0:

    print(
        "PASS: All rows contain notes."
    )

else:

    print(
        "INFO: Notes are optional."
    )


# ============================================================
# LABEL + CONFIDENCE CROSS-TABLE
# ============================================================

print("\n" + "=" * 60)
print("LABEL / CONFIDENCE CROSS-TABLE")
print("=" * 60)


cross_table = pd.crosstab(
    labels,
    confidence
)


print(
    cross_table
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("HUMAN EVALUATION VALIDATION COMPLETE")
print("=" * 60)

print(
    "\nSTATUS: PASS"
)

print(
    "\nIndependent human labels are ready "
    "for the next TrustGate experiment."
)

print(
    "\nValidated file:"
)

print(
    INPUT_FILE
)

print(
    "\nRecords:",
    len(df)
)

print("=" * 60)