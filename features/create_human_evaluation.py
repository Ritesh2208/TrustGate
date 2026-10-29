import json
import csv
import os


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/trustgate_100_labeled_v3.json"
)

OUTPUT_FILE = (
    "outputs/trustgate_human_evaluation.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("Loading TrustGate answers...")
print("=" * 60)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "Records:",
    len(data)
)


# ============================================================
# CREATE ANNOTATION FILE
# ============================================================

rows = []


for item in data:

    rows.append({

        "id":
            item["id"],

        "question":
            item["question"],

        "qwen_answer":
            item["answer"],

        "reference_answer":
            item.get(
                "reference_answer",
                ""
            ),

        # Human annotator fills this:
        #
        # RELIABLE
        # UNCERTAIN
        # UNRELIABLE
        "human_label":
            "",

        # Human confidence:
        #
        # 1 = low
        # 2 = moderate
        # 3 = high
        "human_confidence":
            "",

        "human_notes":
            ""
    })


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


fieldnames = [
    "id",
    "question",
    "qwen_answer",
    "reference_answer",
    "human_label",
    "human_confidence",
    "human_notes"
]


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print(
    "\n" + "=" * 60
)

print(
    "HUMAN EVALUATION FILE CREATED!"
)

print(
    "=" * 60
)

print(
    "File:"
)

print(
    OUTPUT_FILE
)

print(
    "\nRows:",
    len(rows)
)

print(
    "\nOpen this CSV in Excel."
)

print(
    "Fill in human_label, human_confidence, "
    "and human_notes."
)