
import json
import csv
import os

INPUT_FILE = "outputs/trustgate_standardized_new_300_answers.json"
OUTPUT_FILE = "outputs/trustgate_human_evaluation_new_300.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}. "
        "Make sure you are running this from the TrustGate project folder "
        "and that the 300-sample generation completed successfully."
    )

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

if len(records) != 300:
    raise ValueError(
        f"Expected 300 generated records, found {len(records)}."
    )

rows = []
for record in records:
    rows.append({
        "id": record["id"],
        "question": record["question"],
        "model_answer": record["answer"],
        "reference_answer": record["reference_answer"],
        "human_label": "",
        "annotator_notes": ""
    })

fieldnames = [
    "id",
    "question",
    "model_answer",
    "reference_answer",
    "human_label",
    "annotator_notes"
]

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("=" * 72)
print("TRUSTGATE — 300 SAMPLE HUMAN ANNOTATION TEMPLATE")
print("=" * 72)
print(f"Records: {len(rows)}")
print(f"Created: {OUTPUT_FILE}")
print()
print("Allowed human_label values:")
print("  RELIABLE")
print("  UNCERTAIN")
print("  UNRELIABLE")
print()
print("IMPORTANT:")
print("- Do not change question, model_answer, or reference_answer.")
print("- Do not use the reserved 100 questions.")
print("- Leave annotator_notes optional, but use them for difficult cases.")
print("- Labels must be based on factual correctness, not model confidence.")
