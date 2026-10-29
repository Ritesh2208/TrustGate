import json
import pandas as pd
import os


TRAINING_FILE = "outputs/trustgate_standardized_500_labeled.json"
TRUTHFULQA_FILE = "data/TruthfulQA.csv"
OUTPUT_FILE = "outputs/trustgate_heldout_290.json"


# ============================================================
# LOAD TRAINING QUESTIONS
# ============================================================

with open(
    TRAINING_FILE,
    "r",
    encoding="utf-8"
) as f:
    training_data = json.load(f)

training_questions = {
    str(item["question"]).strip()
    for item in training_data
}


# ============================================================
# LOAD TRUTHFULQA
# ============================================================

df = pd.read_csv(
    TRUTHFULQA_FILE
)

print("TruthfulQA rows:", len(df))
print("Training questions:", len(training_questions))


# ============================================================
# FIND HELD-OUT QUESTIONS
# ============================================================

heldout = []

for csv_index, row in df.iterrows():

    question = str(
        row["Question"]
    ).strip()

    if question not in training_questions:

        heldout.append({
            "heldout_id": len(heldout),
            "truthfulqa_csv_index": int(csv_index),
            "question": question,
            "best_answer": str(
                row["Best Answer"]
            ),
            "best_incorrect_answer": str(
                row["Best Incorrect Answer"]
            ),
            "correct_answers": str(
                row["Correct Answers"]
            ),
            "incorrect_answers": str(
                row["Incorrect Answers"]
            ),
            "category": str(
                row["Category"]
            ),
            "type": str(
                row["Type"]
            ),
            "source": str(
                row["Source"]
            )
        })


# ============================================================
# VERIFY
# ============================================================

print("\nHeld-out questions:", len(heldout))

assert len(heldout) == 290, (
    f"Expected 290 held-out questions, "
    f"found {len(heldout)}"
)


# Make sure there is no overlap
heldout_questions = {
    item["question"]
    for item in heldout
}

overlap = (
    training_questions
    & heldout_questions
)

print(
    "Training/held-out overlap:",
    len(overlap)
)

assert len(overlap) == 0


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        heldout,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nSaved:")
print(OUTPUT_FILE)

print("\nFirst 3 held-out questions:")

for item in heldout[:3]:

    print(
        f"\n{item['heldout_id']}: "
        f"{item['question']}"
    )