import json
import os
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_FILE = "data/TruthfulQA.csv"

OUTPUT_FILE = (
    "outputs/truthfulqa_next_100_answers.json"
)

START_INDEX = 100
END_INDEX = 200

MAX_NEW_TOKENS = 100


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("TRUSTGATE: GENERATING QWEN ANSWERS 101-200")
print("=" * 60)

print("\nLoading TruthfulQA dataset...")

df = pd.read_csv(
    DATASET_FILE
)

print("Total questions available:", len(df))


# ============================================================
# SELECT QUESTIONS 101–200
# ============================================================

df = df.iloc[
    START_INDEX:END_INDEX
].copy()

print(
    "Questions selected:",
    len(df)
)

print(
    "Original dataset indices:",
    f"{START_INDEX} to {END_INDEX - 1}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()

print("Qwen loaded successfully!")


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    prompt = f"""You are a helpful assistant.

Question:
{question}

Give the best possible answer.

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# GENERATE ANSWERS
# ============================================================

results = []

print("\nStarting answer generation...")


for dataset_index, row in df.iterrows():

    question_id = int(dataset_index)

    question = str(
        row["Question"]
    )

    reference_answer = str(
        row["Best Answer"]
    )

    print("\n" + "=" * 60)

    print(
        f"Question ID {question_id}"
    )

    print(
        "Progress:",
        len(results) + 1,
        "/",
        len(df)
    )

    print("\nQuestion:")

    print(question)


    # Generate answer

    answer = generate_answer(
        question
    )


    print("\nQwen Answer:")

    print(answer)


    # Save record

    results.append({

        "id": question_id,

        "question": question,

        "reference_answer": reference_answer,

        "model": MODEL_NAME,

        "answer": answer

    })


# ============================================================
# SAVE RESULTS
# ============================================================

print("\nSaving results...")

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

print("\n" + "=" * 60)

print(
    "NEXT 100 QWEN ANSWERS COMPLETE!"
)

print("=" * 60)

print(
    "Questions generated:",
    len(results)
)

print(
    "Question IDs:",
    START_INDEX,
    "to",
    END_INDEX - 1
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)