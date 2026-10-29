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
    "outputs/truthfulqa_100_answers.json"
)

NUM_QUESTIONS = 100

MAX_NEW_TOKENS = 100


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# LOAD TRUTHFULQA
# ============================================================

print(
    "Loading local TruthfulQA CSV..."
)

df = pd.read_csv(
    DATASET_FILE
)

print(
    "Dataset loaded!"
)

print(
    "Total questions:",
    len(df)
)


# ============================================================
# CHECK COLUMNS
# ============================================================

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# TAKE QUESTIONS
# ============================================================

df = df.head(
    NUM_QUESTIONS
)

print(
    "\nQuestions selected:",
    len(df)
)


# ============================================================
# LOAD QWEN
# ============================================================

print(
    "\nLoading Qwen..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()

print(
    "Qwen loaded!"
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question
):

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
            pad_token_id=(
                tokenizer.eos_token_id
            )
        )

    generated_ids = (
        outputs[0][
            inputs["input_ids"].shape[1]:
        ]
    )

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# GENERATE DATASET
# ============================================================

results = []

print(
    "\nStarting generation..."
)


for position, (_, row) in enumerate(
    df.iterrows()
):

    question = str(
        row["Question"]
    )

    reference_answer = str(
        row["Best Answer"]
    )


    print(
        "\n" + "=" * 60
    )

    print(
        f"Question {position + 1}/{len(df)}"
    )

    print(
        "Question:"
    )

    print(
        question
    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        question
    )


    print(
        "\nLLM Answer:"
    )

    print(
        answer
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = {

        "id": position,

        "question": question,

        "reference_answer":
            reference_answer,

        "model": MODEL_NAME,

        "answer": answer
    }


    results.append(
        result
    )


# ============================================================
# SAVE
# ============================================================

print(
    "\nSaving dataset..."
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
    "ANSWER DATASET GENERATION COMPLETE!"
)

print(
    "Questions generated:",
    len(results)
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)