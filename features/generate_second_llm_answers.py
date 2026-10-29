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

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

DATASET_FILE = "data/TruthfulQA.csv"

OUTPUT_FILE = (
    "outputs/truthfulqa_100_smollm_answers.json"
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

print("=" * 60)
print("LOADING TRUTHFULQA")
print("=" * 60)

df = pd.read_csv(
    DATASET_FILE
)

print(
    "Total questions:",
    len(df)
)


# ============================================================
# SELECT SAME 100 QUESTIONS
# ============================================================

df = df.head(
    NUM_QUESTIONS
)

print(
    "Questions selected:",
    len(df)
)


# ============================================================
# LOAD SECOND LLM
# ============================================================

print("\n" + "=" * 60)
print("LOADING SECOND LLM")
print("=" * 60)

print(
    "Model:",
    MODEL_NAME
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
    "Model loaded successfully!"
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    messages = [
        {
            "role": "user",
            "content": (
                "Answer the following question accurately and concisely.\n\n"
                f"Question: {question}"
            )
        }
    ]


    # Use chat template if available
    try:

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    except Exception:

        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Answer:\n"
        )


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


print("\n" + "=" * 60)
print("STARTING SECOND LLM GENERATION")
print("=" * 60)


for position, (_, row) in enumerate(
    df.iterrows()
):

    question = str(
        row["Question"]
    )


    reference_answer = str(
        row["Best Answer"]
    )


    print("\n" + "-" * 60)

    print(
        f"Question {position + 1}/{len(df)}"
    )

    print(
        "Question:",
        question
    )


    answer = generate_answer(
        question
    )


    print(
        "\nSecond LLM Answer:"
    )

    print(
        answer
    )


    results.append(
        {

            "id": position,

            "question": question,

            "reference_answer":
                reference_answer,

            "model":
                MODEL_NAME,

            "answer":
                answer
        }
    )


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
    "SECOND LLM GENERATION COMPLETE!"
)

print("=" * 60)

print(
    "Questions generated:",
    len(results)
)

print(
    "Saved to:",
    OUTPUT_FILE
)
