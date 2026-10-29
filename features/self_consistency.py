import json
import os
import re
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_FILE = "outputs/truthfulqa_10.json"

OUTPUT_FILE = "outputs/truthfulqa_with_consistency.json"

NUM_SAMPLES = 5


# ============================================================
# LOAD EXISTING DATA
# ============================================================

print("Loading existing TrustGate dataset...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "Questions loaded:",
    len(data)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

print("Model loaded!")


# ============================================================
# NORMALIZE ANSWER
# ============================================================

def normalize_answer(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GENERATE ONE ANSWER
# ============================================================

def generate_sample(question):

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

            max_new_tokens=60,

            do_sample=True,

            temperature=0.7,

            top_p=0.9
        )

    answer = tokenizer.decode(
        outputs[0][
            inputs["input_ids"].shape[1]:
        ],
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# CALCULATE SELF-CONSISTENCY
# ============================================================

def calculate_consistency(answers):

    normalized = [
        normalize_answer(answer)
        for answer in answers
    ]

    counts = Counter(normalized)

    most_common_answer, frequency = (
        counts.most_common(1)[0]
    )

    consistency = (
        frequency / len(normalized)
    )

    return consistency, most_common_answer


# ============================================================
# PROCESS QUESTIONS
# ============================================================

print("\nStarting self-consistency analysis...\n")


for index, item in enumerate(data):

    question = item["question"]

    print("=" * 60)

    print(
        f"Question {index + 1}/{len(data)}"
    )

    print("\nQuestion:")
    print(question)

    samples = []

    # --------------------------------------------------------
    # Generate 5 different answers
    # --------------------------------------------------------

    for sample_number in range(NUM_SAMPLES):

        answer = generate_sample(
            question
        )

        samples.append(answer)

        print(
            f"\nSample {sample_number + 1}:"
        )

        print(answer)

    # --------------------------------------------------------
    # Calculate consistency
    # --------------------------------------------------------

    consistency, majority_answer = (
        calculate_consistency(samples)
    )

    print(
        "\nSelf-Consistency:"
    )

    print(
        round(consistency, 4)
    )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    item["self_consistency"] = consistency

    item["consistency_samples"] = samples

    item["majority_answer"] = majority_answer


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


print("\n" + "=" * 60)

print("SELF-CONSISTENCY COMPLETE!")

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)