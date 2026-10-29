import json
import os

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_FILE = (
    "outputs/truthfulqa_next_100_answers.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_next_100_features.json"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TRUSTGATE FEATURE EXTRACTION: NEXT 100")
print("=" * 60)

print("\nLoading next 100-question dataset...")

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

print(
    "ID range:",
    data[0]["id"],
    "to",
    data[-1]["id"]
)


# ============================================================
# LOAD QWEN
# ============================================================

print("\nLoading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()

print("Qwen loaded!")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    question,
    answer
):

    # --------------------------------------------------------
    # EXACT SAME PROMPT STRUCTURE AS ORIGINAL 100
    # --------------------------------------------------------

    prompt = f"""You are a helpful assistant.

Question:
{question}

Give the best possible answer.

Answer:
{answer}
"""


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )


    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True
        )


    # ========================================================
    # TOKEN PROBABILITIES
    # ========================================================

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    token_probabilities = []

    sequence_length = (
        inputs["input_ids"].shape[1]
    )


    for position in range(
        sequence_length - 1
    ):

        next_token = inputs[
            "input_ids"
        ][
            0,
            position + 1
        ]


        probability = probabilities[
            0,
            position,
            next_token
        ].item()


        token_probabilities.append(
            probability
        )


    if token_probabilities:

        confidence = (
            sum(token_probabilities)
            /
            len(token_probabilities)
        )

    else:

        confidence = 0.0


    # ========================================================
    # ENTROPY
    # ========================================================

    entropy_values = []


    for position in range(
        sequence_length - 1
    ):

        distribution = probabilities[
            0,
            position
        ]


        distribution = torch.clamp(
            distribution,
            min=1e-12
        )


        entropy = -torch.sum(
            distribution *
            torch.log(
                distribution
            )
        ).item()


        entropy_values.append(
            entropy
        )


    if entropy_values:

        average_entropy = (
            sum(entropy_values)
            /
            len(entropy_values)
        )

    else:

        average_entropy = 0.0


    # ========================================================
    # HIDDEN STATE
    # ========================================================

    last_hidden_state = (
        outputs.hidden_states[-1]
    )


    hidden_vector = (
        last_hidden_state[
            0,
            -1,
            :
        ]
        .detach()
        .cpu()
        .tolist()
    )


    return (
        float(confidence),
        float(average_entropy),
        hidden_vector
    )


# ============================================================
# PROCESS DATASET
# ============================================================

results = []

print("\nStarting feature extraction...")


for position, item in enumerate(data):

    question = item.get(
        "question",
        ""
    )

    answer = item.get(
        "answer",
        ""
    )

    reference_answer = item.get(
        "reference_answer",
        ""
    )


    print(
        f"Processing {position + 1}/{len(data)} "
        f"(ID: {item.get('id')})"
    )


    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    confidence, entropy, hidden_state = (
        extract_features(
            question,
            answer
        )
    )


    # --------------------------------------------------------
    # SAVE RECORD
    # --------------------------------------------------------

    result = {

        "id":
            item.get("id"),

        "question":
            question,

        "answer":
            answer,

        "reference_answer":
            reference_answer,

        "model":
            MODEL_NAME,

        "confidence":
            confidence,

        "entropy":
            entropy,

        "hidden_state":
            hidden_state

    }


    results.append(
        result
    )


# ============================================================
# SAVE RESULTS
# ============================================================

print("\nSaving extracted features...")

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
print("NEXT 100 FEATURE EXTRACTION COMPLETE!")
print("=" * 60)

print(
    "Records processed:",
    len(results)
)

print(
    "Hidden dimension:",
    len(results[0]["hidden_state"])
)

print(
    "Saved to:",
    OUTPUT_FILE
)