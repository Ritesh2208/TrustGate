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
    "outputs/truthfulqa_100_answers.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_100_features.json"
)

MAX_NEW_TOKENS = 100


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

print("Loading 100-question dataset...")

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
# EXTRACT FEATURES
# ============================================================

def extract_features(
    question,
    answer
):

    # --------------------------------------------------------
    # Reconstruct the same prompt used during generation
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
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True
        )


    # ========================================================
    # TOKEN LOGITS
    # ========================================================

    logits = outputs.logits


    # --------------------------------------------------------
    # Calculate probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=-1
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    # Probability assigned to the actual next token

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

    # Last hidden layer

    last_hidden_state = (
        outputs.hidden_states[-1]
    )


    # Take the final token representation

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
# PROCESS 100 QUESTIONS
# ============================================================

results = []

print(
    "\nStarting 100-question feature extraction..."
)


for position, item in enumerate(
    data
):

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
        "\n" + "=" * 60
    )

    print(
        f"Question {position + 1}/{len(data)}"
    )

    print(
        "Question:"
    )

    print(
        question
    )


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    confidence, entropy, hidden_state = (
        extract_features(
            question,
            answer
        )
    )


    print(
        "\nConfidence:",
        round(
            confidence,
            4
        )
    )


    print(
        "Entropy:",
        round(
            entropy,
            4
        )
    )


    print(
        "Hidden-state dimension:",
        len(hidden_state)
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "id":
            item.get(
                "id"
            ),

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
# SAVE
# ============================================================

print(
    "\nSaving feature dataset..."
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
    "100-QUESTION FEATURE EXTRACTION COMPLETE!"
)

print(
    "Records:",
    len(results)
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)