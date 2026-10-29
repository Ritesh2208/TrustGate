import json
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
    "outputs/truthfulqa_with_verifier_v2.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_with_hidden_states.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading TrustGate feature dataset...")

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
# EXTRACT HIDDEN STATE
# ============================================================

def extract_hidden_state(
    question,
    answer
):

    prompt = f"""You are a helpful assistant.

Question:
{question}

Answer:
{answer}
"""


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True
        )


    # --------------------------------------------------------
    # Get final hidden layer
    # --------------------------------------------------------

    last_hidden_state = (
        outputs.hidden_states[-1]
    )


    # Shape:
    #
    # [batch_size, sequence_length, hidden_size]
    #
    # Example for Qwen 0.5B:
    #
    # [1, tokens, hidden_size]
    #


    # --------------------------------------------------------
    # Mean pooling
    # --------------------------------------------------------

    attention_mask = (
        inputs["attention_mask"]
    )


    mask = attention_mask.unsqueeze(
        -1
    ).float()


    masked_hidden_states = (
        last_hidden_state * mask
    )


    summed = (
        masked_hidden_states.sum(
            dim=1
        )
    )


    token_count = (
        mask.sum(
            dim=1
        ).clamp(
            min=1e-9
        )
    )


    pooled = (
        summed / token_count
    )


    # --------------------------------------------------------
    # Convert to list
    # --------------------------------------------------------

    vector = (
        pooled[0]
        .cpu()
        .tolist()
    )


    return vector


# ============================================================
# PROCESS QUESTIONS
# ============================================================

print(
    "\nStarting hidden-state extraction..."
)


for index, item in enumerate(data):

    question = item[
        "question"
    ]

    answer = item[
        "answer"
    ]


    print(
        "\n" + "=" * 60
    )

    print(
        f"Question {index + 1}/{len(data)}"
    )

    print(
        "\nQuestion:"
    )

    print(
        question
    )


    # --------------------------------------------------------
    # Extract hidden representation
    # --------------------------------------------------------

    hidden_vector = extract_hidden_state(
        question,
        answer
    )


    # --------------------------------------------------------
    # Save hidden vector
    # --------------------------------------------------------

    item[
        "hidden_state"
    ] = hidden_vector


    item[
        "hidden_state_dimension"
    ] = len(
        hidden_vector
    )


    print(
        "\nHidden-state dimension:"
    )

    print(
        len(hidden_vector)
    )


    print(
        "\nFirst 10 values:"
    )

    print(
        hidden_vector[:10]
    )


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


print(
    "\n" + "=" * 60
)

print(
    "HIDDEN-STATE EXTRACTION COMPLETE!"
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)