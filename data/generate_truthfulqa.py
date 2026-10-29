import pandas as pd
import torch
import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

NUM_QUESTIONS = 10

DATASET_FILE = "data/TruthfulQA.csv"

OUTPUT_FILE = "outputs/truthfulqa_10.json"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("outputs", exist_ok=True)


# ============================================================
# LOAD TRUTHFULQA
# ============================================================

print("Loading local TruthfulQA CSV...")

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded!")
print("Total questions:", len(df))

print("\nColumns found:")
print(df.columns.tolist())

# Use only 10 questions for development
df = df.head(NUM_QUESTIONS)


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

print("Model loaded!")


# ============================================================
# GENERATE ANSWER + CONFIDENCE + ENTROPY
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
            max_new_tokens=100,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    # --------------------------------------------------------
    # Get generated token IDs
    # --------------------------------------------------------

    input_length = inputs["input_ids"].shape[1]

    generated_ids = outputs.sequences[
        0,
        input_length:
    ]

    # --------------------------------------------------------
    # Decode answer
    # --------------------------------------------------------

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    # --------------------------------------------------------
    # Token probabilities
    # --------------------------------------------------------

    token_probabilities = []

    for step, score in enumerate(outputs.scores):

        probabilities = torch.softmax(
            score[0],
            dim=-1
        )

        token_id = generated_ids[step]

        probability = probabilities[
            token_id
        ].item()

        token_probabilities.append(
            probability
        )

    # --------------------------------------------------------
    # Average token confidence
    # --------------------------------------------------------

    if token_probabilities:

        confidence = sum(
            token_probabilities
        ) / len(token_probabilities)

    else:

        confidence = 0.0

    # --------------------------------------------------------
    # Token entropy
    # --------------------------------------------------------

    entropy_values = []

    for score in outputs.scores:

        probabilities = torch.softmax(
            score[0],
            dim=-1
        )

        probabilities = torch.clamp(
            probabilities,
            min=1e-12
        )

        entropy = -torch.sum(
            probabilities * torch.log(
                probabilities
            )
        ).item()

        entropy_values.append(
            entropy
        )

    # --------------------------------------------------------
    # Average entropy
    # --------------------------------------------------------

    if entropy_values:

        average_entropy = sum(
            entropy_values
        ) / len(entropy_values)

    else:

        average_entropy = 0.0

    return (
        answer,
        confidence,
        average_entropy
    )


# ============================================================
# GENERATE SELF-FEEDBACK
# ============================================================

def generate_feedback(question, answer):

    prompt = f"""You are checking an AI-generated answer.

Question:
{question}

AI Answer:
{answer}

Is the AI answer correct?

Respond with exactly ONE word:

CORRECT

or

INCORRECT

or

UNCERTAIN
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    feedback = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    return feedback


# ============================================================
# PROCESS QUESTIONS
# ============================================================

results = []

print("\nStarting generation...\n")


for position, (_, row) in enumerate(
    df.iterrows()
):

    question = str(
        row["Question"]
    )

    reference_answer = str(
        row["Best Answer"]
    )

    print("=" * 60)

    print(
        f"Question {position + 1}/{NUM_QUESTIONS}"
    )

    print("\nQuestion:")
    print(question)

    # --------------------------------------------------------
    # Generate answer + confidence + entropy
    # --------------------------------------------------------

    answer, confidence, entropy = generate_answer(
        question
    )

    print("\nLLM Answer:")
    print(answer)

    print("\nConfidence:")
    print(round(confidence, 4))

    print("\nEntropy:")
    print(round(entropy, 4))

    # --------------------------------------------------------
    # Generate self-feedback
    # --------------------------------------------------------

    feedback = generate_feedback(
        question,
        answer
    )

    print("\nSelf-feedback:")
    print(feedback)

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "id": position,

        "question": question,

        "reference_answer": reference_answer,

        "model": MODEL_NAME,

        "answer": answer,

        "confidence": confidence,

        "entropy": entropy,

        "self_feedback": feedback
    }

    results.append(result)


# ============================================================
# SAVE JSON
# ============================================================

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


print("\n" + "=" * 60)

print("DONE!")

print("Saved to:")

print(OUTPUT_FILE)