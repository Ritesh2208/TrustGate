import json
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_FILE = (
    "outputs/truthfulqa_100_answers.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_100_consistency.json"
)

# Number of alternative answers per question
NUM_SAMPLES = 5

MAX_NEW_TOKENS = 100

TEMPERATURE = 0.7

TOP_P = 0.9

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# LOAD ORIGINAL 100-QUESTION DATASET
# ============================================================

print(
    "Loading 100-question dataset..."
)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "Total questions:",
    len(data)
)


# ============================================================
# LOAD PREVIOUS PROGRESS
# ============================================================

if os.path.exists(
    OUTPUT_FILE
):

    print(
        "\nExisting consistency results found."
    )

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

else:

    print(
        "\nNo previous progress found."
    )

    results = []


# ============================================================
# FIND COMPLETED QUESTION IDs
# ============================================================

completed_ids = {
    item.get("id")
    for item in results
}


print(
    "\nAlready completed:",
    len(completed_ids)
)

print(
    "Remaining:",
    len(data) - len(completed_ids)
)


# ============================================================
# IF EVERYTHING IS COMPLETE
# ============================================================

if len(completed_ids) >= len(data):

    print(
        "\nAll 100 questions are already complete!"
    )

    print(
        "No new generation is required."
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    raise SystemExit


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
# LOAD SENTENCE EMBEDDING MODEL
# ============================================================

print(
    "\nLoading sentence embedding model..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model loaded!"
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

            do_sample=True,

            temperature=TEMPERATURE,

            top_p=TOP_P,

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
# CALCULATE SELF-CONSISTENCY
# ============================================================

def calculate_consistency(
    answers
):

    if len(answers) < 2:

        return 0.0


    # --------------------------------------------------------
    # Generate semantic embeddings
    # --------------------------------------------------------

    embeddings = (
        embedding_model.encode(
            answers,

            convert_to_numpy=True,

            normalize_embeddings=True
        )
    )


    # --------------------------------------------------------
    # Pairwise cosine similarity
    # --------------------------------------------------------

    similarity_matrix = (
        cosine_similarity(
            embeddings
        )
    )


    scores = []


    for i in range(
        len(answers)
    ):

        for j in range(
            i + 1,
            len(answers)
        ):

            score = float(
                similarity_matrix[i][j]
            )

            scores.append(
                score
            )


    if not scores:

        return 0.0


    # --------------------------------------------------------
    # Average pairwise similarity
    # --------------------------------------------------------

    average_similarity = (
        sum(scores)
        /
        len(scores)
    )


    # --------------------------------------------------------
    # Convert [-1, 1] → [0, 1]
    # --------------------------------------------------------

    consistency = (
        average_similarity + 1.0
    ) / 2.0


    consistency = max(
        0.0,
        min(
            1.0,
            consistency
        )
    )


    return float(
        consistency
    )


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_results(
    results
):

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
# PROCESS REMAINING QUESTIONS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "STARTING / RESUMING SELF-CONSISTENCY"
)

print(
    "=" * 60
)

print(
    "Samples per question:",
    NUM_SAMPLES
)

print(
    "Remaining generations:",
    (
        len(data) - len(completed_ids)
    )
    *
    NUM_SAMPLES
)


for position, item in enumerate(
    data
):

    item_id = item.get(
        "id"
    )


    # --------------------------------------------------------
    # SKIP ALREADY COMPLETED QUESTIONS
    # --------------------------------------------------------

    if item_id in completed_ids:

        continue


    question = item.get(
        "question",
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


    # ========================================================
    # GENERATE MULTIPLE ANSWERS
    # ========================================================

    answers = []


    for sample_index in range(
        NUM_SAMPLES
    ):

        print(
            f"\nGenerating answer "
            f"{sample_index + 1}/{NUM_SAMPLES}..."
        )


        answer = generate_answer(
            question
        )


        answers.append(
            answer
        )


        print(
            "Answer:"
        )

        print(
            answer
        )


    # ========================================================
    # CALCULATE CONSISTENCY
    # ========================================================

    consistency = (
        calculate_consistency(
            answers
        )
    )


    print(
        "\nSelf-consistency:"
    )

    print(
        round(
            consistency,
            4
        )
    )


    # ========================================================
    # CREATE RESULT
    # ========================================================

    result = {

        "id":
            item_id,

        "question":
            question,

        "reference_answer":
            reference_answer,

        "model":
            MODEL_NAME,

        "num_samples":
            NUM_SAMPLES,

        "consistency_answers":
            answers,

        "self_consistency_v2":
            float(
                consistency
            )
    }


    results.append(
        result
    )


    # Add to completed set

    completed_ids.add(
        item_id
    )


    # ========================================================
    # SAVE AFTER EVERY QUESTION
    # ========================================================

    save_results(
        results
    )


    print(
        "\nProgress saved!"
    )

    print(
        f"Completed: "
        f"{len(results)}/{len(data)}"
    )

    print(
        f"Remaining: "
        f"{len(data) - len(results)}"
    )


# ============================================================
# FINAL SAVE
# ============================================================

save_results(
    results
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "100-QUESTION SELF-CONSISTENCY COMPLETE!"
)

print(
    "=" * 60
)

print(
    "Questions processed:",
    len(results)
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)