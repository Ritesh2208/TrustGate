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
    "outputs/truthfulqa_next_100_consistency.json"
)


NUM_GENERATIONS = 5

MAX_NEW_TOKENS = 100

TEMPERATURE = 0.8

TOP_P = 0.9


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
print("TRUSTGATE NEXT 100 SELF-CONSISTENCY")
print("=" * 60)

print("\nLoading answers...")

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
    "First ID:",
    data[0]["id"]
)

print(
    "Last ID:",
    data[-1]["id"]
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

model.eval()

print(
    "Qwen loaded!"
)


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(question):

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
# LOAD EMBEDDING MODEL
# ============================================================

print(
    "\nLoading embedding model..."
)

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)


embedding_model = SentenceTransformer(

    "sentence-transformers/all-MiniLM-L6-v2"

)

print(
    "Embedding model loaded!"
)


# ============================================================
# CALCULATE SELF-CONSISTENCY
# ============================================================

def calculate_consistency(responses):

    if len(responses) < 2:

        return 0.0


    embeddings = embedding_model.encode(

        responses

    )


    similarities = cosine_similarity(

        embeddings

    )


    total_similarity = 0.0

    comparisons = 0


    for i in range(
        len(responses)
    ):

        for j in range(
            i + 1,
            len(responses)
        ):

            total_similarity += similarities[i][j]

            comparisons += 1


    if comparisons == 0:

        return 0.0


    consistency = (

        total_similarity
        /
        comparisons

    )


    return float(
        consistency
    )


# ============================================================
# PROCESS DATASET
# ============================================================

results = []


print(
    "\nStarting self-consistency analysis..."
)


for position, record in enumerate(data):

    record_id = record["id"]

    question = record["question"]

    original_answer = record["answer"]


    print(
        "\n" + "=" * 60
    )

    print(
        f"Processing {position + 1}/{len(data)}"
    )

    print(
        "ID:",
        record_id
    )


    # --------------------------------------------------------
    # GENERATE MULTIPLE RESPONSES
    # --------------------------------------------------------

    responses = []


    # Include original deterministic answer
    responses.append(
        original_answer
    )


    print(
        "Generating alternative answers..."
    )


    for generation in range(
        NUM_GENERATIONS - 1
    ):

        print(
            f"Generation {generation + 1}/{NUM_GENERATIONS - 1}"
        )


        response = generate_response(
            question
        )


        responses.append(
            response
        )


    # --------------------------------------------------------
    # CALCULATE CONSISTENCY
    # --------------------------------------------------------

    consistency_score = calculate_consistency(
        responses
    )


    print(
        "Self-consistency:",
        round(
            consistency_score,
            4
        )
    )


    # --------------------------------------------------------
    # SAVE RECORD
    # --------------------------------------------------------

    result = {

        "id": record_id,

        "question": question,

        "reference_answer":
            record["reference_answer"],

        "answer":
            original_answer,

        "self_consistency":
            consistency_score,

        "num_generations":
            NUM_GENERATIONS,

        "generated_responses":
            responses

    }


    results.append(
        result
    )


# ============================================================
# SAVE RESULTS
# ============================================================

print(
    "\nSaving results..."
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
# SUMMARY
# ============================================================

scores = [

    record["self_consistency"]

    for record in results

]


print(
    "\n" + "=" * 60
)

print(
    "NEXT 100 SELF-CONSISTENCY COMPLETE!"
)

print(
    "=" * 60
)


print(
    "Records processed:",
    len(results)
)


print(
    "Minimum:",
    round(
        min(scores),
        4
    )
)


print(
    "Mean:",
    round(
        sum(scores) / len(scores),
        4
    )
)


print(
    "Maximum:",
    round(
        max(scores),
        4
    )
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)