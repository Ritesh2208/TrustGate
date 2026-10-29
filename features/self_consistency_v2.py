import json
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
    "outputs/truthfulqa_with_hidden_states.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_with_consistency_v2.json"
)

NUM_SAMPLES = 5

MAX_NEW_TOKENS = 100

TEMPERATURE = 0.7

TOP_P = 0.9

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading TrustGate dataset...")

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
# GENERATE ONE ANSWER
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
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
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

    embeddings = embedding_model.encode(
        answers,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Pairwise cosine similarity
    # --------------------------------------------------------

    similarity_matrix = cosine_similarity(
        embeddings
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

    consistency = (
        sum(scores) /
        len(scores)
    )

    # --------------------------------------------------------
    # Convert similarity to [0, 1]
    #
    # Cosine similarity can technically be [-1, 1].
    # For TrustGate we map it to [0, 1].
    # --------------------------------------------------------

    consistency = (
        consistency + 1.0
    ) / 2.0

    consistency = max(
        0.0,
        min(
            1.0,
            consistency
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Convert NumPy float32 to Python float
    # so json.dump() can serialize it.
    # --------------------------------------------------------

    return float(
        consistency
    )


# ============================================================
# PROCESS QUESTIONS
# ============================================================

print(
    "\nStarting self-consistency generation..."
)


for index, item in enumerate(data):

    question = item[
        "question"
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

    answers = []


    # --------------------------------------------------------
    # Generate multiple answers
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Calculate consistency
    # --------------------------------------------------------

    consistency = calculate_consistency(
        answers
    )

    # Explicit Python float conversion
    consistency = float(
        consistency
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


    # --------------------------------------------------------
    # Save generated answers
    # --------------------------------------------------------

    item[
        "consistency_answers"
    ] = answers


    # --------------------------------------------------------
    # Save consistency score
    # --------------------------------------------------------

    item[
        "self_consistency_v2"
    ] = consistency


# ============================================================
# SAVE OUTPUT
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
        data,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "SELF-CONSISTENCY V2 COMPLETE!"
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)