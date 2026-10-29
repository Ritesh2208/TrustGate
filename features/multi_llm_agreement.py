import json
import os

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


# ============================================================
# SETTINGS
# ============================================================

QWEN_FILE = (
    "outputs/truthfulqa_100_answers.json"
)

SMOLLM_FILE = (
    "outputs/truthfulqa_100_smollm_answers.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_100_multi_llm_agreement.json"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TRUSTGATE MULTI-LLM AGREEMENT ANALYSIS")
print("=" * 60)

print("\nLoading Qwen answers...")

with open(
    QWEN_FILE,
    "r",
    encoding="utf-8"
) as f:

    qwen_data = json.load(f)


print(
    "Qwen records:",
    len(qwen_data)
)


print("\nLoading SmolLM answers...")

with open(
    SMOLLM_FILE,
    "r",
    encoding="utf-8"
) as f:

    smollm_data = json.load(f)


print(
    "SmolLM records:",
    len(smollm_data)
)


# ============================================================
# VALIDATE IDS
# ============================================================

qwen_by_id = {
    item["id"]: item
    for item in qwen_data
}

smollm_by_id = {
    item["id"]: item
    for item in smollm_data
}


qwen_ids = set(
    qwen_by_id.keys()
)

smollm_ids = set(
    smollm_by_id.keys()
)


if qwen_ids != smollm_ids:

    raise ValueError(
        "Qwen and SmolLM IDs do not match."
    )


print(
    "\nID alignment: PASS"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")
print(
    EMBEDDING_MODEL
)


tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

model = AutoModel.from_pretrained(
    EMBEDDING_MODEL
)

model.eval()


print(
    "Embedding model loaded!"
)


# ============================================================
# CREATE EMBEDDING
# ============================================================

def get_embedding(
    text
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )


    token_embeddings = (
        outputs.last_hidden_state
    )


    attention_mask = (
        inputs["attention_mask"]
        .unsqueeze(-1)
        .expand(
            token_embeddings.size()
        )
        .float()
    )


    summed = torch.sum(
        token_embeddings
        * attention_mask,
        dim=1
    )


    counts = torch.clamp(
        attention_mask.sum(
            dim=1
        ),
        min=1e-9
    )


    embedding = (
        summed / counts
    )


    return (
        embedding
        .squeeze()
        .cpu()
        .numpy()
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b
):

    norm_a = np.linalg.norm(
        vector_a
    )

    norm_b = np.linalg.norm(
        vector_b
    )


    if norm_a == 0 or norm_b == 0:

        return 0.0


    return float(

        np.dot(
            vector_a,
            vector_b
        )

        /

        (

            norm_a
            *
            norm_b

        )
    )


# ============================================================
# ANALYZE AGREEMENT
# ============================================================

results = []


print(
    "\nStarting agreement analysis..."
)


sorted_ids = sorted(
    qwen_ids
)


for position, question_id in enumerate(
    sorted_ids
):

    qwen_item = (
        qwen_by_id[
            question_id
        ]
    )


    smollm_item = (
        smollm_by_id[
            question_id
        ]
    )


    question = (
        qwen_item[
            "question"
        ]
    )


    qwen_answer = (
        qwen_item[
            "answer"
        ]
    )


    smollm_answer = (
        smollm_item[
            "answer"
        ]
    )


    print(
        f"Processing "
        f"{position + 1}/"
        f"{len(sorted_ids)}"
    )


    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    qwen_embedding = (
        get_embedding(
            qwen_answer
        )
    )


    smollm_embedding = (
        get_embedding(
            smollm_answer
        )
    )


    # --------------------------------------------------------
    # CALCULATE AGREEMENT
    # --------------------------------------------------------

    agreement_score = (
        cosine_similarity(
            qwen_embedding,
            smollm_embedding
        )
    )


    # --------------------------------------------------------
    # AGREEMENT CATEGORY
    # --------------------------------------------------------

    if agreement_score >= 0.80:

        agreement_category = (
            "HIGH_AGREEMENT"
        )

    elif agreement_score >= 0.60:

        agreement_category = (
            "MEDIUM_AGREEMENT"
        )

    else:

        agreement_category = (
            "LOW_AGREEMENT"
        )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    result = {

        "id":
            question_id,

        "question":
            question,

        "reference_answer":
            qwen_item[
                "reference_answer"
            ],

        "qwen_model":
            qwen_item[
                "model"
            ],

        "qwen_answer":
            qwen_answer,

        "smollm_model":
            smollm_item[
                "model"
            ],

        "smollm_answer":
            smollm_answer,

        "multi_llm_agreement":
            agreement_score,

        "agreement_category":
            agreement_category
    }


    results.append(
        result
    )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
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

scores = np.array(
    [

        item[
            "multi_llm_agreement"
        ]

        for item in results

    ]
)


print(
    "\n" + "=" * 60
)

print(
    "MULTI-LLM AGREEMENT SUMMARY"
)

print(
    "=" * 60
)


print(
    "Minimum:",
    round(
        float(
            scores.min()
        ),
        4
    )
)


print(
    "Mean:",
    round(
        float(
            scores.mean()
        ),
        4
    )
)


print(
    "Median:",
    round(
        float(
            np.median(
                scores
            )
        ),
        4
    )
)


print(
    "Maximum:",
    round(
        float(
            scores.max()
        ),
        4
    )
)


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

categories = {

    "HIGH_AGREEMENT": 0,

    "MEDIUM_AGREEMENT": 0,

    "LOW_AGREEMENT": 0

}


for item in results:

    category = (
        item[
            "agreement_category"
        ]
    )


    categories[
        category
    ] += 1


print(
    "\nAgreement distribution:"
)


for category, count in categories.items():

    percentage = (

        count
        /

        len(results)

        *

        100

    )


    print(

        f"{category}: "
        f"{count} "
        f"({percentage:.1f}%)"

    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MULTI-LLM AGREEMENT ANALYSIS COMPLETE!"
)

print(
    "=" * 60
)


print(
    "\nRecords:",
    len(results)
)


print(
    "Saved to:"
)


print(
    OUTPUT_FILE
)