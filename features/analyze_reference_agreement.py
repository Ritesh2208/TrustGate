import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/trustgate_100_master_901.json"
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

OUTPUT_FILE = (
    "outputs/reference_agreement_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading TrustGate master dataset...")
print("=" * 60)

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
# LOAD EMBEDDING MODEL
# ============================================================

print(
    "\nLoading sentence embedding model..."
)

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model loaded!"
)


# ============================================================
# PREPARE TEXT
# ============================================================

questions = []
answers = []
references = []

ids = []

for item in data:

    ids.append(
        item["id"]
    )

    questions.append(
        str(
            item.get(
                "question",
                ""
            )
        )
    )

    answers.append(
        str(
            item.get(
                "answer",
                ""
            )
        )
    )

    references.append(
        str(
            item.get(
                "reference_answer",
                ""
            )
        )
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print(
    "\nEncoding Qwen answers..."
)

answer_embeddings = model.encode(
    answers,
    normalize_embeddings=True,
    show_progress_bar=True
)


print(
    "\nEncoding reference answers..."
)

reference_embeddings = model.encode(
    references,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

semantic_similarity = np.sum(
    answer_embeddings
    * reference_embeddings,
    axis=1
)


# ============================================================
# LEXICAL OVERLAP
# ============================================================

def lexical_overlap(
    answer,
    reference
):

    answer_words = set(
        answer.lower().split()
    )

    reference_words = set(
        reference.lower().split()
    )


    if not reference_words:

        return 0.0


    intersection = (
        answer_words
        &
        reference_words
    )


    return (
        len(intersection)
        /
        len(reference_words)
    )


lexical_scores = []

for answer, reference in zip(
    answers,
    references
):

    lexical_scores.append(
        lexical_overlap(
            answer,
            reference
        )
    )


# ============================================================
# BUILD ANALYSIS TABLE
# ============================================================

rows = []


for i, item in enumerate(
    data
):

    rows.append(
        {
            "id":
                item["id"],

            "question":
                item["question"],

            "confidence":
                item[
                    "confidence"
                ],

            "entropy":
                item[
                    "entropy"
                ],

            "self_consistency":
                item[
                    "self_consistency"
                ],

            "retrieval_agreement":
                item[
                    "retrieval_agreement"
                ],

            "verifier_score":
                item[
                    "verifier_score"
                ],

            "verifier_label":
                item[
                    "verifier_label"
                ],

            "semantic_similarity":
                float(
                    semantic_similarity[i]
                ),

            "lexical_overlap":
                float(
                    lexical_scores[i]
                )
        }
    )


df = pd.DataFrame(
    rows
)


# ============================================================
# STATISTICS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "REFERENCE AGREEMENT STATISTICS"
)

print(
    "=" * 60
)


for column in [
    "semantic_similarity",
    "lexical_overlap"
]:

    values = df[
        column
    ]


    print(
        f"\n{column}"
    )

    print(
        "Minimum:",
        round(
            values.min(),
            4
        )
    )

    print(
        "25th percentile:",
        round(
            values.quantile(
                0.25
            ),
            4
        )
    )

    print(
        "Median:",
        round(
            values.median(),
            4
        )
    )

    print(
        "75th percentile:",
        round(
            values.quantile(
                0.75
            ),
            4
        )
    )

    print(
        "Maximum:",
        round(
            values.max(),
            4
        )
    )

    print(
        "Mean:",
        round(
            values.mean(),
            4
        )
    )


# ============================================================
# PROPOSED DATA-DRIVEN BANDS
# ============================================================

q25 = df[
    "semantic_similarity"
].quantile(
    0.25
)

q75 = df[
    "semantic_similarity"
].quantile(
    0.75
)


print(
    "\n" + "=" * 60
)

print(
    "DATA-DRIVEN SEMANTIC BANDS"
)

print(
    "=" * 60
)

print(
    "Lower boundary:",
    round(
        q25,
        4
    )
)

print(
    "Upper boundary:",
    round(
        q75,
        4
    )
)

print(
    "\nThese are diagnostic boundaries only."
)

print(
    "We will inspect them before assigning labels."
)


# ============================================================
# SHOW LOWEST AGREEMENT ANSWERS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "10 LOWEST REFERENCE AGREEMENTS"
)

print(
    "=" * 60
)


lowest = df.sort_values(
    "semantic_similarity"
).head(
    10
)


for _, row in lowest.iterrows():

    print(
        "\nID:",
        row["id"]
    )

    print(
        "Question:",
        row["question"]
    )

    print(
        "Semantic similarity:",
        round(
            row[
                "semantic_similarity"
            ],
            4
        )
    )

    print(
        "Retrieval:",
        round(
            row[
                "retrieval_agreement"
            ],
            4
        )
    )

    print(
        "Verifier:",
        row[
            "verifier_label"
        ]
    )


# ============================================================
# SHOW HIGHEST AGREEMENT ANSWERS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "10 HIGHEST REFERENCE AGREEMENTS"
)

print(
    "=" * 60
)


highest = df.sort_values(
    "semantic_similarity",
    ascending=False
).head(
    10
)


for _, row in highest.iterrows():

    print(
        "\nID:",
        row["id"]
    )

    print(
        "Question:",
        row["question"]
    )

    print(
        "Semantic similarity:",
        round(
            row[
                "semantic_similarity"
            ],
            4
        )
    )

    print(
        "Retrieval:",
        round(
            row[
                "retrieval_agreement"
            ],
            4
        )
    )

    print(
        "Verifier:",
        row[
            "verifier_label"
        ]
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n" + "=" * 60
)

print(
    "REFERENCE AGREEMENT ANALYSIS COMPLETE!"
)

print(
    "=" * 60
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)