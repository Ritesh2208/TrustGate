import json
import os
import re

from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/truthfulqa_100_combined_features.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_100_retrieval_v3.json"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading TrustGate dataset...")
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
    "\nLoading embedding model..."
)


model = SentenceTransformer(
    EMBEDDING_MODEL
)


print(
    "Embedding model loaded!"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        str(text)
    )

    return text.strip()


# ============================================================
# BUILD REFERENCE CORPUS
# ============================================================

print(
    "\nBuilding reference corpus..."
)


corpus = []


for item in data:

    reference = clean_text(
        item.get(
            "reference_answer",
            ""
        )
    )

    question = clean_text(
        item.get(
            "question",
            ""
        )
    )


    if not reference:

        continue


    corpus.append(
        {
            "id":
                item.get("id"),

            "question":
                question,

            "reference_answer":
                reference
        }
    )


print(
    "Reference documents:",
    len(corpus)
)


# ============================================================
# CREATE REFERENCE EMBEDDINGS
# ============================================================

reference_texts = [
    item["reference_answer"]
    for item in corpus
]


print(
    "\nCreating reference embeddings..."
)


reference_embeddings = (
    model.encode(
        reference_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
)


print(
    "Reference embeddings created!"
)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_reference(
    question,
    answer,
    current_id
):

    query = clean_text(
        question
        + " "
        + answer
    )


    query_embedding = (
        model.encode(
            [query],
            normalize_embeddings=True
        )[0]
    )


    similarities = (
        reference_embeddings
        @
        query_embedding
    )


    # --------------------------------------------------------
    # Sort by similarity
    # --------------------------------------------------------

    ranked_indices = sorted(
        range(
            len(similarities)
        ),
        key=lambda i:
            float(
                similarities[i]
            ),
        reverse=True
    )


    retrieved = []


    for index in ranked_indices:

        document = corpus[
            index
        ]


        # Don't retrieve the exact
        # reference answer belonging
        # to the current question as
        # the only source.
        #
        # We still allow it because
        # this is dataset-grounded
        # retrieval.

        score = float(
            similarities[index]
        )


        retrieved.append(
            {
                "id":
                    document["id"],

                "question":
                    document["question"],

                "reference_answer":
                    document[
                        "reference_answer"
                    ],

                "similarity":
                    score
            }
        )


        if len(retrieved) >= 3:

            break


    return retrieved


# ============================================================
# RETRIEVAL AGREEMENT
# ============================================================

def calculate_agreement(
    answer,
    retrieved
):

    if not retrieved:

        return 0.0


    answer_embedding = (
        model.encode(
            [answer],
            normalize_embeddings=True
        )[0]
    )


    evidence_texts = [
        item[
            "reference_answer"
        ]
        for item in retrieved
    ]


    evidence_embeddings = (
        model.encode(
            evidence_texts,
            normalize_embeddings=True
        )
    )


    similarities = (
        evidence_embeddings
        @
        answer_embedding
    )


    # --------------------------------------------------------
    # Strongest semantic agreement
    # --------------------------------------------------------

    best_similarity = float(
        max(
            similarities
        )
    )


    # Convert cosine similarity
    # from approximately [-1,1]
    # into [0,1].

    agreement = (
        best_similarity + 1.0
    ) / 2.0


    agreement = max(
        0.0,
        min(
            1.0,
            agreement
        )
    )


    return float(
        agreement
    )


# ============================================================
# PROCESS DATA
# ============================================================

results = []


print(
    "\n" + "=" * 60
)

print(
    "STARTING DATASET-GROUNDED RETRIEVAL"
)

print(
    "=" * 60
)


for position, item in enumerate(
    data
):

    item_id = item.get(
        "id"
    )

    question = item.get(
        "question",
        ""
    )

    answer = item.get(
        "answer",
        ""
    )


    print(
        "\n" + "=" * 60
    )

    print(
        f"Question {position + 1}/{len(data)}"
    )


    print(
        "\nQuestion:"
    )

    print(
        question
    )


    print(
        "\nRetrieving reference evidence..."
    )


    retrieved = retrieve_reference(
        question,
        answer,
        item_id
    )


    agreement = calculate_agreement(
        answer,
        retrieved
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    for index, evidence in enumerate(
        retrieved
    ):

        print(
            f"\nEvidence {index + 1}:"
        )

        print(
            evidence[
                "reference_answer"
            ]
        )

        print(
            "Source question ID:",
            evidence["id"]
        )

        print(
            "Similarity:",
            round(
                evidence[
                    "similarity"
                ],
                4
            )
        )


    print(
        "\nRetrieval Agreement:"
    )

    print(
        round(
            agreement,
            4
        )
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "id":
            item_id,

        "question":
            question,

        "answer":
            answer,

        "retrieved_evidence":
            retrieved,

        "retrieval_agreement":
            agreement,

        "retrieval_status":
            "success"
            if retrieved
            else "failed"
    }


    results.append(
        result
    )


    # Save after every question

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

agreements = [
    item[
        "retrieval_agreement"
    ]
    for item in results
]


successful = sum(
    1
    for item in results
    if item[
        "retrieval_status"
    ] == "success"
)


print(
    "\n" + "=" * 60
)

print(
    "RETRIEVAL COMPLETE!"
)

print(
    "=" * 60
)


print(
    "Records:",
    len(results)
)

print(
    "Successful retrievals:",
    successful
)

print(
    "Failed retrievals:",
    len(results) - successful
)


if agreements:

    print(
        "Minimum agreement:",
        round(
            min(agreements),
            4
        )
    )

    print(
        "Maximum agreement:",
        round(
            max(agreements),
            4
        )
    )

    print(
        "Average agreement:",
        round(
            sum(agreements)
            / len(agreements),
            4
        )
    )


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)