import json
import os
import re

from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/truthfulqa_next_100_features.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_next_100_retrieval.json"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TRUSTGATE NEXT 100 DATASET-GROUNDED RETRIEVAL")
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


print(
    "First ID:",
    data[0].get("id")
)


print(
    "Last ID:",
    data[-1].get("id")
)


# ============================================================
# LOAD PREVIOUS PROGRESS
# ============================================================

results = []

completed_ids = set()


if os.path.exists(OUTPUT_FILE):

    print(
        "\nExisting output file found."
    )

    try:

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            results = json.load(f)


        completed_ids = {

            item.get("id")

            for item in results

        }


        print(
            "Previously completed:",
            len(completed_ids)
        )


    except Exception as e:

        print(
            "WARNING: Could not load previous output."
        )

        print(
            "Error:",
            e
        )

        results = []

        completed_ids = set()


else:

    print(
        "\nNo previous progress found."
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
    # SORT BY SIMILARITY
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
    # STRONGEST SEMANTIC AGREEMENT
    # --------------------------------------------------------

    best_similarity = float(

        max(
            similarities
        )

    )


    # Convert cosine similarity
    # from approximately [-1, 1]
    # to [0, 1].

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
# SAVE PROGRESS
# ============================================================

def save_progress():

    sorted_results = sorted(

        results,

        key=lambda x: x.get("id")

    )


    with open(

        OUTPUT_FILE,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            sorted_results,

            f,

            indent=4,

            ensure_ascii=False

        )


# ============================================================
# PROCESS DATA
# ============================================================

remaining_data = [

    item

    for item in data

    if item.get("id")

    not in completed_ids

]


print(
    "\n" + "=" * 60
)

print(
    "STARTING NEXT 100 DATASET-GROUNDED RETRIEVAL"
)

print(
    "=" * 60
)


print(
    "\nAlready completed:",
    len(completed_ids)
)


print(
    "Remaining:",
    len(remaining_data)
)


# ============================================================
# PROCESS ONLY UNFINISHED RECORDS
# ============================================================

for position, item in enumerate(

    remaining_data,

    start=1

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

        f"Remaining question "

        f"{position}/{len(remaining_data)}"

    )


    print(
        "Question ID:",
        item_id
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
    # DISPLAY RETRIEVED EVIDENCE
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
    # CREATE RESULT
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


    completed_ids.add(
        item_id
    )


    # --------------------------------------------------------
    # SAVE AFTER EVERY QUESTION
    # --------------------------------------------------------

    save_progress()


    print(
        "\nProgress saved!"
    )


    print(

        "Total completed:",

        len(results)

    )


# ============================================================
# FINAL SUMMARY
# ============================================================

results = sorted(

    results,

    key=lambda x: x.get("id")

)


save_progress()


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
    "NEXT 100 RETRIEVAL COMPLETE!"
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

    len(results)
    -
    successful

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

            /

            len(agreements),

            4

        )

    )


print(
    "\nSaved to:"
)


print(
    OUTPUT_FILE
)