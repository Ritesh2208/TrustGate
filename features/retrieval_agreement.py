import json
import os
import re

import pandas as pd
import requests

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "outputs/truthfulqa_with_consistency.json"

DATASET_FILE = "data/TruthfulQA.csv"

OUTPUT_FILE = "outputs/truthfulqa_with_retrieval.json"

TOP_K = 3

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


df = pd.read_csv(
    DATASET_FILE
)


# ============================================================
# CREATE QUESTION → SOURCE LOOKUP
# ============================================================

source_lookup = {}

for _, row in df.iterrows():

    question = str(
        row["Question"]
    ).strip()

    source = str(
        row["Source"]
    ).strip()

    source_lookup[question] = source


print(
    "Questions loaded:",
    len(data)
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded!")


# ============================================================
# DOWNLOAD WEB PAGE
# ============================================================

def download_page(url):

    try:

        headers = {
            "User-Agent":
            "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:

            return ""

        return response.text

    except Exception as e:

        print(
            "Could not retrieve:",
            url
        )

        print(
            "Reason:",
            str(e)
        )

        return ""


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_paragraphs(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    paragraphs = []

    for paragraph in soup.find_all("p"):

        text = paragraph.get_text(
            " ",
            strip=True
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        if len(text) >= 80:

            paragraphs.append(
                text
            )

    return paragraphs


# ============================================================
# SPLIT SOURCE URLS
# ============================================================

def get_urls(source_text):

    urls = re.findall(
        r"https?://[^\s,;]+",
        source_text
    )

    return urls


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

def retrieve_evidence(
    question,
    source_text
):

    urls = get_urls(
        source_text
    )

    if not urls:

        return []


    all_paragraphs = []


    # Try the first two sources
    for url in urls[:2]:

        print(
            "\nRetrieving:",
            url
        )

        html = download_page(
            url
        )

        if not html:

            continue

        paragraphs = extract_paragraphs(
            html
        )

        all_paragraphs.extend(
            paragraphs
        )


    if not all_paragraphs:

        return []


    # Remove duplicate paragraphs
    all_paragraphs = list(
        dict.fromkeys(
            all_paragraphs
        )
    )


    # Encode question and evidence
    question_embedding = (
        embedding_model.encode(
            [question]
        )
    )

    paragraph_embeddings = (
        embedding_model.encode(
            all_paragraphs
        )
    )


    # Similarity between question and passages
    question_scores = cosine_similarity(
        question_embedding,
        paragraph_embeddings
    )[0]


    # Select top-K passages
    ranked_indices = (
        question_scores
        .argsort()[::-1][:TOP_K]
    )


    evidence = []

    for index in ranked_indices:

        evidence.append(
            all_paragraphs[index]
        )


    return evidence


# ============================================================
# CALCULATE RETRIEVAL AGREEMENT
# ============================================================

def calculate_retrieval_agreement(
    answer,
    evidence
):

    if not evidence:

        return None


    answer_embedding = (
        embedding_model.encode(
            [answer]
        )
    )

    evidence_embeddings = (
        embedding_model.encode(
            evidence
        )
    )


    scores = cosine_similarity(
        answer_embedding,
        evidence_embeddings
    )[0]


    # Average similarity with retrieved evidence
    agreement = float(
        scores.mean()
    )


    return agreement


# ============================================================
# PROCESS DATA
# ============================================================

print(
    "\nStarting retrieval agreement..."
)


for index, item in enumerate(data):

    question = item[
        "question"
    ]

    answer = item[
        "answer"
    ]

    print("\n" + "=" * 60)

    print(
        f"Question {index + 1}/{len(data)}"
    )

    print(
        "\nQuestion:"
    )

    print(question)


    source = source_lookup.get(
        question,
        ""
    )


    # --------------------------------------------------------
    # Retrieve evidence
    # --------------------------------------------------------

    evidence = retrieve_evidence(
        question,
        source
    )


    if evidence:

        print(
            "\nRetrieved evidence:"
        )

        for i, passage in enumerate(
            evidence
        ):

            print(
                f"\nEvidence {i + 1}:"
            )

            print(
                passage[:500]
            )


    else:

        print(
            "\nNo evidence retrieved."
        )


    # --------------------------------------------------------
    # Calculate agreement
    # --------------------------------------------------------

    agreement = (
        calculate_retrieval_agreement(
            answer,
            evidence
        )
    )


    if agreement is not None:

        print(
            "\nRetrieval Agreement:"
        )

        print(
            round(
                agreement,
                4
            )
        )

    else:

        print(
            "\nRetrieval Agreement:"
        )

        print(
            "N/A"
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    item[
        "retrieval_agreement"
    ] = agreement

    item[
        "retrieved_evidence"
    ] = evidence


# ============================================================
# SAVE OUTPUT
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
    "RETRIEVAL AGREEMENT COMPLETE!"
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)