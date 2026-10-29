import json
import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "outputs/truthfulqa_100_retrieval_v3.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_100_verified.json"
)

MODEL_NAME = (
    "cross-encoder/nli-deberta-v3-base"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading retrieval dataset...")
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
# LOAD VERIFIER
# ============================================================

print(
    "\nLoading external verifier..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print(
    "External verifier loaded!"
)


# ============================================================
# MODEL LABELS
# ============================================================

print(
    "\nModel labels:"
)

print(
    model.config.id2label
)


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(
    label
):

    label = str(
        label
    ).lower()


    if "entail" in label:

        return "ENTAILMENT"


    if "contrad" in label:

        return "CONTRADICTION"


    if "neutral" in label:

        return "NEUTRAL"


    return "UNKNOWN"


# ============================================================
# VERIFY ONE ANSWER
# ============================================================

def verify_pair(
    answer,
    evidence
):

    premise = str(
        evidence
    )

    hypothesis = str(
        answer
    )


    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]


    predicted_id = int(
        torch.argmax(
            probabilities
        ).item()
    )


    raw_label = (
        model.config.id2label[
            predicted_id
        ]
    )


    label = normalize_label(
        raw_label
    )


    confidence = float(
        probabilities[
            predicted_id
        ].item()
    )


    # --------------------------------------------------------
    # Convert NLI result to TrustGate verifier score
    # --------------------------------------------------------

    if label == "ENTAILMENT":

        score = 1.0

    elif label == "CONTRADICTION":

        score = -1.0

    elif label == "NEUTRAL":

        score = 0.0

    else:

        score = 0.0


    return {
        "raw_label":
            raw_label,

        "label":
            label,

        "score":
            score,

        "confidence":
            confidence
    }


# ============================================================
# PROCESS QUESTIONS
# ============================================================

results = []


print(
    "\n" + "=" * 60
)

print(
    "STARTING EXTERNAL VERIFICATION"
)

print(
    "=" * 60
)


for position, item in enumerate(
    data
):

    question = item.get(
        "question",
        ""
    )

    answer = item.get(
        "answer",
        ""
    )

    evidence_list = item.get(
        "retrieved_evidence",
        []
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
        "\nAnswer:"
    )

    print(
        answer[:500]
    )


    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not evidence_list:

        result = {

            "id":
                item.get("id"),

            "question":
                question,

            "answer":
                answer,

            "retrieval_agreement":
                item.get(
                    "retrieval_agreement",
                    0.0
                ),

            "verifier_label":
                "UNKNOWN",

            "verifier_score":
                0.0,

            "verifier_confidence":
                0.0,

            "verifier_evidence":
                []
        }


        results.append(
            result
        )


        print(
            "\nVerifier: UNKNOWN"
        )

        print(
            "No evidence available."
        )


        continue


    # --------------------------------------------------------
    # Verify each retrieved evidence
    # --------------------------------------------------------

    evidence_results = []


    for evidence_number, evidence in enumerate(
        evidence_list
    ):

        evidence_text = evidence.get(
            "reference_answer",
            evidence.get(
                "text",
                ""
            )
        )


        print(
            f"\nVerifying evidence "
            f"{evidence_number + 1}..."
        )


        verification = verify_pair(
            answer,
            evidence_text
        )


        evidence_result = {

            "source_id":
                evidence.get(
                    "id"
                ),

            "evidence":
                evidence_text,

            "retrieval_similarity":
                evidence.get(
                    "similarity",
                    0.0
                ),

            "raw_label":
                verification[
                    "raw_label"
                ],

            "label":
                verification[
                    "label"
                ],

            "score":
                verification[
                    "score"
                ],

            "confidence":
                verification[
                    "confidence"
                ]
        }


        evidence_results.append(
            evidence_result
        )


        print(
            "Raw label:",
            verification[
                "raw_label"
            ]
        )

        print(
            "Verifier label:",
            verification[
                "label"
            ]
        )

        print(
            "Score:",
            verification[
                "score"
            ]
        )

        print(
            "Confidence:",
            round(
                verification[
                    "confidence"
                ],
                4
            )
        )


    # --------------------------------------------------------
    # Select strongest verification
    # --------------------------------------------------------

    # We use retrieval similarity to determine
    # which evidence is most relevant.

    best_evidence = max(
        evidence_results,
        key=lambda x:
            x[
                "retrieval_similarity"
            ]
    )


    final_label = best_evidence[
        "label"
    ]

    final_score = float(
        best_evidence[
            "score"
        ]
    )

    final_confidence = float(
        best_evidence[
            "confidence"
        ]
    )


    # --------------------------------------------------------
    # Save question result
    # --------------------------------------------------------

    result = {

        "id":
            item.get("id"),

        "question":
            question,

        "answer":
            answer,

        "retrieval_agreement":
            float(
                item.get(
                    "retrieval_agreement",
                    0.0
                )
            ),

        "verifier_label":
            final_label,

        "verifier_score":
            final_score,

        "verifier_confidence":
            final_confidence,

        "verifier_evidence":
            evidence_results
    }


    results.append(
        result
    )


    print(
        "\nFINAL VERIFIER:"
    )

    print(
        final_label
    )

    print(
        "Score:",
        final_score
    )

    print(
        "Confidence:",
        round(
            final_confidence,
            4
        )
    )


# ============================================================
# SAVE
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

label_counts = {

    "ENTAILMENT":
        0,

    "CONTRADICTION":
        0,

    "NEUTRAL":
        0,

    "UNKNOWN":
        0
}


for item in results:

    label = item.get(
        "verifier_label",
        "UNKNOWN"
    )


    if label not in label_counts:

        label = "UNKNOWN"


    label_counts[
        label
    ] += 1


print(
    "\n" + "=" * 60
)

print(
    "EXTERNAL VERIFICATION COMPLETE!"
)

print(
    "=" * 60
)


print(
    "Records:",
    len(results)
)


print(
    "\nVerifier label distribution:"
)


for label, count in label_counts.items():

    print(
        f"{label}: {count}"
    )


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)