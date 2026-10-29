import json
import os
import re

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
    "outputs/truthfulqa_next_100_retrieval.json"
)

OUTPUT_FILE = (
    "outputs/truthfulqa_next_100_verified.json"
)

MAX_NEW_TOKENS = 100


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
print("TRUSTGATE NEXT 100 VERIFIER")
print("=" * 60)


print("\nLoading retrieval results...")


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
# RESUME SUPPORT
# ============================================================

results = []
completed_ids = set()


if os.path.exists(OUTPUT_FILE):

    print(
        "\nExisting verifier output found."
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
            "Could not load previous progress."
        )

        print(
            "Error:",
            e
        )

        results = []
        completed_ids = set()


else:

    print(
        "\nNo previous verifier progress found."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading verifier model..."
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
    "Verifier model loaded!"
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
# CREATE VERIFIER PROMPT
# ============================================================

def create_verifier_prompt(

    question,

    answer,

    retrieved_evidence

):

    evidence_text = ""


    for i, evidence in enumerate(

        retrieved_evidence

    ):

        evidence_text += (

            f"\nEvidence {i + 1}: "

            f"{evidence.get('reference_answer', '')}"

        )


    prompt = f"""
You are a factual reliability verifier.

Evaluate whether the answer is supported by
the available evidence.

Question:
{question}

Answer:
{answer}

Available Evidence:
{evidence_text}

Rate the answer using ONLY one number between 0 and 1.

0 means:
Completely unreliable or unsupported.

0.5 means:
Partially supported or uncertain.

1 means:
Reliable and strongly supported.

Return ONLY the number.
"""

    return prompt


# ============================================================
# EXTRACT SCORE
# ============================================================

def extract_score(text):

    matches = re.findall(

        r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])",

        str(text)

    )


    if matches:

        try:

            score = float(
                matches[0]
            )

            score = max(
                0.0,
                min(
                    1.0,
                    score
                )
            )

            return score

        except:

            pass


    return 0.5


# ============================================================
# GENERATE VERIFIER SCORE
# ============================================================

def get_verifier_score(

    question,

    answer,

    retrieved_evidence

):

    prompt = create_verifier_prompt(

        question,

        answer,

        retrieved_evidence

    )


    messages = [

        {

            "role": "user",

            "content": prompt

        }

    ]


    text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True

    )


    inputs = tokenizer(

        text,

        return_tensors="pt"

    )


    with torch.no_grad():

        output = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id

        )


    generated_tokens = output[

        0,

        inputs["input_ids"].shape[1]:

    ]


    generated_text = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True

    ).strip()


    score = extract_score(
        generated_text
    )


    return score, generated_text


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
# REMAINING RECORDS
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
    "STARTING VERIFICATION"
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
# PROCESS RECORDS
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


    retrieved_evidence = item.get(

        "retrieved_evidence",

        []

    )


    retrieval_agreement = item.get(

        "retrieval_agreement",

        0.0

    )


    print(
        "\n" + "=" * 60
    )


    print(

        f"Question {position}/{len(remaining_data)}"

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
        "\nRunning verifier..."
    )


    verifier_score, verifier_output = (

        get_verifier_score(

            question,

            answer,

            retrieved_evidence

        )

    )


    print(
        "\nVerifier Output:"
    )


    print(
        verifier_output
    )


    print(
        "\nVerifier Score:"
    )


    print(
        round(
            verifier_score,
            4
        )
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result = {

        "id":
            item_id,

        "question":
            question,

        "answer":
            answer,

        "retrieved_evidence":
            retrieved_evidence,

        "retrieval_agreement":
            retrieval_agreement,

        "verifier_score":
            verifier_score,

        "verifier_output":
            verifier_output

    }


    results.append(
        result
    )


    completed_ids.add(
        item_id
    )


    # ========================================================
    # SAVE AFTER EVERY RECORD
    # ========================================================

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


scores = [

    item["verifier_score"]

    for item in results

]


print(
    "\n" + "=" * 60
)

print(
    "NEXT 100 VERIFICATION COMPLETE!"
)

print(
    "=" * 60
)


print(
    "Records processed:",
    len(results)
)


if scores:

    print(
        "Minimum verifier score:",
        round(
            min(scores),
            4
        )
    )


    print(
        "Maximum verifier score:",
        round(
            max(scores),
            4
        )
    )


    print(
        "Average verifier score:",
        round(
            sum(scores) / len(scores),
            4
        )
    )


print(
    "\nSaved to:"
)


print(
    OUTPUT_FILE
)