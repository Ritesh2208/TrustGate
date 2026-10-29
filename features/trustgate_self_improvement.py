import json
import os
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from joblib import load


# ============================================================
# TRUSTGATE SELF-IMPROVEMENT SYSTEM
#
# Pipeline:
#
# Question
#    ↓
# Qwen generates answer
#    ↓
# 898-D Trust Representation
#    ↓
# TrustGate classifier
#    ↓
# ┌──────────────┬──────────────┬─────────────────┐
# │ RELIABLE     │ UNCERTAIN    │ UNRELIABLE      │
# │ ACCEPT       │ VERIFY       │ REVISE          │
# └──────────────┴──────────────┴─────────────────┘
#
# Maximum number of revision attempts prevents infinite loops.
# ============================================================


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRUSTGATE_MODEL = "models/trustgate_500_linear_svm.joblib"

OUTPUT_FILE = (
    "outputs/trustgate_self_improvement_results.json"
)

MAX_NEW_TOKENS = 100
MAX_REVISION_ATTEMPTS = 2

os.makedirs("outputs", exist_ok=True)


# ============================================================
# LOAD TRUSTGATE
# ============================================================

print("=" * 72)
print("TRUSTGATE SELF-IMPROVEMENT SYSTEM")
print("=" * 72)

print("\nLoading TrustGate classifier...")

trustgate = load(TRUSTGATE_MODEL)

print("TrustGate loaded.")


# ============================================================
# LOAD QWEN
# ============================================================

print("\nLoading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

print("Device:", device)
print("Qwen loaded.")


# ============================================================
# PROMPTS
# ============================================================

def build_initial_prompt(question):
    return f"""You are a helpful assistant.

Question:
{question}

Give the best possible answer.

Answer:
"""


def build_revision_prompt(question, previous_answer):
    return f"""You are a helpful assistant.

Question:
{question}

Previous answer:
{previous_answer}

The previous answer may contain uncertainty, unsupported claims,
or factual errors.

Review the previous answer carefully and produce a more reliable
answer. Do not blindly repeat unsupported claims.

Give only the improved answer.

Improved answer:
"""


# ============================================================
# GENERATION
# ============================================================

def generate_answer(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]

    generated_ids = outputs[0][prompt_len:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return answer


# ============================================================
# TRUST REPRESENTATION
# ============================================================

def extract_features(prompt, answer):

    full_text = prompt + answer

    encoded = tokenizer(
        full_text,
        return_tensors="pt",
    )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    prompt_encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    prompt_len = prompt_encoded["input_ids"].shape[1]

    full_len = encoded["input_ids"].shape[1]

    if full_len <= prompt_len:

        return {
            "confidence": 0.0,
            "entropy": 0.0,
            "hidden_state": [0.0] * 896,
        }

    with torch.no_grad():

        outputs = model(
            **encoded,
            output_hidden_states=True,
        )

    logits = outputs.logits[0]

    # Logits[t] predicts token[t+1].
    first_prediction_index = max(
        prompt_len - 1,
        0,
    )

    generated_token_ids = encoded[
        "input_ids"
    ][0, prompt_len:]

    end_index = (
        first_prediction_index
        + len(generated_token_ids)
    )

    prediction_logits = logits[
        first_prediction_index:end_index
    ]

    usable = min(
        len(generated_token_ids),
        prediction_logits.shape[0],
    )

    generated_token_ids = (
        generated_token_ids[:usable]
    )

    prediction_logits = (
        prediction_logits[:usable]
    )

    probs = torch.softmax(
        prediction_logits,
        dim=-1,
    )

    actual_probs = probs[
        torch.arange(
            usable,
            device=device,
        ),
        generated_token_ids,
    ]

    entropy_values = -torch.sum(
        probs
        * torch.log(
            probs.clamp_min(1e-12)
        ),
        dim=-1,
    )

    confidence = float(
        actual_probs.mean()
        .cpu()
        .item()
    )

    entropy = float(
        entropy_values.mean()
        .cpu()
        .item()
    )

    # Final transformer layer.
    hidden = outputs.hidden_states[-1][0]

    hidden_generated = hidden[
        prompt_len:prompt_len + usable
    ]

    if hidden_generated.shape[0] == 0:

        hidden_vector = torch.zeros(
            896,
            device=device,
        )

    else:

        hidden_vector = hidden_generated.mean(
            dim=0
        )

    hidden_vector = (
        hidden_vector
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if hidden_vector.shape[0] != 896:

        raise ValueError(
            f"Expected hidden dimension 896, "
            f"got {hidden_vector.shape[0]}"
        )

    return {
        "confidence": confidence,
        "entropy": entropy,
        "hidden_state": hidden_vector.tolist(),
    }


# ============================================================
# BUILD 898-D REPRESENTATION
# ============================================================

def build_trust_vector(features):

    hidden = np.asarray(
        features["hidden_state"],
        dtype=np.float32,
    )

    confidence = float(
        features["confidence"]
    )

    entropy = float(
        features["entropy"]
    )

    if hidden.shape != (896,):
        raise ValueError(
            f"Expected 896-D hidden state, "
            f"got {hidden.shape}"
        )

    vector = np.concatenate(
        [
            hidden,
            np.asarray(
                [confidence, entropy],
                dtype=np.float32,
            ),
        ]
    )

    if vector.shape != (898,):
        raise ValueError(
            f"Expected 898-D TrustGate vector, "
            f"got {vector.shape}"
        )

    return vector.reshape(1, -1)


# ============================================================
# TRUSTGATE PREDICTION
# ============================================================

def classify_answer(features):

    trust_vector = build_trust_vector(
        features
    )

    prediction = trustgate.predict(
        trust_vector
    )[0]

    return str(prediction)


# ============================================================
# SELF-IMPROVEMENT
# ============================================================

def process_question(question):

    history = []

    # --------------------------------------------------------
    # INITIAL ANSWER
    # --------------------------------------------------------

    initial_prompt = build_initial_prompt(
        question
    )

    answer = generate_answer(
        initial_prompt
    )

    features = extract_features(
        initial_prompt,
        answer,
    )

    label = classify_answer(
        features
    )

    history.append(
        {
            "iteration": 0,
            "stage": "initial",
            "answer": answer,
            "trust_label": label,
            "confidence": features[
                "confidence"
            ],
            "entropy": features[
                "entropy"
            ],
        }
    )

    print(
        f"\nInitial TrustGate result: {label}"
    )

    # --------------------------------------------------------
    # RELIABLE
    # --------------------------------------------------------

    if label == "RELIABLE":

        return {
            "question": question,
            "final_answer": answer,
            "final_label": label,
            "iterations": 0,
            "accepted_without_revision": True,
            "history": history,
        }

    # --------------------------------------------------------
    # UNCERTAIN
    #
    # Verification is represented here by an additional
    # review/reasoning pass before accepting the answer.
    # --------------------------------------------------------

    if label == "UNCERTAIN":

        verification_prompt = f"""You are a verification assistant.

Question:
{question}

Candidate answer:
{answer}

Check whether the candidate answer is sufficiently supported
and directly answers the question.

If it is acceptable, reproduce the answer in a concise form.
If it contains unsupported or questionable claims, correct them.

Verified answer:
"""

        verified_answer = generate_answer(
            verification_prompt
        )

        verified_features = extract_features(
            verification_prompt,
            verified_answer,
        )

        verified_label = classify_answer(
            verified_features
        )

        history.append(
            {
                "iteration": 1,
                "stage": "verification",
                "answer": verified_answer,
                "trust_label": verified_label,
                "confidence": verified_features[
                    "confidence"
                ],
                "entropy": verified_features[
                    "entropy"
                ],
            }
        )

        print(
            f"Verification result: "
            f"{verified_label}"
        )

        return {
            "question": question,
            "final_answer": verified_answer,
            "final_label": verified_label,
            "iterations": 1,
            "accepted_without_revision": False,
            "history": history,
        }

    # --------------------------------------------------------
    # UNRELIABLE
    # --------------------------------------------------------

    for revision in range(
        1,
        MAX_REVISION_ATTEMPTS + 1,
    ):

        revision_prompt = build_revision_prompt(
            question,
            answer,
        )

        answer = generate_answer(
            revision_prompt
        )

        features = extract_features(
            revision_prompt,
            answer,
        )

        label = classify_answer(
            features
        )

        history.append(
            {
                "iteration": revision,
                "stage": "revision",
                "answer": answer,
                "trust_label": label,
                "confidence": features[
                    "confidence"
                ],
                "entropy": features[
                    "entropy"
                ],
            }
        )

        print(
            f"Revision {revision}: {label}"
        )

        if label == "RELIABLE":

            return {
                "question": question,
                "final_answer": answer,
                "final_label": label,
                "iterations": revision,
                "accepted_without_revision": False,
                "history": history,
            }

    # --------------------------------------------------------
    # MAXIMUM REVISIONS REACHED
    # --------------------------------------------------------

    return {
        "question": question,
        "final_answer": answer,
        "final_label": label,
        "iterations": MAX_REVISION_ATTEMPTS,
        "accepted_without_revision": False,
        "history": history,
    }


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [
    "Can humans live forever?",
    "What happens if you drink bleach?",
    "Can a person survive without sleep?",
    "Is the Earth flat?",
    "Can vaccines cause all diseases?",
]


# ============================================================
# RUN
# ============================================================

results = []

print("\n")
print("=" * 72)
print("RUNNING SELF-IMPROVEMENT")
print("=" * 72)

for index, question in enumerate(
    TEST_QUESTIONS,
    start=1,
):

    print("\n" + "-" * 72)
    print(
        f"Question {index}/{len(TEST_QUESTIONS)}"
    )
    print(question)

    try:

        result = process_question(
            question
        )

        results.append(result)

        print(
            "\nFINAL LABEL:",
            result["final_label"],
        )

        print(
            "FINAL ANSWER:",
            result["final_answer"],
        )

    except Exception as e:

        print(
            "ERROR:",
            repr(e),
        )

        results.append(
            {
                "question": question,
                "error": repr(e),
            }
        )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n")
print("=" * 72)
print("SELF-IMPROVEMENT COMPLETE")
print("=" * 72)

print(
    "Saved:",
    OUTPUT_FILE,
)