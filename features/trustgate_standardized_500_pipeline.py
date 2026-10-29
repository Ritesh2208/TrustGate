
import json
import os
import random
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# TRUSTGATE STANDARDIZED DATA PIPELINE
# Phase 1: 500-sample generation + CONSISTENT CORE FEATURES
#
# Core representation:
#   896-D hidden state
#   + confidence
#   + entropy
#   = 898-D TrustGate representation
#
# This script deliberately does NOT fabricate retrieval/verifier
# values. Those will be added only after their extraction procedures
# are standardized across the whole dataset.
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_FILE = "data/TruthfulQA.csv"

OUTPUT_ANSWERS = "outputs/trustgate_standardized_500_answers.json"
OUTPUT_FEATURES = "outputs/trustgate_standardized_500_features.json"

NUM_QUESTIONS = 500
MAX_NEW_TOKENS = 100

# Deterministic generation makes the 500-sample benchmark reproducible.
DO_SAMPLE = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("outputs", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("TRUSTGATE STANDARDIZED 500-SAMPLE PIPELINE")
print("=" * 72)

print("\nLoading TruthfulQA...")
df = pd.read_csv(DATASET_FILE)

required = ["Question", "Best Answer"]
missing = [x for x in required if x not in df.columns]

if missing:
    raise ValueError(f"Missing TruthfulQA columns: {missing}")

if len(df) < NUM_QUESTIONS:
    raise ValueError(
        f"TruthfulQA contains only {len(df)} rows, "
        f"but NUM_QUESTIONS={NUM_QUESTIONS}"
    )

df = df.head(NUM_QUESTIONS).copy()

print("Questions selected:", len(df))


# ============================================================
# LOAD QWEN
# ============================================================

print("\nLoading Qwen...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Device:", device)
print("Qwen loaded.")


# ============================================================
# PROMPT
# ============================================================

def build_prompt(question):
    return f"""You are a helpful assistant.

Question:
{question}

Give the best possible answer.

Answer:
"""


# ============================================================
# GENERATION
# ============================================================

def generate_answer(question):
    prompt = build_prompt(question)

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
            do_sample=DO_SAMPLE,
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
# STANDARDIZED FEATURE EXTRACTION
# ============================================================

def extract_features(question, answer):
    """
    Extract the same three core measurements for every sample.

    Confidence:
        mean probability assigned to the actual next token over
        the generated answer sequence.

    Entropy:
        mean Shannon entropy of the next-token distribution over
        the generated answer sequence.

    Hidden state:
        mean final-layer hidden representation over generated
        answer tokens, resulting in exactly 896 dimensions.
    """

    prompt = build_prompt(question)

    # Reconstruct prompt + generated answer so the feature
    # extraction procedure is identical for every record.
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

    # Need at least one generated token.
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
    # Generated token positions begin at prompt_len.
    first_prediction_index = max(prompt_len - 1, 0)

    generated_token_ids = encoded["input_ids"][0, prompt_len:]

    end_index = first_prediction_index + len(generated_token_ids)

    prediction_logits = logits[
        first_prediction_index:end_index
    ]

    usable = min(
        len(generated_token_ids),
        prediction_logits.shape[0],
    )

    generated_token_ids = generated_token_ids[:usable]
    prediction_logits = prediction_logits[:usable]

    probs = torch.softmax(
        prediction_logits,
        dim=-1,
    )

    actual_probs = probs[
        torch.arange(usable, device=device),
        generated_token_ids,
    ]

    # Shannon entropy.
    entropy_values = -torch.sum(
        probs * torch.log(probs.clamp_min(1e-12)),
        dim=-1,
    )

    confidence = float(
        actual_probs.mean().cpu().item()
    )

    entropy = float(
        entropy_values.mean().cpu().item()
    )

    # Last transformer layer.
    hidden = outputs.hidden_states[-1][0]

    # Mean generated-token representation.
    hidden_generated = hidden[
        prompt_len:prompt_len + usable
    ]

    if hidden_generated.shape[0] == 0:
        hidden_vector = torch.zeros(
            896,
            device=device,
        )
    else:
        hidden_vector = hidden_generated.mean(dim=0)

    hidden_vector = hidden_vector.cpu().numpy().astype(
        np.float32
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
# GENERATE + EXTRACT
# ============================================================

answers = []
features = []

print("\nStarting standardized generation and extraction...")

for position, (_, row) in enumerate(df.iterrows()):

    question = str(row["Question"])
    reference_answer = str(row["Best Answer"])

    sample_id = position

    print("\n" + "=" * 72)
    print(f"Sample {position + 1}/{NUM_QUESTIONS}")
    print("ID:", sample_id)
    print("Question:", question)

    answer = generate_answer(question)

    print("Answer:", answer)

    feature = extract_features(
        question,
        answer,
    )

    answer_record = {
        "id": sample_id,
        "question": question,
        "reference_answer": reference_answer,
        "model": MODEL_NAME,
        "generation": {
            "do_sample": DO_SAMPLE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
        },
        "answer": answer,
    }

    feature_record = {
        "id": sample_id,
        "question": question,
        "answer": answer,
        "reference_answer": reference_answer,
        "model": MODEL_NAME,
        "confidence": feature["confidence"],
        "entropy": feature["entropy"],
        "hidden_state": feature["hidden_state"],
        "hidden_dimension": 896,
        "signal_dimension": 2,
        "fusion_dimension": 898,
        "feature_definition": [
            "mean_actual_token_probability",
            "mean_token_entropy",
            "mean_generated_token_final_hidden_state",
        ],
    }

    answers.append(answer_record)
    features.append(feature_record)

    if (position + 1) % 25 == 0:
        with open(
            OUTPUT_ANSWERS,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                answers,
                f,
                indent=2,
                ensure_ascii=False,
            )

        with open(
            OUTPUT_FEATURES,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                features,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"\nCheckpoint saved at {position + 1} samples."
        )


# ============================================================
# FINAL SAVE
# ============================================================

with open(
    OUTPUT_ANSWERS,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        answers,
        f,
        indent=2,
        ensure_ascii=False,
    )

with open(
    OUTPUT_FEATURES,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        features,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SUMMARY
# ============================================================

conf = np.asarray(
    [x["confidence"] for x in features],
    dtype=float,
)

ent = np.asarray(
    [x["entropy"] for x in features],
    dtype=float,
)

print("\n" + "=" * 72)
print("STANDARDIZED PIPELINE COMPLETE")
print("=" * 72)

print("Samples:", len(features))
print("Hidden dimension:", 896)
print("Core signal dimension:", 2)
print("Core fusion dimension:", 898)

print("\nConfidence:")
print("  Mean:", float(conf.mean()))
print("  Std:", float(conf.std()))

print("\nEntropy:")
print("  Mean:", float(ent.mean()))
print("  Std:", float(ent.std()))

print("\nSaved:")
print(" ", OUTPUT_ANSWERS)
print(" ", OUTPUT_FEATURES)

print("\nNOTE:")
print(
    "Retrieval agreement, verifier score, and self-consistency "
    "are intentionally not fabricated or filled with zeros. "
    "They require separate standardized extraction procedures."
)
