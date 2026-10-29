
import json
import os
import random
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# TRUSTGATE — NEW 300-SAMPLE STANDARDIZED EXPANSION
#
# Goal:
#   Existing 200 + NEW 300 = 500 development samples
#
# This script:
#   1. Reads the local TruthfulQA CSV.
#   2. Excludes every question already present in the existing
#      200 labeled datasets.
#   3. Reserves the final 100 unseen questions as a future
#      independent test pool.
#   4. Generates exactly 300 NEW development samples.
#   5. Extracts the standardized 898-D core representation:
#        896-D hidden state + confidence + entropy
#
# It does NOT create labels. Human labels are assigned later.
# It does NOT overwrite old datasets.
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_FILE = "data/TruthfulQA.csv"

ORIGINAL_LABEL_FILE = "outputs/trustgate_human_evaluation_completed.csv"
NEXT_LABEL_FILE = "outputs/trustgate_human_evaluation_next_100_completed.csv"

OUTPUT_NEW_ANSWERS = (
    "outputs/trustgate_standardized_new_300_answers.json"
)

OUTPUT_NEW_FEATURES = (
    "outputs/trustgate_standardized_new_300_features.json"
)

OUTPUT_RESERVED_TEST = (
    "outputs/trustgate_reserved_unseen_100.json"
)

NUM_NEW = 300
RESERVED_TEST = 100
MAX_NEW_TOKENS = 100

DO_SAMPLE = False
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("outputs", exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def normalize_question(q):
    return " ".join(str(q).strip().lower().split())


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_existing_questions_from_labels(label_file):
    if not os.path.exists(label_file):
        return set()

    df = pd.read_csv(label_file)

    if "question" not in df.columns:
        print(
            f"WARNING: {label_file} has no 'question' column. "
            "IDs will be checked separately where possible."
        )
        return set()

    return {
        normalize_question(q)
        for q in df["question"].dropna().tolist()
    }


def get_existing_questions_from_feature_file(path):
    if not os.path.exists(path):
        return set()

    try:
        records = load_json(path)
    except Exception:
        return set()

    return {
        normalize_question(x.get("question", ""))
        for x in records
        if x.get("question")
    }


def build_prompt(question):
    return f"""You are a helpful assistant.

Question:
{question}

Give the best possible answer.

Answer:
"""


# ============================================================
# LOAD EXISTING QUESTIONS
# ============================================================

print("=" * 72)
print("TRUSTGATE — NEW 300 SAMPLE EXPANSION")
print("=" * 72)

print("\nLoading existing question sets...")

used_questions = set()

# Existing human-labeled original 100.
used_questions.update(
    get_existing_questions_from_labels(
        ORIGINAL_LABEL_FILE
    )
)

# Existing human-labeled Next-100.
used_questions.update(
    get_existing_questions_from_labels(
        NEXT_LABEL_FILE
    )
)

# Also inspect feature files in case label CSVs do not contain
# complete question text.
used_questions.update(
    get_existing_questions_from_feature_file(
        "outputs/truthfulqa_100_combined_features.json"
    )
)

used_questions.update(
    get_existing_questions_from_feature_file(
        "outputs/trustgate_next_100_master_901.json"
    )
)

print("Unique previously used questions detected:", len(used_questions))


# ============================================================
# LOAD TRUTHFULQA
# ============================================================

print("\nLoading TruthfulQA...")

df = pd.read_csv(DATASET_FILE)

required = ["Question", "Best Answer"]

missing = [
    x for x in required
    if x not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Total TruthfulQA rows:", len(df))


# ============================================================
# REMOVE PREVIOUSLY USED QUESTIONS
# ============================================================

df["_normalized_question"] = df["Question"].map(
    normalize_question
)

unused = df[
    ~df["_normalized_question"].isin(used_questions)
].copy()

# Remove duplicate questions within the remaining pool.
unused = unused.drop_duplicates(
    subset=["_normalized_question"]
).reset_index(drop=True)

print(
    "Previously used questions removed:",
    len(df) - len(unused)
)

print(
    "Unused unique questions available:",
    len(unused)
)

required_pool = NUM_NEW + RESERVED_TEST

if len(unused) < required_pool:
    raise ValueError(
        f"Need {required_pool} unseen questions "
        f"but only {len(unused)} are available."
    )


# ============================================================
# SELECT NEW DEVELOPMENT + RESERVED TEST
# ============================================================
#
# We use a fixed random seed so the selection is reproducible.
# The final RESERVED_TEST set is not generated or labeled here;
# it is only stored as a question/reference pool for later.
# ============================================================

rng = np.random.default_rng(SEED)

selected_indices = rng.choice(
    len(unused),
    size=required_pool,
    replace=False,
)

selected = unused.iloc[
    selected_indices
].copy().reset_index(drop=True)

new_df = selected.iloc[
    :NUM_NEW
].copy().reset_index(drop=True)

reserved_df = selected.iloc[
    NUM_NEW:NUM_NEW + RESERVED_TEST
].copy().reset_index(drop=True)

print("\nSelected:")
print("  New development questions:", len(new_df))
print("  Reserved future test questions:", len(reserved_df))


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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Device:", device)
print("Qwen loaded.")


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
# STANDARDIZED CORE FEATURES
# ============================================================

def extract_features(question, answer):

    prompt = build_prompt(question)

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

    # logits[t] predicts token[t+1].
    first_prediction_index = max(
        prompt_len - 1,
        0,
    )

    generated_token_ids = (
        encoded["input_ids"][0, prompt_len:]
    )

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
        hidden_vector = (
            hidden_generated.mean(
                dim=0
            )
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
# GENERATE NEW 300
# ============================================================

answers = []
features = []

print("\nStarting generation of NEW 300 samples...")

for position, (_, row) in enumerate(
    new_df.iterrows()
):

    question = str(row["Question"])
    reference_answer = str(row["Best Answer"])

    # New IDs are explicitly separated from the old 0–99 ranges.
    sample_id = 200 + position

    print("\n" + "=" * 72)
    print(
        f"NEW SAMPLE {position + 1}/{NUM_NEW}"
    )
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
        "label_status": "PENDING_HUMAN_ANNOTATION",
    }

    answers.append(answer_record)
    features.append(feature_record)

    # Checkpoint every 25 samples.
    if (position + 1) % 25 == 0:

        with open(
            OUTPUT_NEW_ANSWERS,
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
            OUTPUT_NEW_FEATURES,
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
            f"\nCheckpoint saved: "
            f"{position + 1}/{NUM_NEW}"
        )


# ============================================================
# SAVE RESERVED TEST QUESTIONS
# ============================================================

reserved_records = []

for position, (_, row) in enumerate(
    reserved_df.iterrows()
):

    reserved_records.append(
        {
            "reserved_id": (
                500 + position
            ),
            "question": str(
                row["Question"]
            ),
            "reference_answer": str(
                row["Best Answer"]
            ),
            "model": MODEL_NAME,
            "status": "RESERVED_UNSEEN_TEST",
        }
    )


with open(
    OUTPUT_RESERVED_TEST,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        reserved_records,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL SAVE
# ============================================================

with open(
    OUTPUT_NEW_ANSWERS,
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
    OUTPUT_NEW_FEATURES,
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
# VALIDATION
# ============================================================

generated_questions = {
    normalize_question(x["question"])
    for x in answers
}

if len(generated_questions) != NUM_NEW:
    raise ValueError(
        "Duplicate question detected in new 300."
    )

overlap = (
    generated_questions
    & used_questions
)

if overlap:
    raise ValueError(
        f"ERROR: {len(overlap)} generated questions "
        "overlap with previously used questions."
    )

print("\n" + "=" * 72)
print("NEW 300 STANDARDIZED GENERATION COMPLETE")
print("=" * 72)

print("New samples generated:", len(answers))
print("Reserved unseen test questions:", len(reserved_records))
print("Hidden dimension:", 896)
print("Core fusion dimension:", 898)
print("Previously used questions excluded:", len(used_questions))
print("Duplicate new questions:", 0)
print("Overlap with previous questions:", 0)

print("\nSaved:")
print(" ", OUTPUT_NEW_ANSWERS)
print(" ", OUTPUT_NEW_FEATURES)
print(" ", OUTPUT_RESERVED_TEST)

print("\nNEXT STEP:")
print(
    "Human-label the new 300 samples as "
    "RELIABLE / UNCERTAIN / UNRELIABLE."
)
print(
    "Do not use the reserved 100 questions for "
    "training or model selection."
)
