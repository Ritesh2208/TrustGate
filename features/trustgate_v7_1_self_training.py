import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import torch
import joblib

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

from external_verifier import ExternalVerifier


# ============================================================
# TRUSTGATE V7.1
# VERIFIED SELF-TRAINING
# ============================================================
#
# 500 human-labeled TrustGate examples
#                 +
#       externally retrieved evidence
#                 +
#       evidence interpretation
#                 ↓
#          Trust Experiences
#                 ↓
#            TrustGate V2
#
# Automatically generated labels are NOT human labels.
#
# TruthfulQA reference answers are NOT used to create
# pseudo-labels.
#
# ============================================================


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_DATA = (
    "outputs/trustgate_standardized_500_labeled.json"
)

HELDOUT_DATA = (
    "outputs/trustgate_heldout_290.json"
)

V1_MODEL = (
    "models/trustgate_500_22d_linear_svm.joblib"
)

V2_MODEL = (
    "models/trustgate_v2_self_trained.joblib"
)

EXPERIENCE_FILE = (
    "outputs/trustgate_v7_1_verified_experiences.json"
)

SUMMARY_FILE = (
    "outputs/trustgate_v7_1_summary.json"
)


# ------------------------------------------------------------
# Development/evaluation split
# ------------------------------------------------------------

DEVELOPMENT_SIZE = 40


# ------------------------------------------------------------
# Evidence acceptance
#
# We deliberately allow EVIDENCE_FOUND to proceed to the
# evidence judge. It is NOT automatically treated as truth.
# ------------------------------------------------------------

MIN_EVIDENCE_SENTENCES = 3
MIN_SOURCES = 2


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(path, data):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    temp = path + ".tmp"

    with open(
        temp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    os.replace(
        temp,
        path
    )


def clean_text(text):

    if not text:
        return ""

    text = str(text)

    markers = [
        "\nQuestion:",
        "\nAnswer:",
        "\nEvidence:",
        "\nCritique:",
        "\nYou are an AI",
        "\nA)",
        "\nB)",
    ]

    lower = text.lower()

    positions = []

    for marker in markers:

        pos = lower.find(
            marker.lower()
        )

        if pos > 0:
            positions.append(pos)

    if positions:

        text = text[
            :min(positions)
        ]

    return " ".join(
        text.split()
    ).strip()


# ============================================================
# LOAD QWEN
# ============================================================

print()
print("=" * 70)
print("TRUSTGATE V7.1")
print("VERIFIED SELF-TRAINING")
print("=" * 70)

print()
print("Loading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto"
)

model.eval()

device = next(
    model.parameters()
).device

print(
    "Device:",
    device
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading 500 human-labeled data...")

base_data = load_json(
    BASE_DATA
)

assert len(base_data) == 500

print(
    "Human examples:",
    len(base_data)
)


print()
print("Loading held-out questions...")

heldout = load_json(
    HELDOUT_DATA
)

assert len(heldout) == 290

development = heldout[
    :DEVELOPMENT_SIZE
]

evaluation = heldout[
    DEVELOPMENT_SIZE:
]

print(
    "Development:",
    len(development)
)

print(
    "Untouched evaluation:",
    len(evaluation)
)


# ============================================================
# EXTERNAL VERIFIER
# ============================================================

print()
print("Loading external verifier...")

verifier = ExternalVerifier(
    max_sources=3,
    timeout=10
)

print(
    "External verifier ready."
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    prompt = (
        "Answer the question directly and concisely.\n"
        "Return only the answer.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            pad_token_id=(
                tokenizer.eos_token_id
            )
        )

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return clean_text(answer)


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

def extract_evidence(result):

    evidence = []

    # --------------------------------------------------------
    # Supporting sentences
    # --------------------------------------------------------

    for sentence in result.get(
        "supporting_sentences",
        []
    ):

        if isinstance(
            sentence,
            str
        ):

            sentence = sentence.strip()

            if sentence:
                evidence.append(
                    sentence
                )

    # --------------------------------------------------------
    # Contradicting sentences
    # --------------------------------------------------------

    for sentence in result.get(
        "contradicting_sentences",
        []
    ):

        if isinstance(
            sentence,
            str
        ):

            sentence = sentence.strip()

            if sentence:
                evidence.append(
                    sentence
                )

    # --------------------------------------------------------
    # Source sentences
    # --------------------------------------------------------

    for source in result.get(
        "sources",
        []
    ):

        if not isinstance(
            source,
            dict
        ):
            continue

        sentences = source.get(
            "sentences",
            []
        )

        if not isinstance(
            sentences,
            list
        ):
            continue

        for sentence in sentences:

            if isinstance(
                sentence,
                str
            ):

                sentence = sentence.strip()

                if sentence:
                    evidence.append(
                        sentence
                    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique = []

    seen = set()

    for item in evidence:

        key = item.lower()

        if key in seen:
            continue

        seen.add(key)

        unique.append(item)

    return unique


# ============================================================
# EVIDENCE JUDGE
# ============================================================

def judge_evidence(
    question,
    answer,
    evidence
):

    if not evidence:

        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "reason": "No evidence available."
        }

    evidence_text = "\n".join(
        [
            f"- {x}"
            for x in evidence[:15]
        ]
    )

    prompt = (
        "You are an evidence consistency judge.\n\n"
        "Determine whether the retrieved evidence "
        "supports the central factual claim in the answer.\n\n"
        "Return exactly one of:\n"
        "SUPPORTED\n"
        "CONTRADICTED\n"
        "UNCERTAIN\n\n"
        "Do not judge whether the evidence itself is "
        "good because of writing style. Compare the "
        "answer's central claim with the evidence.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "VERDICT:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            num_beams=1,
            pad_token_id=(
                tokenizer.eos_token_id
            )
        )

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    raw = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    raw_upper = raw.upper()

    # --------------------------------------------------------
    # Parse first valid verdict
    # --------------------------------------------------------

    if "CONTRADICTED" in raw_upper:

        verdict = "CONTRADICTED"

    elif "SUPPORTED" in raw_upper:

        verdict = "SUPPORTED"

    elif "UNCERTAIN" in raw_upper:

        verdict = "UNCERTAIN"

    else:

        verdict = "UNCERTAIN"

    # --------------------------------------------------------
    # This is deliberately a categorical judgment.
    # We don't pretend it is a calibrated probability.
    # --------------------------------------------------------

    if verdict in [
        "SUPPORTED",
        "CONTRADICTED"
    ]:

        confidence = 1.0

    else:

        confidence = 0.0

    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw_output": raw,
        "reason":
            "Qwen evidence consistency judgment."
    }


# ============================================================
# BUILD TRUST REPRESENTATION
# ============================================================

def analyze_answer(answer):

    encoded = tokenizer(
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    input_ids = encoded[
        "input_ids"
    ].to(device)

    attention_mask = encoded[
        "attention_mask"
    ].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

    hidden = outputs.hidden_states[-1]

    hidden_vector = (
        hidden[0]
        .mean(dim=0)
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if hidden_vector.shape[0] != 896:

        raise ValueError(
            "Expected 896-D hidden vector, got "
            f"{hidden_vector.shape[0]}"
        )

    logits = outputs.logits[0]

    if logits.shape[0] < 2:

        confidence = 0.0
        entropy = 0.0

    else:

        probs = torch.softmax(
            logits[:-1].float(),
            dim=-1
        )

        target_tokens = input_ids[
            0,
            1:
        ]

        positions = torch.arange(
            target_tokens.shape[0],
            device=device
        )

        actual_probs = probs[
            positions,
            target_tokens
        ]

        confidence = float(
            actual_probs.mean()
            .float()
            .cpu()
        )

        token_entropy = -(
            probs
            * torch.log(
                probs + 1e-12
            )
        ).sum(dim=-1)

        entropy = float(
            token_entropy.mean()
            .float()
            .cpu()
        )

    return {
        "hidden_state":
            hidden_vector,

        "confidence":
            confidence,

        "entropy":
            entropy
    }


# ============================================================
# CREATE VERIFIED EXPERIENCES
# ============================================================

experiences = []

stats = {
    "total": 0,
    "supported": 0,
    "contradicted": 0,
    "uncertain": 0,
    "approved": 0,
    "rejected": 0,
}


print()
print("=" * 70)
print("GENERATING VERIFIED EXPERIENCES")
print("=" * 70)


for index, item in enumerate(
    development
):

    stats["total"] += 1

    question = item[
        "question"
    ]

    print()
    print(
        f"[{index + 1}/{len(development)}]"
    )

    print(
        "Question:",
        question
    )

    try:

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = generate_answer(
            question
        )

        print(
            "Answer:",
            answer
        )

        # ----------------------------------------------------
        # Retrieve external evidence
        # ----------------------------------------------------

        verification = verifier.verify(
            question,
            answer
        )

        status = verification.get(
            "verification_status",
            "UNKNOWN"
        )

        evidence = extract_evidence(
            verification
        )

        source_count = int(
            verification.get(
                "source_count",
                0
            )
        )

        evidence_count = int(
            verification.get(
                "evidence_sentence_count",
                0
            )
        )

        print(
            "Retriever status:",
            status
        )

        print(
            "Sources:",
            source_count
        )

        print(
            "Evidence:",
            evidence_count
        )

        # ----------------------------------------------------
        # Evidence requirement
        # ----------------------------------------------------

        if (
            source_count < MIN_SOURCES
            or
            evidence_count
            < MIN_EVIDENCE_SENTENCES
        ):

            print(
                "→ REJECTED: insufficient evidence."
            )

            stats["rejected"] += 1

            continue

        # ----------------------------------------------------
        # Judge evidence
        # ----------------------------------------------------

        judgment = judge_evidence(
            question,
            answer,
            evidence
        )

        verdict = judgment[
            "verdict"
        ]

        print(
            "Evidence judge:",
            verdict
        )

        if verdict == "SUPPORTED":

            stats["supported"] += 1

        elif verdict == "CONTRADICTED":

            stats["contradicted"] += 1

        else:

            stats["uncertain"] += 1

        # ----------------------------------------------------
        # Only clear verdicts become experiences
        # ----------------------------------------------------

        if verdict == "UNCERTAIN":

            print(
                "→ REJECTED: uncertain."
            )

            stats["rejected"] += 1

            continue

        # ----------------------------------------------------
        # Convert evidence verdict into TrustGate label
        # ----------------------------------------------------

        if verdict == "SUPPORTED":

            pseudo_label = "RELIABLE"

        else:

            pseudo_label = "UNRELIABLE"

        # ----------------------------------------------------
        # Trust representation
        # ----------------------------------------------------

        representation = analyze_answer(
            answer
        )

        experience = {

            "experience_id":
                len(experiences) + 1,

            "question":
                question,

            "answer":
                answer,

            "hidden_state":
                representation[
                    "hidden_state"
                ].tolist(),

            "confidence":
                representation[
                    "confidence"
                ],

            "entropy":
                representation[
                    "entropy"
                ],

            "pseudo_label":
                pseudo_label,

            "label_source":
                "EXTERNAL_EVIDENCE_PLUS_EVIDENCE_JUDGE",

            "retriever_status":
                status,

            "judge_verdict":
                verdict,

            "judge_output":
                judgment,

            "support_score":
                verification.get(
                    "support_score",
                    0.0
                ),

            "contradiction_score":
                verification.get(
                    "contradiction_score",
                    0.0
                ),

            "source_count":
                source_count,

            "evidence_sentence_count":
                evidence_count,

            "evidence":
                evidence[:15],

            "quality_status":
                "APPROVED",

            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat()

        }

        experiences.append(
            experience
        )

        stats["approved"] += 1

        print(
            "→ APPROVED:",
            pseudo_label
        )

    except Exception as exc:

        print(
            "→ ERROR:",
            repr(exc)
        )

        stats["rejected"] += 1


# ============================================================
# SAVE EXPERIENCES
# ============================================================

save_json(
    EXPERIENCE_FILE,
    experiences
)

print()
print("=" * 70)
print("EXPERIENCE GENERATION COMPLETE")
print("=" * 70)

print(
    "Total:",
    stats["total"]
)

print(
    "Supported:",
    stats["supported"]
)

print(
    "Contradicted:",
    stats["contradicted"]
)

print(
    "Uncertain:",
    stats["uncertain"]
)

print(
    "Approved:",
    stats["approved"]
)

print(
    "Rejected:",
    stats["rejected"]
)

print(
    "Saved:",
    EXPERIENCE_FILE
)


# ============================================================
# STOP IF NO EXPERIENCES
# ============================================================

if len(experiences) == 0:

    print()
    print(
        "No verified experiences were produced."
    )

    print(
        "TrustGate V2 will not be trained."
    )

    raise SystemExit(0)


# ============================================================
# BUILD SELF-TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("BUILDING TRUSTGATE V2")
print("=" * 70)


base_hidden = np.array(
    [
        item["hidden_state"]
        for item in base_data
    ],
    dtype=np.float32
)

base_confidence = np.array(
    [
        item["confidence"]
        for item in base_data
    ],
    dtype=np.float32
)

base_entropy = np.array(
    [
        item["entropy"]
        for item in base_data
    ],
    dtype=np.float32
)

base_labels = np.array(
    [
        item["label"]
        for item in base_data
    ]
)


experience_hidden = np.array(
    [
        item["hidden_state"]
        for item in experiences
    ],
    dtype=np.float32
)

experience_confidence = np.array(
    [
        item["confidence"]
        for item in experiences
    ],
    dtype=np.float32
)

experience_entropy = np.array(
    [
        item["entropy"]
        for item in experiences
    ],
    dtype=np.float32
)

experience_labels = np.array(
    [
        item["pseudo_label"]
        for item in experiences
    ])


# ============================================================
# COMBINE
# ============================================================

all_hidden = np.vstack([
    base_hidden,
    experience_hidden
])

all_confidence = np.concatenate([
    base_confidence,
    experience_confidence
])

all_entropy = np.concatenate([
    base_entropy,
    experience_entropy
])

all_labels = np.concatenate([
    base_labels,
    experience_labels
])


print(
    "Human examples:",
    len(base_data)
)

print(
    "Verified experiences:",
    len(experiences)
)

print(
    "Combined training examples:",
    len(all_labels)
)


# ============================================================
# FIT 22-D REPRESENTATION
# ============================================================

hidden_scaler = StandardScaler()

hidden_scaled = (
    hidden_scaler.fit_transform(
        all_hidden
    )
)

pca = PCA(
    n_components=20,
    random_state=42
)

hidden_20 = (
    pca.fit_transform(
        hidden_scaled
    )
)

X = np.hstack([
    hidden_20,
    all_confidence.reshape(-1, 1),
    all_entropy.reshape(-1, 1)
])

final_scaler = StandardScaler()

X_scaled = (
    final_scaler.fit_transform(
        X
    )
)


# ============================================================
# TRAIN V2
# ============================================================

classifier = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

classifier.fit(
    X_scaled,
    all_labels
)


# ============================================================
# TRAINING METRICS
# ============================================================

predictions = classifier.predict(
    X_scaled
)

accuracy = accuracy_score(
    all_labels,
    predictions
)

balanced = balanced_accuracy_score(
    all_labels,
    predictions
)

macro_f1 = f1_score(
    all_labels,
    predictions,
    average="macro"
)

print()
print(
    "V2 training accuracy:",
    round(accuracy, 4)
)

print(
    "V2 training balanced accuracy:",
    round(balanced, 4)
)

print(
    "V2 training macro F1:",
    round(macro_f1, 4)
)

print()
print(
    classification_report(
        all_labels,
        predictions,
        labels=[
            "RELIABLE",
            "UNCERTAIN",
            "UNRELIABLE"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE V2
# ============================================================

bundle = {

    "hidden_scaler":
        hidden_scaler,

    "pca":
        pca,

    "final_scaler":
        final_scaler,

    "classifier":
        classifier,

    "input_hidden_dimension":
        896,

    "pca_dimension":
        20,

    "final_dimension":
        22,

    "signal_features":
        [
            "confidence",
            "entropy"
        ],

    "human_training_examples":
        len(base_data),

    "verified_experiences":
        len(experiences),

    "training_type":
        "SELF_TRAINING",

    "label_source":
        "HUMAN_LABELS_PLUS_EXTERNAL_EVIDENCE_PSEUDO_LABELS",

    "created_at":
        datetime.now(
            timezone.utc
        ).isoformat()

}

os.makedirs(
    os.path.dirname(
        V2_MODEL
    ),
    exist_ok=True
)

joblib.dump(
    bundle,
    V2_MODEL
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "model_v1":
        V1_MODEL,

    "model_v2":
        V2_MODEL,

    "human_examples":
        len(base_data),

    "development_questions":
        len(development),

    "untouched_evaluation_questions":
        len(evaluation),

    "verified_experiences":
        len(experiences),

    "experience_statistics":
        stats,

    "v2_training_examples":
        len(all_labels),

    "v2_training_accuracy":
        float(accuracy),

    "v2_training_balanced_accuracy":
        float(balanced),

    "v2_training_macro_f1":
        float(macro_f1),

    "created_at":
        datetime.now(
            timezone.utc
        ).isoformat()

}

save_json(
    SUMMARY_FILE,
    summary
)


print()
print("=" * 70)
print("TRUSTGATE V7.1 COMPLETE")
print("=" * 70)

print(
    "Verified experiences:",
    len(experiences)
)

print(
    "TrustGate V2:",
    V2_MODEL
)

print(
    "Summary:",
    SUMMARY_FILE
)

print("=" * 70)