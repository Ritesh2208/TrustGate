import os
import json
import tempfile
import re

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from features.trustgate_controller import TrustGateController
from features.external_verifier import ExternalVerifier


# ============================================================
# TRUSTGATE V13
# FINAL CONTROLLED SELF-IMPROVEMENT EXPERIMENT
# ============================================================
#
# Architecture:
#
# Question
#    |
#    v
# Qwen initial answer A0
#    |
#    v
# TrustGate
#    |
#    v
# External evidence
#    |
#    v
# Generate A1 from QUESTION + EVIDENCE
#    |
#    v
# If A1 == A0 -> one correction retry
#    |
#    v
# Verify A1
#    |
#    +----------------------+
#    |                      |
# SUPPORTED            NOT SUPPORTED
#    |                      |
# ACCEPT                 ROLLBACK
#
# Controlled expected answers are used ONLY
# for final evaluation of the demonstration.
# They are NEVER supplied to the model.
# ============================================================


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_PATH = (
    "outputs/trustgate_v13_self_improvement_results.json"
)

CHECKPOINT_PATH = (
    "outputs/trustgate_v13_self_improvement_checkpoint.json"
)

MAX_NEW_TOKENS = 60

SUPPORT_THRESHOLD = 0.45


# ============================================================
# CONTROLLED TEST CASES
# ============================================================
#
# These initial answers are deliberately incorrect.
#
# expected_answer is ONLY an evaluation oracle.
# It is NOT provided to Qwen during correction.
# ============================================================

CASES = [

    {
        "id": 1,
        "question": "What is the capital of France?",
        "initial_answer": (
            "London is the capital of France."
        ),
        "expected_answer": (
            "Paris is the capital of France."
        ),
    },

    {
        "id": 2,
        "question": (
            "What gas do humans need to breathe "
            "to survive?"
        ),
        "initial_answer": (
            "Humans need carbon dioxide to breathe "
            "and survive."
        ),
        "expected_answer": (
            "Humans need oxygen to breathe and survive."
        ),
    },

    {
        "id": 3,
        "question": (
            "What planet do humans live on?"
        ),
        "initial_answer": (
            "Humans live on Mars."
        ),
        "expected_answer": (
            "Humans live on Earth."
        ),
    },

    {
        "id": 4,
        "question": (
            "What is the largest ocean on Earth?"
        ),
        "initial_answer": (
            "The Atlantic Ocean is the largest "
            "ocean on Earth."
        ),
        "expected_answer": (
            "The Pacific Ocean is the largest "
            "ocean on Earth."
        ),
    },

    {
        "id": 5,
        "question": (
            "What is the chemical formula for water?"
        ),
        "initial_answer": (
            "The chemical formula for water is CO2."
        ),
        "expected_answer": (
            "The chemical formula for water is H2O."
        ),
    },
]


# ============================================================
# ATOMIC JSON SAVE
# ============================================================

def atomic_json_save(data, path):

    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    fd, temp_path = tempfile.mkstemp(
        prefix=".trustgate_v13_",
        suffix=".json",
        dir=directory
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temp_path,
            path
        )

    except Exception:

        try:
            os.remove(temp_path)
        except OSError:
            pass

        raise


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("TRUSTGATE V13")
print("FINAL CONTROLLED SELF-IMPROVEMENT EXPERIMENT")
print("=" * 72)

print()
print("Cases:", len(CASES))
print("Checkpointing: ENABLED")
print(
    "Support threshold:",
    SUPPORT_THRESHOLD
)
print(
    "Correction retry: ENABLED"
)
print(
    "Controlled oracle: EVALUATION ONLY"
)


# ============================================================
# LOAD QWEN
# ============================================================

print()
print("Loading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)

model.eval()

print("Qwen loaded.")


# ============================================================
# LOAD TRUSTGATE
# ============================================================

print()
print("Loading TrustGate...")

controller = TrustGateController()

print("TrustGate loaded.")


# ============================================================
# LOAD EXTERNAL VERIFIER
# ============================================================

print()
print("Loading external verifier...")

verifier = ExternalVerifier()

print("External verifier loaded.")


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(instruction):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual answering assistant. "
                "Answer the question directly. "
                "Use evidence when it is provided. "
                "Do not invent facts. "
                "Do not mention the correction process. "
                "Keep the answer concise."
            ),
        },
        {
            "role": "user",
            "content": instruction,
        },
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

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    generated = output[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claim(question, answer):

    prompt = f"""
Question:

{question}

Answer:

{answer}

Extract ONLY the single central factual claim.

Rules:

- Preserve negation.
- Preserve names.
- Preserve numbers.
- Do not add information.
- Do not explain.
- Return one short factual sentence.

Return only the claim.
"""

    claim = generate_text(prompt)

    claim = re.sub(
        r"^(Claim|Central claim)\s*:\s*",
        "",
        claim,
        flags=re.IGNORECASE
    )

    lines = [
        line.strip()
        for line in claim.splitlines()
        if line.strip()
    ]

    if lines:
        claim = lines[0]

    return claim.strip()


# ============================================================
# TRUST FEATURE EXTRACTION
# ============================================================

def get_trust_features(question, answer):

    system_text = (
        "You are a factual question answering assistant."
    )

    prompt = (
        system_text
        + "\n\n"
        + question
        + "\n\n"
    )

    full_text = prompt + answer

    encoded = tokenizer(
        full_text,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(model.device)
        for key, value in encoded.items()
    }

    prompt_encoded = tokenizer(
        prompt,
        return_tensors="pt"
    )

    prompt_len = (
        prompt_encoded["input_ids"].shape[1]
    )

    full_len = (
        encoded["input_ids"].shape[1]
    )

    if full_len <= prompt_len:

        return {
            "hidden_state": [0.0] * 896,
            "confidence": 0.0,
            "entropy": 0.0,
        }

    with torch.no_grad():

        outputs = model(
            **encoded,
            output_hidden_states=True
        )

    logits = outputs.logits[0].float()

    first_prediction_index = max(
        prompt_len - 1,
        0
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
        prediction_logits.shape[0]
    )

    generated_token_ids = (
        generated_token_ids[:usable]
    )

    prediction_logits = (
        prediction_logits[:usable]
    )

    if usable == 0:

        return {
            "hidden_state": [0.0] * 896,
            "confidence": 0.0,
            "entropy": 0.0,
        }

    probabilities = torch.softmax(
        prediction_logits,
        dim=-1
    )

    actual_probabilities = probabilities[
        torch.arange(
            usable,
            device=model.device
        ),
        generated_token_ids
    ]

    entropy_values = -torch.sum(
        probabilities
        * torch.log(
            probabilities.clamp_min(1e-12)
        ),
        dim=-1
    )

    confidence = float(
        actual_probabilities.mean()
        .detach()
        .cpu()
        .item()
    )

    entropy = float(
        entropy_values.mean()
        .detach()
        .cpu()
        .item()
    )

    hidden = (
        outputs.hidden_states[-1][0]
        .float()
    )

    hidden_generated = hidden[
        prompt_len:
        prompt_len + usable
    ]

    if hidden_generated.shape[0] == 0:

        hidden_vector = torch.zeros(
            896,
            device=model.device
        )

    else:

        hidden_vector = (
            hidden_generated.mean(
                dim=0
            )
        )

    hidden_vector = (
        hidden_vector
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if hidden_vector.shape[0] != 896:

        raise ValueError(
            "Expected hidden state dimension 896, "
            f"got {hidden_vector.shape[0]}"
        )

    return {
        "hidden_state": hidden_vector.tolist(),
        "confidence": confidence,
        "entropy": entropy,
    }


# ============================================================
# TRUSTGATE CLASSIFICATION
# ============================================================

def classify_answer(question, answer):

    features = get_trust_features(
        question,
        answer
    )

    label = controller.classify(
        features["hidden_state"],
        features["confidence"],
        features["entropy"]
    )

    return {
        "label": label,
        "confidence": features["confidence"],
        "entropy": features["entropy"],
    }


# ============================================================
# EXTERNAL VERIFICATION
# ============================================================

def verify_answer(question, answer):

    claim = extract_claim(
        question,
        answer
    )

    verification = verifier.verify(
        question,
        answer
    )

    return {
        "claim": claim,
        "verification": verification,
    }


# ============================================================
# EXTRACT EVIDENCE TEXT
# ============================================================

def build_evidence_text(verification):

    sources = verification.get(
        "sources",
        []
    )

    evidence_blocks = []

    for index, source in enumerate(
        sources,
        start=1
    ):

        title = source.get(
            "title",
            f"Source {index}"
        )

        sentences = source.get(
            "sentences",
            []
        )

        evidence = " ".join(
            str(item)
            for item in sentences
        )

        evidence = evidence[:2500]

        evidence_blocks.append(
            f"SOURCE {index}: {title}\n"
            f"{evidence}"
        )

    return "\n\n".join(
        evidence_blocks
    )


# ============================================================
# V13 QUESTION-FIRST CORRECTION
# ============================================================
#
# IMPORTANT:
#
# We intentionally DO NOT give A0 as the primary answer
# to reproduce.
#
# Instead, Qwen is asked to answer the ORIGINAL QUESTION
# using retrieved evidence.
#
# This reduces anchoring on a wrong A0.
# ============================================================

def generate_correction(
    question,
    old_answer,
    verification,
    retry=False
):

    evidence_text = build_evidence_text(
        verification
    )

    if retry:

        instruction = f"""
The previous correction attempt was not acceptable.

Answer the original question again using the
retrieved evidence below.

ORIGINAL QUESTION:

{question}

RETRIEVED EVIDENCE:

{evidence_text}

The previous answer was:

{old_answer}

IMPORTANT:

- Do not repeat the previous answer if the evidence
  indicates it is incorrect.
- Determine the factual answer from the evidence.
- Answer the ORIGINAL QUESTION.
- Use the strongest directly relevant evidence.
- Do not mention this instruction.
- Do not mention the previous answer.
- Do not explain the correction process.
- Give one short factual answer.

Return ONLY the answer.
"""

    else:

        instruction = f"""
Answer the following question using the retrieved
evidence.

ORIGINAL QUESTION:

{question}

RETRIEVED EVIDENCE:

{evidence_text}

The current answer is known to be unreliable,
so independently determine the correct answer
from the question and evidence.

Rules:

- Answer the ORIGINAL QUESTION.
- Do not copy the unreliable answer.
- Do not assume the current answer is correct.
- Use directly relevant evidence.
- Preserve important names and facts.
- Do not mention sources.
- Do not mention the correction process.
- Keep the answer short.
- Return ONLY the corrected answer.

Answer:
"""

    return generate_text(
        instruction
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SAME-ANSWER DETECTION
# ============================================================

def answers_effectively_same(
    a0,
    a1
):

    a0_normalized = normalize_text(
        a0
    )

    a1_normalized = normalize_text(
        a1
    )

    if a0_normalized == a1_normalized:

        return True

    a0_tokens = set(
        a0_normalized.split()
    )

    a1_tokens = set(
        a1_normalized.split()
    )

    if not a0_tokens or not a1_tokens:

        return False

    intersection = (
        len(
            a0_tokens & a1_tokens
        )
    )

    union = (
        len(
            a0_tokens | a1_tokens
        )
    )

    similarity = (
        intersection
        /
        max(union, 1)
    )

    return similarity >= 0.90


# ============================================================
# CANDIDATE SUPPORT DECISION
# ============================================================
#
# V13:
#
# support >= 0.45
# AND
# support > contradiction
#
# This is intentionally less conservative than V12's
# 0.60 threshold, but still prevents accepting evidence
# when contradiction is equal to or greater than support.
# ============================================================

def candidate_supported(
    verification
):

    status = verification.get(
        "verification_status",
        ""
    )

    support_score = float(
        verification.get(
            "support_score",
            0.0
        )
    )

    contradiction_score = float(
        verification.get(
            "contradiction_score",
            0.0
        )
    )

    if (
        support_score >= SUPPORT_THRESHOLD
        and
        support_score > contradiction_score
    ):

        return True

    if (
        status == "SUPPORTED"
        and
        support_score > contradiction_score
    ):

        return True

    return False


# ============================================================
# CONTROLLED ORACLE
# ============================================================
#
# This is ONLY used after the runtime decision.
#
# It does NOT participate in ACCEPT/ROLLBACK.
# ============================================================

def oracle_matches(
    answer,
    expected_answer
):

    answer_normalized = normalize_text(
        answer
    )

    expected_normalized = normalize_text(
        expected_answer
    )

    if (
        answer_normalized
        ==
        expected_normalized
    ):

        return True

    answer_tokens = set(
        answer_normalized.split()
    )

    expected_tokens = set(
        expected_normalized.split()
    )

    if not expected_tokens:

        return False

    overlap = (
        len(
            answer_tokens
            &
            expected_tokens
        )
        /
        len(expected_tokens)
    )

    return overlap >= 0.75


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(
    case,
    index
):

    case_id = case["id"]

    question = case["question"]

    a0 = case["initial_answer"]

    expected_answer = case[
        "expected_answer"
    ]

    print()
    print("=" * 72)
    print(
        f"V13 SELF-IMPROVEMENT CASE "
        f"{index}/{len(CASES)}"
    )
    print("=" * 72)

    print()
    print("Question:")
    print(question)

    print()
    print("A0:")
    print(a0)


    # ========================================================
    # A0 TRUSTGATE
    # ========================================================

    a0_trust = classify_answer(
        question,
        a0
    )

    print()
    print(
        "A0 TrustGate:",
        a0_trust["label"]
    )

    print(
        "A0 confidence:",
        round(
            a0_trust["confidence"],
            4
        )
    )

    print(
        "A0 entropy:",
        round(
            a0_trust["entropy"],
            4
        )
    )


    # ========================================================
    # A0 VERIFICATION
    # ========================================================

    a0_verified = verify_answer(
        question,
        a0
    )

    a0_verification = (
        a0_verified["verification"]
    )

    a0_status = a0_verification.get(
        "verification_status",
        "UNKNOWN"
    )

    print()
    print(
        "A0 external status:",
        a0_status
    )

    print(
        "A0 support:",
        a0_verification.get(
            "support_score",
            0.0
        )
    )

    print(
        "A0 contradiction:",
        a0_verification.get(
            "contradiction_score",
            0.0
        )
    )

    print(
        "A0 sources:",
        a0_verification.get(
            "source_count",
            0
        )
    )


    # ========================================================
    # CONTROLLED CORRECTION TRIGGER
    # ========================================================
    #
    # All five A0 answers are deliberately wrong.
    #
    # Therefore this controlled experiment always enters
    # the correction path.
    #
    # We still record TrustGate and verifier outputs.
    #
    # ========================================================

    if a0_status == "CONTRADICTED":

        trigger_reason = (
            "EXTERNAL_CONTRADICTION"
        )

    elif a0_trust["label"] in [
        "UNRELIABLE",
        "UNCERTAIN"
    ]:

        trigger_reason = (
            "TRUSTGATE_LOW_TRUST"
        )

    else:

        trigger_reason = (
            "CONTROLLED_TEST_CORRECTION"
        )


    print()
    print(
        "Correction trigger:",
        trigger_reason
    )


    # ========================================================
    # FIRST CORRECTION ATTEMPT
    # ========================================================

    print()
    print(
        "Generating A1 from QUESTION + EVIDENCE..."
    )

    a1 = generate_correction(
        question,
        a0,
        a0_verification,
        retry=False
    )

    print()
    print("A1:")
    print(a1)


    # ========================================================
    # CORRECTION RETRY
    # ========================================================

    retry_used = False

    if answers_effectively_same(
        a0,
        a1
    ):

        retry_used = True

        print()
        print(
            "A1 is effectively the same as A0."
        )

        print(
            "Running one correction retry..."
        )

        a1 = generate_correction(
            question,
            a0,
            a0_verification,
            retry=True
        )

        print()
        print("A1 RETRY:")
        print(a1)


    # ========================================================
    # A1 TRUSTGATE
    # ========================================================

    a1_trust = classify_answer(
        question,
        a1
    )

    print()
    print(
        "A1 TrustGate:",
        a1_trust["label"]
    )

    print(
        "A1 confidence:",
        round(
            a1_trust["confidence"],
            4
        )
    )

    print(
        "A1 entropy:",
        round(
            a1_trust["entropy"],
            4
        )
    )


    # ========================================================
    # A1 VERIFICATION
    # ========================================================

    a1_verified = verify_answer(
        question,
        a1
    )

    a1_verification = (
        a1_verified["verification"]
    )

    a1_status = a1_verification.get(
        "verification_status",
        "UNKNOWN"
    )

    a1_support = float(
        a1_verification.get(
            "support_score",
            0.0
        )
    )

    a1_contradiction = float(
        a1_verification.get(
            "contradiction_score",
            0.0
        )
    )

    print()
    print(
        "A1 external status:",
        a1_status
    )

    print(
        "A1 support:",
        a1_support
    )

    print(
        "A1 contradiction:",
        a1_contradiction
    )

    print(
        "A1 sources:",
        a1_verification.get(
            "source_count",
            0
        )
    )


    # ========================================================
    # FINAL ACCEPT / ROLLBACK
    # ========================================================

    a1_supported = candidate_supported(
        a1_verification
    )

    if a1_supported:

        decision_name = "ACCEPT"

        decision_reason = (
            "A1_SUPPORTED_BY_EVIDENCE"
        )

    else:

        decision_name = "ROLLBACK"

        decision_reason = (
            "A1_NOT_SUFFICIENTLY_SUPPORTED"
        )


    print()
    print(
        "FINAL DECISION:",
        decision_name
    )

    print(
        "Reason:",
        decision_reason
    )


    # ========================================================
    # CONTROLLED EVALUATION
    # ========================================================

    oracle_correct = oracle_matches(
        a1,
        expected_answer
    )

    if (
        decision_name == "ACCEPT"
        and
        oracle_correct
    ):

        improvement_result = (
            "GENUINE_CONTROLLED_CORRECTION"
        )

    elif (
        decision_name == "ACCEPT"
        and
        not oracle_correct
    ):

        improvement_result = (
            "FALSE_ACCEPT"
        )

    elif (
        decision_name == "ROLLBACK"
        and
        oracle_correct
    ):

        improvement_result = (
            "CORRECT_CANDIDATE_REJECTED"
        )

    else:

        improvement_result = (
            "CORRECTION_REJECTED"
        )


    print()
    print(
        "Expected answer:"
    )

    print(
        expected_answer
    )

    print()
    print(
        "Controlled oracle:",
        "CORRECT"
        if oracle_correct
        else
        "INCORRECT"
    )

    print(
        "Improvement evaluation:",
        improvement_result
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "id": case_id,

        "question": question,

        "correction_trigger": {

            "reason": trigger_reason,

            "a0_trustgate": (
                a0_trust["label"]
            ),

            "a0_external_status": (
                a0_status
            ),
        },

        "initial": {

            "answer": a0,

            "claim": a0_verified[
                "claim"
            ],

            "trustgate": a0_trust,

            "verification": (
                a0_verification
            ),
        },

        "candidate": {

            "answer": a1,

            "claim": a1_verified[
                "claim"
            ],

            "trustgate": a1_trust,

            "verification": (
                a1_verification
            ),
        },

        "comparison": {

            "retry_used": retry_used,

            "confidence_delta": (
                a1_trust["confidence"]
                -
                a0_trust["confidence"]
            ),

            "entropy_delta": (
                a1_trust["entropy"]
                -
                a0_trust["entropy"]
            ),

            "trustgate_changed": (
                a0_trust["label"]
                !=
                a1_trust["label"]
            ),

            "support_score": (
                a1_support
            ),

            "contradiction_score": (
                a1_contradiction
            ),

            "candidate_supported": (
                a1_supported
            ),
        },

        "controlled_evaluation": {

            "expected_answer": (
                expected_answer
            ),

            "oracle_correct": (
                oracle_correct
            ),

            "improvement_result": (
                improvement_result
            ),
        },

        "decision": {

            "decision": decision_name,

            "reason": decision_reason,
        },
    }


# ============================================================
# INITIAL STATE
# ============================================================

results = []

start_index = 0

accepted = 0
rollback = 0
errors = 0

genuine_corrections = 0
false_accepts = 0
correct_rejections = 0


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if os.path.exists(
    CHECKPOINT_PATH
):

    try:

        with open(
            CHECKPOINT_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            checkpoint = json.load(file)

        if (
            checkpoint.get(
                "total_cases"
            )
            ==
            len(CASES)
        ):

            results = checkpoint.get(
                "results",
                []
            )

            start_index = checkpoint.get(
                "next_index",
                len(results)
            )

            counts = checkpoint.get(
                "counts",
                {}
            )

            accepted = counts.get(
                "accepted",
                0
            )

            rollback = counts.get(
                "rollback",
                0
            )

            errors = counts.get(
                "errors",
                0
            )

            genuine_corrections = counts.get(
                "genuine_corrections",
                0
            )

            false_accepts = counts.get(
                "false_accepts",
                0
            )

            correct_rejections = counts.get(
                "correct_rejections",
                0
            )

            print()
            print("=" * 72)
            print("CHECKPOINT FOUND")
            print("=" * 72)

            print(
                "Completed:",
                len(results),
                "/",
                len(CASES)
            )

            print(
                "Resuming from:",
                start_index + 1
            )

        else:

            print()
            print(
                "Existing checkpoint does not "
                "match V13."
            )

            print(
                "Starting fresh."
            )

    except Exception as error:

        print()
        print(
            "Checkpoint load error:"
        )

        print(
            repr(error)
        )

        print(
            "Starting fresh."
        )


# ============================================================
# MAIN LOOP
# ============================================================

for i in range(
    start_index,
    len(CASES)
):

    case = CASES[i]

    try:

        result = process_case(
            case,
            i + 1
        )

        results.append(
            result
        )

        decision = result.get(
            "decision",
            {}
        ).get(
            "decision"
        )

        evaluation = result.get(
            "controlled_evaluation",
            {}
        ).get(
            "improvement_result"
        )

        if decision == "ACCEPT":

            accepted += 1

        elif decision == "ROLLBACK":

            rollback += 1

        if (
            evaluation
            ==
            "GENUINE_CONTROLLED_CORRECTION"
        ):

            genuine_corrections += 1

        elif (
            evaluation
            ==
            "FALSE_ACCEPT"
        ):

            false_accepts += 1

        elif (
            evaluation
            ==
            "CORRECT_CANDIDATE_REJECTED"
        ):

            correct_rejections += 1

    except Exception as error:

        errors += 1

        print()
        print(
            "CASE ERROR:",
            repr(error)
        )

        results.append(
            {
                "id": case.get("id"),
                "question": case.get("question"),
                "error": repr(error),
                "decision": {
                    "decision": "ERROR",
                    "reason": repr(error),
                },
            }
        )


    # ========================================================
    # CHECKPOINT AFTER EVERY CASE
    # ========================================================

    checkpoint = {

        "experiment": (
            "TrustGate V13 Controlled "
            "Self-Improvement"
        ),

        "model": MODEL_NAME,

        "support_threshold": (
            SUPPORT_THRESHOLD
        ),

        "total_cases": len(CASES),

        "completed_cases": len(
            results
        ),

        "next_index": i + 1,

        "counts": {

            "accepted": accepted,

            "rollback": rollback,

            "errors": errors,

            "genuine_corrections": (
                genuine_corrections
            ),

            "false_accepts": (
                false_accepts
            ),

            "correct_rejections": (
                correct_rejections
            ),
        },

        "results": results,
    }

    atomic_json_save(
        checkpoint,
        CHECKPOINT_PATH
    )

    print()
    print(
        "CHECKPOINT SAVED:",
        f"{i + 1}/{len(CASES)}"
    )


# ============================================================
# FINAL METRICS
# ============================================================

acceptance_rate = (
    100.0
    * accepted
    /
    max(
        len(results),
        1
    )
)

genuine_rate = (
    100.0
    * genuine_corrections
    /
    max(
        len(results),
        1
    )
)


# ============================================================
# FINAL OUTPUT
# ============================================================

final_output = {

    "experiment": (
        "TrustGate V13 Controlled "
        "Self-Improvement"
    ),

    "model": MODEL_NAME,

    "support_threshold": (
        SUPPORT_THRESHOLD
    ),

    "total_cases": len(CASES),

    "completed_cases": len(
        results
    ),

    "counts": {

        "accepted": accepted,

        "rollback": rollback,

        "errors": errors,

        "genuine_controlled_corrections": (
            genuine_corrections
        ),

        "false_accepts": (
            false_accepts
        ),

        "correct_candidate_rejections": (
            correct_rejections
        ),

        "acceptance_rate_percent": round(
            acceptance_rate,
            2
        ),

        "genuine_correction_rate_percent": round(
            genuine_rate,
            2
        ),
    },

    "architecture": {

        "trustgate_role": (
            "Reliability trigger and trust representation"
        ),

        "correction_strategy": (
            "Question-first generation using retrieved evidence"
        ),

        "retry_strategy": (
            "One retry when A1 is effectively identical to A0"
        ),

        "candidate_acceptance": (
            "support >= 0.45 and support > contradiction"
        ),

        "rollback_condition": (
            "Candidate lacks sufficient evidence support"
        ),

        "checkpointing": (
            "Atomic checkpoint after every case"
        ),
    },

    "controlled_evaluation_note": (
        "Expected answers are evaluation oracles only. "
        "They are never provided to the correction generator "
        "and never participate in the runtime ACCEPT/ROLLBACK "
        "decision."
    ),

    "results": results,
}


# ============================================================
# FINAL ATOMIC SAVE
# ============================================================

atomic_json_save(
    final_output,
    OUTPUT_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 72)
print("TRUSTGATE V13 COMPLETE")
print("=" * 72)

print()
print(
    "Cases processed:",
    len(results),
    "/",
    len(CASES)
)

print(
    "Accepted:",
    accepted
)

print(
    "Rollback:",
    rollback
)

print(
    "Errors:",
    errors
)

print(
    "Genuine controlled corrections:",
    genuine_corrections
)

print(
    "False accepts:",
    false_accepts
)

print(
    "Correct candidates rejected:",
    correct_rejections
)

print(
    "Acceptance rate:",
    round(
        acceptance_rate,
        2
    ),
    "%"
)

print(
    "Genuine correction rate:",
    round(
        genuine_rate,
        2
    ),
    "%"
)

print()
print("Final results:")

print(
    os.path.abspath(
        OUTPUT_PATH
    )
)

print()
print("Checkpoint:")

print(
    os.path.abspath(
        CHECKPOINT_PATH
    )
)

print()
print("=" * 72)