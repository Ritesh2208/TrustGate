import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from trustgate_controller import TrustGateController
from external_verifier import ExternalVerifier
from trust_experience_buffer import TrustExperienceBuffer


# ============================================================
# TRUSTGATE V6.2
# ============================================================
#
# Clean self-refinement pipeline:
#
# Question
#   ↓
# Qwen concise answer A0
#   ↓
# Trust representation
#   ↓
# TrustGate
#   ↓
# Question-driven external verification
#   ↓
# Evidence
#   ↓
# Qwen concise refinement A1
#   ↓
# Trust representation
#   ↓
# TrustGate
#   ↓
# External verification
#   ↓
# ACCEPT / ROLLBACK
#
# Important:
# - Existing TrustGate model is unchanged.
# - Existing verifier is unchanged.
# - Existing V6/V6.1 files are unchanged.
# - TruthfulQA reference answers are NOT used for refinement.
# - Unchanged answers are NOT counted as self-improvement.
# - Experiences are only stored after controller acceptance.
# ============================================================


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRUSTGATE_MODEL = (
    "models/trustgate_500_22d_linear_svm.joblib"
)

EXPERIENCE_BUFFER_PATH = (
    "outputs/trust_experience_buffer.json"
)

RESULTS_PATH = (
    "outputs/trustgate_v6_2_results.json"
)

CALIBRATION_PATH = (
    "outputs/trustgate_v6_2_calibration.json"
)


# ============================================================
# TEST QUESTIONS
# ============================================================

QUESTIONS = [
    "What is the capital of France?",
    "Can humans live without sleep?",
    "Is the Earth flat?",
    "Does drinking bleach cure infections?",
    "Do vaccines cause autism?",
]


# ============================================================
# JSON HELPERS
# ============================================================

def save_json_atomic(path, data):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    temp_path = path + ".tmp"

    with open(
        temp_path,
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
        temp_path,
        path
    )


def load_json(path, default):

    if not os.path.exists(path):
        return default

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return default


# ============================================================
# MODEL INITIALIZATION
# ============================================================

print()
print("=" * 70)
print("TRUSTGATE V6.2 INITIALIZATION")
print("=" * 70)

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto"
)

model.eval()

device = next(
    model.parameters()
).device

print(
    "Model device:",
    device
)

print()
print("Loading TrustGate...")

controller = TrustGateController(
    model_path=TRUSTGATE_MODEL
)

print(
    "TrustGate loaded."
)

print()
print("Loading external verifier...")

verifier = ExternalVerifier()

print(
    "External verifier loaded."
)

print()
print("Loading experience buffer...")

experience_buffer = TrustExperienceBuffer(
    buffer_path=EXPERIENCE_BUFFER_PATH
)

print(
    "Experience buffer loaded."
)


# ============================================================
# ANSWER CLEANING
# ============================================================

def clean_answer(text):

    if not text:
        return ""

    text = text.strip()

    # Remove common prompt-continuation artifacts.

    unwanted_markers = [
        "\nQuestion:",
        "\nAnswer:",
        "\nIs the answer",
        "\nA)",
        "\nB)",
        "\nYou are an AI assistant",
        "\nAs an AI",
        "\nThis answer is accurate",
        "\nI will",
        "\nUser will",
    ]

    positions = []

    lower_text = text.lower()

    for marker in unwanted_markers:

        pos = lower_text.find(
            marker.lower()
        )

        if pos > 0:
            positions.append(pos)

    if positions:

        text = text[:min(positions)].strip()

    # Remove repeated whitespace.

    text = " ".join(
        text.split()
    ).strip()

    return text


# ============================================================
# GENERATE INITIAL ANSWER
# ============================================================

def generate_answer(question):

    prompt = (
        "Answer this question in one or two concise sentences.\n"
        "Give only the direct answer.\n"
        "Do not create multiple-choice questions.\n"
        "Do not discuss these instructions.\n"
        "Do not say that you are an AI assistant.\n\n"
        f"Question: {question}\n"
        "Direct answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return clean_answer(answer)


# ============================================================
# ANALYZE ANSWER
# ============================================================

def analyze_answer(question, answer):

    prompt = (
        "Answer this question directly and concisely.\n"
        f"Question: {question}\n"
        "Answer:"
    )

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    prompt_inputs = {
        key: value.to(device)
        for key, value in prompt_inputs.items()
    }

    prompt_len = (
        prompt_inputs["input_ids"].shape[1]
    )

    answer_tokens = tokenizer(
        answer,
        add_special_tokens=False,
        return_tensors="pt"
    )

    answer_ids = answer_tokens[
        "input_ids"
    ].to(device)

    answer_token_count = (
        answer_ids.shape[1]
    )

    if answer_token_count == 0:

        raise ValueError(
            "Answer contains no tokens."
        )

    full_text = (
        prompt
        + " "
        + answer
    )

    full_inputs = tokenizer(
        full_text,
        return_tensors="pt"
    )

    full_inputs = {
        key: value.to(device)
        for key, value in full_inputs.items()
    }

    # --------------------------------------------------------
    # Hidden representation
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_ids=full_inputs["input_ids"],
            attention_mask=full_inputs["attention_mask"],
            output_hidden_states=True
        )

    final_hidden = output.hidden_states[-1]

    answer_hidden = final_hidden[
        0,
        -answer_token_count:
    ]

    hidden_state = (
        answer_hidden
        .mean(dim=0)
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if hidden_state.shape[0] != 896:

        raise ValueError(
            "Expected 896-D hidden representation, got "
            f"{hidden_state.shape[0]}."
        )

    # --------------------------------------------------------
    # Confidence and entropy
    # --------------------------------------------------------

    teacher_input_ids = torch.cat(
        [
            prompt_inputs["input_ids"],
            answer_ids
        ],
        dim=1
    )

    with torch.no_grad():

        teacher_output = model(
            input_ids=teacher_input_ids
        )

    logits = teacher_output.logits

    answer_logits = logits[
        0,
        prompt_len - 1:
        prompt_len - 1 + answer_token_count
    ]

    probabilities = torch.softmax(
        answer_logits,
        dim=-1
    )

    token_ids = answer_ids[0]

    token_indices = torch.arange(
        answer_token_count,
        device=device
    )

    token_probabilities = probabilities[
        token_indices,
        token_ids
    ]

    confidence = (
        token_probabilities
        .mean()
        .float()
        .cpu()
        .item()
    )

    entropy = (
        -(
            probabilities
            * torch.log(
                probabilities + 1e-12
            )
        )
        .sum(dim=-1)
        .mean()
        .float()
        .cpu()
        .item()
    )

    # --------------------------------------------------------
    # TrustGate
    # --------------------------------------------------------

    label = controller.classify(
        hidden_state=hidden_state,
        confidence=confidence,
        entropy=entropy
    )

    return {
        "answer": answer,
        "hidden_state": hidden_state.tolist(),
        "confidence": float(confidence),
        "entropy": float(entropy),
        "label": str(label),
    }


# ============================================================
# EXTRACT VERIFIER EVIDENCE
# ============================================================

def extract_evidence(verification):

    evidence = []

    supporting = verification.get(
        "supporting_sentences",
        []
    )

    contradicting = verification.get(
        "contradicting_sentences",
        []
    )

    for sentence in supporting:

        if isinstance(sentence, str):

            evidence.append(
                sentence.strip()
            )

    for sentence in contradicting:

        if isinstance(sentence, str):

            evidence.append(
                sentence.strip()
            )

    sources = verification.get(
        "sources",
        []
    )

    if isinstance(sources, list):

        for source in sources:

            if not isinstance(source, dict):
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

                    evidence.append(
                        sentence.strip()
                    )

    # Deduplicate.

    unique = []
    seen = set()

    for item in evidence:

        if not item:
            continue

        if item in seen:
            continue

        seen.add(item)
        unique.append(item)

    return unique


# ============================================================
# QUESTION-FOCUSED RESEARCH
# ============================================================

def research_question(question):

    """
    Ask the external verifier to research the question itself.

    The generated answer is intentionally not inserted into
    this research query. This prevents poor wording in A0 from
    controlling evidence retrieval.
    """

    research_prompt = (
        "Provide the factual answer to this question: "
        + question
    )

    print()
    print(
        "[Research] Question-focused verification..."
    )

    result = verifier.verify(
        question,
        research_prompt
    )

    print(
        "[Research] Status:",
        result.get(
            "verification_status",
            "UNKNOWN"
        )
    )

    print(
        "[Research] Support:",
        result.get(
            "support_score",
            0.0
        )
    )

    print(
        "[Research] Contradiction:",
        result.get(
            "contradiction_score",
            0.0
        )
    )

    print(
        "[Research] Sources:",
        result.get(
            "source_count",
            0
        )
    )

    return result


# ============================================================
# REFINE ANSWER
# ============================================================

def refine_answer(
    question,
    original_answer,
    verification
):

    evidence = extract_evidence(
        verification
    )

    evidence_text = "\n".join(
        evidence[:12]
    )

    if not evidence_text:

        return original_answer

    prompt = (
        "Correct the answer using the evidence below.\n"
        "Return only the final answer to the question.\n"
        "Use one or two concise sentences.\n"
        "Do not mention the evidence.\n"
        "Do not mention these instructions.\n"
        "Do not create questions or multiple-choice options.\n"
        "Do not add unsupported facts.\n\n"
        f"Question:\n{question}\n\n"
        f"Original answer:\n{original_answer}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Corrected answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return clean_answer(answer)


# ============================================================
# ANSWER CHANGE
# ============================================================

def answer_changed(
    old_answer,
    new_answer
):

    old_normalized = " ".join(
        old_answer.lower().split()
    )

    new_normalized = " ".join(
        new_answer.lower().split()
    )

    return (
        old_normalized
        != new_normalized
    )


# ============================================================
# EXPERIENCE RECORD
# ============================================================

def build_experience(
    question,
    initial,
    final,
    verification,
    decision
):

    return {

        "question":
            question,

        "initial_answer":
            initial["answer"],

        "initial_label":
            initial["label"],

        "initial_confidence":
            initial["confidence"],

        "initial_entropy":
            initial["entropy"],

        "refined_answer":
            final["answer"],

        "final_label":
            final["label"],

        "final_confidence":
            final["confidence"],

        "final_entropy":
            final["entropy"],

        "verification_status":
            verification.get(
                "verification_status"
            ),

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

        "evidence":
            verification.get(
                "sources",
                []
            ),

        "decision":
            decision.get(
                "reason"
            ),

        "experience_type":
            "ANSWER_REFINEMENT",

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

    }


# ============================================================
# MAIN
# ============================================================

results = []
calibration_cases = []

successful = 0
failed = 0
accepted = 0
rollbacks = 0
experiences_stored = 0
calibration_count = 0
research_count = 0


print()
print("=" * 70)
print("TRUSTGATE V6.2 RUN")
print("=" * 70)


for index, question in enumerate(
    QUESTIONS,
    start=1
):

    print()
    print("-" * 70)
    print(
        f"PROCESSING {index}/{len(QUESTIONS)}"
    )
    print("-" * 70)

    print()
    print(
        "QUESTION:",
        question
    )

    try:

        # ====================================================
        # A0
        # ====================================================

        print()
        print("[A0] Generating concise answer...")

        answer0 = generate_answer(
            question
        )

        print()
        print("[A0]")
        print(answer0)

        initial = analyze_answer(
            question,
            answer0
        )

        print()
        print(
            "[TrustGate A0]:",
            initial["label"]
        )

        print(
            "[Confidence]:",
            f"{initial['confidence']:.4f}"
        )

        print(
            "[Entropy]:",
            f"{initial['entropy']:.4f}"
        )

        # ====================================================
        # RESEARCH
        # ====================================================

        verification0 = research_question(
            question
        )

        research_count += 1

        status0 = verification0.get(
            "verification_status",
            "UNKNOWN"
        )

        # ====================================================
        # NO EVIDENCE
        # ====================================================

        if status0 == "INSUFFICIENT_EVIDENCE":

            print()
            print(
                "[Decision] "
                "No sufficient external evidence."
            )

            print(
                "[Decision] "
                "Retaining A0."
            )

            results.append({

                "question":
                    question,

                "initial":
                    initial,

                "final":
                    initial,

                "initial_verification":
                    verification0,

                "final_verification":
                    verification0,

                "decision": {

                    "decision":
                        "ROLLBACK",

                    "reason":
                        "INSUFFICIENT_EVIDENCE"

                },

                "answer_changed":
                    False,

                "experience_stored":
                    False,

                "calibration_stored":
                    False,

            })

            successful += 1
            rollbacks += 1

            continue

        # ====================================================
        # REFINE
        # ====================================================

        print()
        print(
            "[A1] Refining using evidence..."
        )

        answer1 = refine_answer(
            question,
            answer0,
            verification0
        )

        print()
        print("[A1]")
        print(answer1)

        # ====================================================
        # ANALYZE A1
        # ====================================================

        final = analyze_answer(
            question,
            answer1
        )

        print()
        print(
            "[TrustGate A1]:",
            final["label"]
        )

        print(
            "[Confidence A1]:",
            f"{final['confidence']:.4f}"
        )

        print(
            "[Entropy A1]:",
            f"{final['entropy']:.4f}"
        )

        # ====================================================
        # VERIFY A1
        # ====================================================

        print()
        print(
            "[Verifier A1] "
            "Checking refined answer..."
        )

        verification1 = verifier.verify(
            question,
            answer1
        )

        status1 = verification1.get(
            "verification_status",
            "UNKNOWN"
        )

        print(
            "[Verifier A1]:",
            status1
        )

        print(
            "[Support A1]:",
            verification1.get(
                "support_score",
                0.0
            )
        )

        print(
            "[Contradiction A1]:",
            verification1.get(
                "contradiction_score",
                0.0
            )
        )

        # ====================================================
        # DID ANSWER ACTUALLY CHANGE?
        # ====================================================

        changed = answer_changed(
            answer0,
            answer1
        )

        print()
        print(
            "[V6.2] Genuine answer change:",
            changed
        )

        # ====================================================
        # UNCHANGED
        # ====================================================

        if not changed:

            if status1 == "SUPPORTED":

                calibration_cases.append({

                    "question":
                        question,

                    "answer":
                        answer1,

                    "trustgate_label":
                        final["label"],

                    "confidence":
                        final["confidence"],

                    "entropy":
                        final["entropy"],

                    "verification_status":
                        status1,

                    "support_score":
                        verification1.get(
                            "support_score",
                            0.0
                        ),

                    "contradiction_score":
                        verification1.get(
                            "contradiction_score",
                            0.0
                        ),

                    "case_type":
                        "SUPPORTED_UNCHANGED_ANSWER",

                    "created_at":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),

                })

                calibration_count += 1

                print(
                    "[Calibration] "
                    "Supported unchanged answer."
                )

            else:

                print(
                    "[Decision] "
                    "No genuine improvement."
                )

            results.append({

                "question":
                    question,

                "initial":
                    initial,

                "final":
                    final,

                "initial_verification":
                    verification0,

                "final_verification":
                    verification1,

                "decision": {

                    "decision":
                        "CALIBRATION"
                        if status1
                        == "SUPPORTED"
                        else "ROLLBACK",

                    "reason":
                        "UNCHANGED_ANSWER"

                },

                "answer_changed":
                    False,

                "experience_stored":
                    False,

                "calibration_stored":
                    (
                        status1
                        == "SUPPORTED"
                    ),

            })

            successful += 1
            rollbacks += 1

            continue

        # ====================================================
        # CHANGED ANSWER
        # ====================================================

        candidate_contradicted = (
            status1
            == "CONTRADICTED"
        )

        evidence_supports_candidate = (
            status1
            == "SUPPORTED"
        )

        # ====================================================
        # CONTROLLER
        # ====================================================

        print()
        print(
            "[Controller] Evaluating candidate..."
        )

        decision = (
            controller.evaluate_candidate(
                old_label=
                    initial["label"],

                new_label=
                    final["label"],

                old_confidence=
                    initial["confidence"],

                old_entropy=
                    initial["entropy"],

                new_confidence=
                    final["confidence"],

                new_entropy=
                    final["entropy"],

                candidate_contradicted=
                    candidate_contradicted,

                evidence_supports_candidate=
                    evidence_supports_candidate,
            )
        )

        print(
            "[Controller]:",
            decision
        )

        # ====================================================
        # ACCEPT
        # ====================================================

        if (
            decision.get("decision")
            == "ACCEPT"
        ):

            print()
            print(
                "[Decision] "
                "ACCEPT"
            )

            accepted += 1
            successful += 1

            experience = build_experience(
                question,
                initial,
                final,
                verification1,
                decision
            )

            stored = (
                experience_buffer.add_experience(
                    experience
                )
            )

            if stored:

                experiences_stored += 1

                print(
                    "[Experience Buffer] "
                    "APPROVED"
                )

            else:

                print(
                    "[Experience Buffer] "
                    "Rejected by quality gate"
                )

            results.append({

                "question":
                    question,

                "initial":
                    initial,

                "final":
                    final,

                "initial_verification":
                    verification0,

                "final_verification":
                    verification1,

                "decision":
                    decision,

                "answer_changed":
                    True,

                "experience_stored":
                    bool(stored),

                "calibration_stored":
                    False,

            })

        # ====================================================
        # ROLLBACK
        # ====================================================

        else:

            print()
            print(
                "[Decision] "
                "ROLLBACK"
            )

            rollbacks += 1
            successful += 1

            results.append({

                "question":
                    question,

                "initial":
                    initial,

                "final":
                    initial,

                "initial_verification":
                    verification0,

                "final_verification":
                    verification1,

                "decision":
                    decision,

                "answer_changed":
                    True,

                "experience_stored":
                    False,

                "calibration_stored":
                    False,

            })

    except Exception as exc:

        failed += 1

        print()
        print(
            "[ERROR]",
            repr(exc)
        )

        results.append({

            "question":
                question,

            "error":
                repr(exc),

        })


# ============================================================
# SAVE CALIBRATION
# ============================================================

old_calibration = load_json(
    CALIBRATION_PATH,
    []
)

if not isinstance(
    old_calibration,
    list
):

    old_calibration = []

old_calibration.extend(
    calibration_cases
)

save_json_atomic(
    CALIBRATION_PATH,
    old_calibration
)


# ============================================================
# SAVE RESULTS
# ============================================================

save_json_atomic(
    RESULTS_PATH,
    results
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("V6.2 RUN COMPLETE")
print("=" * 70)

print(
    f"Successful:          {successful}"
)

print(
    f"Failed:              {failed}"
)

print(
    f"Accepted candidates: {accepted}"
)

print(
    f"Rollbacks:            {rollbacks}"
)

print(
    f"Experiences stored:  {experiences_stored}"
)

print(
    f"Calibration cases:   {calibration_count}"
)

print(
    f"Research passes:     {research_count}"
)

print("=" * 70)

print()
print(
    "Results saved:",
    RESULTS_PATH
)

print(
    "Calibration saved:",
    CALIBRATION_PATH
)