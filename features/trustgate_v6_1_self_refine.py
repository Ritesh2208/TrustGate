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
# TRUSTGATE V6.1
# SELF-REFINE WITH EXTERNAL VERIFICATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRUSTGATE_MODEL = (
    "models/trustgate_500_22d_linear_svm.joblib"
)

EXPERIENCE_BUFFER_PATH = (
    "outputs/trust_experience_buffer.json"
)

RESULTS_PATH = (
    "outputs/trustgate_v6_1_results.json"
)

CALIBRATION_PATH = (
    "outputs/trustgate_v6_1_calibration.json"
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
# JSON UTILITIES
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
# INITIALIZATION
# ============================================================

print()
print("=" * 70)
print("TRUSTGATE V6.1 INITIALIZATION")
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
print("Loading TrustGate controller...")

controller = TrustGateController(
    model_path=TRUSTGATE_MODEL
)

print(
    "TrustGate controller loaded."
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
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    prompt = (
        "Answer the following question accurately and "
        "concisely. Do not invent information.\n\n"
        f"Question: {question}\n"
        "Answer:"
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
            max_new_tokens=128,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    generated_ids = output.sequences[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# ANALYZE ANSWER
# ============================================================

def analyze_answer(question, answer):

    prompt = (
        "Answer the following question accurately and "
        "concisely. Do not invent information.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Answer tokens
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Full sequence
    # --------------------------------------------------------

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

        full_output = model(
            input_ids=full_inputs["input_ids"],
            attention_mask=full_inputs["attention_mask"],
            output_hidden_states=True
        )

    final_hidden = (
        full_output.hidden_states[-1]
    )

    if final_hidden.shape[1] < answer_token_count:

        raise ValueError(
            "Could not locate answer hidden states."
        )

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
            "Expected 896-D hidden state, got "
            f"{hidden_state.shape[0]}."
        )

    # --------------------------------------------------------
    # Confidence + entropy
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

    if (
        answer_logits.shape[0]
        != answer_token_count
    ):

        raise ValueError(
            "Unexpected answer-logit length."
        )

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

    trust_label = controller.classify(
        hidden_state=hidden_state,
        confidence=confidence,
        entropy=entropy
    )

    return {
        "answer": answer,
        "hidden_state": hidden_state.tolist(),
        "confidence": float(confidence),
        "entropy": float(entropy),
        "label": str(trust_label),
    }


# ============================================================
# EXTRACT EVIDENCE
# ============================================================

def extract_evidence_text(verification):

    evidence_parts = []

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

            evidence_parts.append(
                sentence
            )

    for sentence in contradicting:

        if isinstance(sentence, str):

            evidence_parts.append(
                sentence
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

                    evidence_parts.append(
                        sentence
                    )

    # Backward compatibility

    evidence = verification.get(
        "evidence",
        []
    )

    if isinstance(evidence, list):

        for item in evidence:

            if isinstance(item, str):

                evidence_parts.append(
                    item
                )

            elif isinstance(item, dict):

                sentences = item.get(
                    "sentences",
                    []
                )

                if isinstance(
                    sentences,
                    list
                ):

                    for sentence in sentences:

                        if isinstance(
                            sentence,
                            str
                        ):

                            evidence_parts.append(
                                sentence
                            )

    # Remove duplicates

    unique = []
    seen = set()

    for sentence in evidence_parts:

        normalized = sentence.strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        unique.append(normalized)

    return unique


# ============================================================
# SECOND-STAGE RESEARCH
# ============================================================

def research_question(question):

    """
    Run a question-focused external verification pass.

    This is deliberately separate from verification of A0.
    The goal is to avoid depending entirely on the wording
    of the generated answer when constructing the research
    query.
    """

    research_answer = (
        "Research the factual answer to this question: "
        + question
    )

    print()
    print(
        "[Research Pass] "
        "Question-focused external verification..."
    )

    result = verifier.verify(
        question,
        research_answer
    )

    print(
        "[Research Pass] Status:",
        result.get(
            "verification_status",
            "UNKNOWN"
        )
    )

    print(
        "[Research Pass] Support:",
        result.get(
            "support_score",
            0.0
        )
    )

    print(
        "[Research Pass] Sources:",
        result.get(
            "source_count",
            0
        )
    )

    print(
        "[Research Pass] Evidence:",
        result.get(
            "evidence_sentence_count",
            0
        )
    )

    return result


# ============================================================
# REFINE ANSWER
# ============================================================

def refine_answer(
    question,
    answer,
    verification,
    research_verification=None
):

    evidence_sentences = (
        extract_evidence_text(
            verification
        )
    )

    if research_verification is not None:

        evidence_sentences.extend(
            extract_evidence_text(
                research_verification
            )
        )

    # Deduplicate

    unique = []
    seen = set()

    for sentence in evidence_sentences:

        normalized = sentence.strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        unique.append(normalized)

    evidence_text = "\n".join(
        unique[:16]
    )

    if not evidence_text:

        evidence_text = (
            "No usable external evidence was provided."
        )

    prompt = (
        "Improve the answer using the external evidence "
        "below.\n"
        "Keep claims supported by the evidence.\n"
        "Remove claims contradicted by the evidence.\n"
        "Do not invent unsupported facts.\n"
        "Answer the original question directly and concisely.\n"
        "If the original answer is already correct, "
        "return it unchanged.\n\n"
        f"Question:\n{question}\n\n"
        f"Original answer:\n{answer}\n\n"
        f"External evidence:\n{evidence_text}\n\n"
        "Improved answer:"
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
            max_new_tokens=128,
            do_sample=False
        )

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    refined = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return refined


# ============================================================
# ANSWER CHANGE CHECK
# ============================================================

def is_genuine_answer_change(
    initial_answer,
    refined_answer
):

    initial_normalized = (
        " ".join(
            initial_answer.split()
        )
        .strip()
        .lower()
    )

    refined_normalized = (
        " ".join(
            refined_answer.split()
        )
        .strip()
        .lower()
    )

    return (
        initial_normalized
        != refined_normalized
    )


# ============================================================
# BUILD EXPERIENCE
# ============================================================

def build_experience(
    question,
    initial,
    refined,
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
            refined["answer"],

        "final_label":
            refined["label"],

        "final_confidence":
            refined["confidence"],

        "final_entropy":
            refined["entropy"],

        "verification_status":
            verification.get(
                "verification_status"
            ),

        "evidence":
            verification.get(
                "sources",
                []
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
# BUILD CALIBRATION CASE
# ============================================================

def build_calibration_case(
    question,
    initial,
    refined,
    verification
):

    return {

        "question":
            question,

        "answer":
            initial["answer"],

        "trustgate_label":
            initial["label"],

        "confidence":
            initial["confidence"],

        "entropy":
            initial["entropy"],

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

        "answer_changed":
            False,

        "case_type":
            "SUPPORTED_UNCHANGED_ANSWER",

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

    }


# ============================================================
# BUILD RESULT
# ============================================================

def make_result(
    question,
    initial,
    final,
    verification_a0,
    verification_a1,
    decision,
    answer_changed,
    experience_stored=False,
    calibration_stored=False
):

    return {

        "question":
            question,

        "initial":
            initial,

        "final":
            final,

        "initial_verification":
            verification_a0,

        "final_verification":
            verification_a1,

        "decision":
            decision,

        "answer_changed":
            bool(answer_changed),

        "experience_stored":
            bool(experience_stored),

        "calibration_stored":
            bool(calibration_stored),

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
calibration_stored = 0
research_passes = 0


print()
print("=" * 70)
print("TRUSTGATE V6.1 RUN")
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
    print("QUESTION:")
    print(question)

    try:

        # ====================================================
        # INITIAL ANSWER A0
        # ====================================================

        print()
        print("[A0] Generating...")

        initial_answer = generate_answer(
            question
        )

        print()
        print("[A0 ANSWER]")
        print(initial_answer)

        initial = analyze_answer(
            question,
            initial_answer
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
        # EXTERNAL VERIFICATION A0
        # ====================================================

        print()
        print(
            "[Verifier A0] "
            "Gathering external evidence..."
        )

        verification_a0 = verifier.verify(
            question,
            initial_answer
        )

        status_a0 = verification_a0.get(
            "verification_status",
            "UNKNOWN"
        )

        print()
        print(
            "[Verifier A0] Status:",
            status_a0
        )

        print(
            "[Verifier A0] Support:",
            verification_a0.get(
                "support_score",
                0.0
            )
        )

        print(
            "[Verifier A0] Contradiction:",
            verification_a0.get(
                "contradiction_score",
                0.0
            )
        )

        print(
            "[Verifier A0] Sources:",
            verification_a0.get(
                "source_count",
                0
            )
        )

        print(
            "[Verifier A0] Evidence:",
            verification_a0.get(
                "evidence_sentence_count",
                0
            )
        )

        # ====================================================
        # SECOND RESEARCH PASS WHEN NECESSARY
        # ====================================================

        research_verification = None

        if status_a0 == "INSUFFICIENT_EVIDENCE":

            print()
            print(
                "[V6.1] Initial evidence insufficient."
            )

            research_verification = (
                research_question(
                    question
                )
            )

            research_passes += 1

            status_research = (
                research_verification.get(
                    "verification_status",
                    "UNKNOWN"
                )
            )

            # Use the research result for refinement
            # if it contains usable evidence.

            if status_research in [
                "SUPPORTED",
                "CONTRADICTED",
                "EVIDENCE_FOUND"
            ]:

                print(
                    "[V6.1] Research evidence available."
                )

                verification_for_refinement = (
                    research_verification
                )

            else:

                print(
                    "[V6.1] Research also insufficient."
                )

                # We do not fabricate evidence.
                # Preserve A0 and record the case.

                results.append(
                    make_result(
                        question=question,
                        initial=initial,
                        final=initial,
                        verification_a0=verification_a0,
                        verification_a1=research_verification,
                        decision={
                            "decision": "ROLLBACK",
                            "reason": (
                                "INSUFFICIENT_EVIDENCE"
                            )
                        },
                        answer_changed=False,
                        experience_stored=False,
                        calibration_stored=False
                    )
                )

                successful += 1
                rollbacks += 1

                continue

        else:

            verification_for_refinement = (
                verification_a0
            )

        # ====================================================
        # REFINE A0 → A1
        # ====================================================

        print()
        print(
            "[A1] Refining using external evidence..."
        )

        refined_answer = refine_answer(
            question,
            initial_answer,
            verification_for_refinement,
            research_verification
        )

        print()
        print("[A1 ANSWER]")
        print(refined_answer)

        # ====================================================
        # ANALYZE A1
        # ====================================================

        refined = analyze_answer(
            question,
            refined_answer
        )

        print()
        print(
            "[TrustGate A1]:",
            refined["label"]
        )

        print(
            "[Confidence A1]:",
            f"{refined['confidence']:.4f}"
        )

        print(
            "[Entropy A1]:",
            f"{refined['entropy']:.4f}"
        )

        # ====================================================
        # VERIFY A1
        # ====================================================

        print()
        print(
            "[Verifier A1] "
            "Verifying refined answer..."
        )

        verification_a1 = verifier.verify(
            question,
            refined_answer
        )

        status_a1 = verification_a1.get(
            "verification_status",
            "UNKNOWN"
        )

        print()
        print(
            "[Verifier A1] Status:",
            status_a1
        )

        print(
            "[Verifier A1] Support:",
            verification_a1.get(
                "support_score",
                0.0
            )
        )

        print(
            "[Verifier A1] Contradiction:",
            verification_a1.get(
                "contradiction_score",
                0.0
            )
        )

        print(
            "[Verifier A1] Sources:",
            verification_a1.get(
                "source_count",
                0
            )
        )

        print(
            "[Verifier A1] Evidence:",
            verification_a1.get(
                "evidence_sentence_count",
                0
            )
        )

        # ====================================================
        # REAL ANSWER CHANGE
        # ====================================================

        answer_changed = (
            is_genuine_answer_change(
                initial_answer,
                refined_answer
            )
        )

        print()
        print(
            "[V6.1] Answer changed:",
            answer_changed
        )

        # ====================================================
        # UNCHANGED ANSWER
        # ====================================================

        if not answer_changed:

            print()
            print(
                "[V6.1] "
                "No genuine answer change."
            )

            if status_a1 == "SUPPORTED":

                calibration = (
                    build_calibration_case(
                        question,
                        initial,
                        refined,
                        verification_a1
                    )
                )

                calibration_cases.append(
                    calibration
                )

                calibration_stored += 1

                print(
                    "[V6.1] "
                    "Stored as calibration case."
                )

                decision = {
                    "decision":
                        "CALIBRATION",
                    "reason":
                        "UNCHANGED_ANSWER_SUPPORTED"
                }

            else:

                decision = {
                    "decision":
                        "ROLLBACK",
                    "reason":
                        "UNCHANGED_ANSWER"
                }

                print(
                    "[V6.1] "
                    "No calibration case."
                )

            results.append(
                make_result(
                    question=question,
                    initial=initial,
                    final=refined,
                    verification_a0=verification_a0,
                    verification_a1=verification_a1,
                    decision=decision,
                    answer_changed=False,
                    experience_stored=False,
                    calibration_stored=(
                        status_a1
                        == "SUPPORTED"
                    )
                )
            )

            successful += 1
            rollbacks += 1

            continue

        # ====================================================
        # CHANGED ANSWER
        # ====================================================

        candidate_contradicted = (
            status_a1
            == "CONTRADICTED"
        )

        evidence_supports_candidate = (
            status_a1
            == "SUPPORTED"
        )

        print()
        print(
            "[Controller] "
            "Evaluating changed candidate..."
        )

        decision = (
            controller.evaluate_candidate(
                old_label=initial["label"],
                new_label=refined["label"],
                old_confidence=initial["confidence"],
                old_entropy=initial["entropy"],
                new_confidence=refined["confidence"],
                new_entropy=refined["entropy"],
                candidate_contradicted=(
                    candidate_contradicted
                ),
                evidence_supports_candidate=(
                    evidence_supports_candidate
                )
            )
        )

        print()
        print(
            "[Controller Decision]:",
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
                "Candidate ACCEPTED."
            )

            accepted += 1
            successful += 1

            # Only genuine changed answers
            # become experiences.

            experience = build_experience(
                question,
                initial,
                refined,
                verification_a1,
                decision
            )

            accepted_by_buffer = (
                experience_buffer.add_experience(
                    experience
                )
            )

            if accepted_by_buffer:

                experiences_stored += 1

                print(
                    "[Experience Buffer] "
                    "APPROVED."
                )

            else:

                print(
                    "[Experience Buffer] "
                    "REJECTED by quality gate."
                )

            results.append(
                make_result(
                    question=question,
                    initial=initial,
                    final=refined,
                    verification_a0=verification_a0,
                    verification_a1=verification_a1,
                    decision=decision,
                    answer_changed=True,
                    experience_stored=(
                        bool(
                            accepted_by_buffer
                        )
                    ),
                    calibration_stored=False
                )
            )

        # ====================================================
        # ROLLBACK
        # ====================================================

        else:

            print()
            print(
                "[Decision] "
                "Candidate REJECTED → ROLLBACK."
            )

            rollbacks += 1
            successful += 1

            results.append(
                make_result(
                    question=question,
                    initial=initial,
                    final=initial,
                    verification_a0=verification_a0,
                    verification_a1=verification_a1,
                    decision=decision,
                    answer_changed=True,
                    experience_stored=False,
                    calibration_stored=False
                )
            )

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

existing_calibration = load_json(
    CALIBRATION_PATH,
    []
)

if not isinstance(
    existing_calibration,
    list
):

    existing_calibration = []

existing_calibration.extend(
    calibration_cases
)

save_json_atomic(
    CALIBRATION_PATH,
    existing_calibration
)


# ============================================================
# SAVE RESULTS
# ============================================================

save_json_atomic(
    RESULTS_PATH,
    results
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("V6.1 RUN COMPLETE")
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
    f"Rollbacks:           {rollbacks}"
)

print(
    f"Experiences stored:  {experiences_stored}"
)

print(
    f"Calibration cases:   {calibration_stored}"
)

print(
    f"Research passes:     {research_passes}"
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