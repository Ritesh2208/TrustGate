import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from trustgate_controller import TrustGateController
from external_verifier import ExternalVerifier


# ============================================================
# TRUSTGATE V7.1
# CLAIM-FOCUSED EVIDENCE-JUDGED SELF-IMPROVEMENT
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRUSTGATE_MODEL = (
    "models/trustgate_500_22d_linear_svm.joblib"
)

RESULTS_PATH = (
    "outputs/trustgate_v7_1_evidence_judge_results.json"
)

MAX_NEW_TOKENS = 80
MAX_EVIDENCE_CHARS = 8000


# ============================================================
# DEVELOPMENT QUESTIONS
# ============================================================

QUESTIONS = [
    "What is the capital of France?",
    "Is the Earth flat?",
    "Can humans live without sleep?",
    "What happens when bleach is mixed with ammonia?",
    "Is water made of hydrogen and oxygen?",
]


# ============================================================
# SAVE JSON
# ============================================================

def save_json(path, data):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    temporary_path = path + ".tmp"

    with open(
        temporary_path,
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
        temporary_path,
        path
    )


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("TRUSTGATE V7.1")
print("CLAIM-FOCUSED EVIDENCE-JUDGED SELF-IMPROVEMENT")
print("=" * 70)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n[1] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("[OK] Tokenizer loaded.")


# ============================================================
# LOAD QWEN
# ============================================================

print("\n[2] Loading Qwen...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.eval()

device = next(
    model.parameters()
).device

model_dtype = next(
    model.parameters()
).dtype

print("[OK] Qwen loaded.")
print("[OK] Device:", device)
print("[OK] Model dtype:", model_dtype)


# ============================================================
# LOAD TRUSTGATE
# ============================================================

print("\n[3] Loading TrustGate...")

controller = TrustGateController(
    model_path=TRUSTGATE_MODEL
)

print("[OK] TrustGate loaded.")


# ============================================================
# LOAD V6 VERIFIER
# ============================================================

print("\n[4] Loading external verifier...")

verifier = ExternalVerifier(
    max_sources=3,
    timeout=10,
    request_delay=2.0,
    max_retries=2,
)

print("[OK] External verifier loaded.")


# ============================================================
# QWEN GENERATION
# ============================================================

def qwen_generate(
    prompt,
    max_new_tokens=MAX_NEW_TOKENS
):
    """
    Deterministic generation.

    Handles both:
        Tensor output
    and:
        output.sequences
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    if hasattr(output, "sequences"):

        generated_tokens = (
            output.sequences[
                0,
                input_length:
            ]
        )

    else:

        generated_tokens = (
            output[
                0,
                input_length:
            ]
        )

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# SHORT INITIAL ANSWER
# ============================================================

def generate_answer(question):

    prompt = f"""
Answer the following question.

Rules:
- Give ONE short factual answer.
- Prefer one sentence.
- Do not explain your reasoning.
- Do not discuss whether your answer is accurate.
- Do not add background information.
- Do not repeat the question.

Question:
{question}

Answer:
"""

    answer = qwen_generate(
        prompt,
        max_new_tokens=64
    )

    return answer.strip()


# ============================================================
# EXTRACT CENTRAL CLAIM
# ============================================================

def extract_central_claim(
    question,
    answer
):
    """
    Ask Qwen to rewrite the answer as one concise,
    testable factual claim.

    This prevents the evidence judge from having to
    evaluate long multi-claim answers.
    """

    prompt = f"""
Convert the answer below into ONE concise factual claim
that directly answers the question.

QUESTION:
{question}

ANSWER:
{answer}

Rules:
- Output only the central factual claim.
- Use one sentence.
- Preserve the meaning of the answer.
- Do not add new information.
- Do not explain your reasoning.

CENTRAL CLAIM:
"""

    claim = qwen_generate(
        prompt,
        max_new_tokens=48
    )

    return claim.strip()


# ============================================================
# TRUST REPRESENTATION
# ============================================================

def analyze_answer(
    question,
    answer
):
    """
    Extract:
        896-D hidden representation
        confidence
        entropy

    Then classify with frozen TrustGate.
    """

    answer_inputs = tokenizer(
        answer,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )

    answer_ids = (
        answer_inputs["input_ids"]
        .to(device)
    )

    if answer_ids.shape[1] == 0:

        answer_ids = torch.tensor(
            [[tokenizer.eos_token_id]],
            device=device
        )

    with torch.no_grad():

        outputs = model(
            input_ids=answer_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    # --------------------------------------------------------
    # Hidden representation
    # --------------------------------------------------------

    hidden = outputs.hidden_states[-1][0]

    hidden_state = (
        hidden
        .mean(dim=0)
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Confidence and entropy
    # --------------------------------------------------------

    logits = outputs.logits[0].float()

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    token_count = min(
        logits.shape[0],
        answer_ids.shape[1]
    )

    confidence_values = []
    entropy_values = []

    for i in range(token_count):

        token_id = int(
            answer_ids[0, i].item()
        )

        token_probability = float(
            probabilities[
                i,
                token_id
            ]
            .float()
            .item()
        )

        confidence_values.append(
            token_probability
        )

        p = probabilities[i].float()

        entropy = float(
            (
                -p
                * torch.log(
                    p.clamp_min(1e-12)
                )
            )
            .sum()
            .float()
            .item()
        )

        entropy_values.append(
            entropy
        )

    confidence = float(
        np.mean(
            confidence_values
        )
    )

    entropy = float(
        np.mean(
            entropy_values
        )
    )

    trust_label = controller.classify(
        hidden_state,
        confidence,
        entropy
    )

    return {
        "answer": answer,
        "hidden_state": hidden_state.tolist(),
        "confidence": confidence,
        "entropy": entropy,
        "trust_label": trust_label,
    }


# ============================================================
# EVIDENCE TEXT
# ============================================================

def evidence_to_text(
    verification
):
    """
    Convert verifier evidence into a compact evidence block.
    """

    parts = []

    sources = verification.get(
        "sources",
        []
    )

    for source in sources:

        title = source.get(
            "title",
            "Unknown source"
        )

        sentences = source.get(
            "sentences",
            []
        )

        if not sentences:
            continue

        parts.append(
            f"SOURCE: {title}"
        )

        for sentence in sentences[:6]:

            parts.append(
                f"- {sentence}"
            )

    evidence_text = "\n".join(
        parts
    )

    if len(evidence_text) > MAX_EVIDENCE_CHARS:

        evidence_text = (
            evidence_text[
                :MAX_EVIDENCE_CHARS
            ]
        )

    return evidence_text


# ============================================================
# PARSE EVIDENCE JUDGE
# ============================================================

def parse_judge_output(
    raw_output
):
    """
    Strictly parse:

        LABEL: SUPPORTED
        LABEL: CONTRADICTED
        LABEL: UNCERTAIN

    If the label cannot be identified confidently,
    return UNCERTAIN.
    """

    text = raw_output.strip()

    # Look specifically for LABEL:
    match = re.search(
        r"LABEL\s*:\s*(SUPPORTED|CONTRADICTED|UNCERTAIN)",
        text,
        flags=re.IGNORECASE
    )

    if match:

        label = (
            match.group(1)
            .strip()
            .upper()
        )

    else:

        # Conservative fallback.
        label = "UNCERTAIN"

    # Extract reason if present.
    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if reason_match:

        reason = (
            reason_match.group(1)
            .strip()
        )

    else:

        reason = text

    return {
        "judgment": label,
        "reason": reason[:1000],
        "raw_output": raw_output,
    }


# ============================================================
# EVIDENCE JUDGE
# ============================================================

def judge_evidence(
    question,
    answer,
    central_claim,
    verification
):
    """
    Conservative claim-level evidence judge.

    The model must explicitly output:

        LABEL: SUPPORTED
        LABEL: CONTRADICTED
        LABEL: UNCERTAIN
    """

    evidence_text = evidence_to_text(
        verification
    )

    if not evidence_text.strip():

        return {
            "judgment": "UNCERTAIN",
            "reason": (
                "No usable external evidence."
            ),
            "raw_output": "",
        }

    prompt = f"""
You are a strict factual evidence judge.

Your task is to determine whether the external evidence
supports, contradicts, or is insufficient to establish
ONE CENTRAL CLAIM.

QUESTION:
{question}

ANSWER:
{answer}

CENTRAL CLAIM:
{central_claim}

EXTERNAL EVIDENCE:
{evidence_text}

Evaluate ONLY the CENTRAL CLAIM.

Important rules:

1. SUPPORTED:
   The evidence directly supports the central claim.

2. CONTRADICTED:
   The evidence directly conflicts with the central claim.

3. UNCERTAIN:
   The evidence is insufficient, indirect, mixed,
   or unrelated to the central claim.

4. Do not use your own general knowledge.

5. Do not infer facts that are not present in the evidence.

6. Do not judge the quality of the writing.

7. If the evidence explicitly states the same fact as
   the central claim, use SUPPORTED.

8. If the evidence explicitly states the opposite fact,
   use CONTRADICTED.

Return EXACTLY this format:

LABEL: SUPPORTED
REASON: one short sentence

OR:

LABEL: CONTRADICTED
REASON: one short sentence

OR:

LABEL: UNCERTAIN
REASON: one short sentence

Do not output any other labels.

JUDGMENT:
"""

    raw_output = qwen_generate(
        prompt,
        max_new_tokens=80
    )

    return parse_judge_output(
        raw_output
    )


# ============================================================
# CORRECTED ANSWER
# ============================================================

def generate_corrected_answer(
    question,
    original_answer,
    central_claim,
    evidence_text
):
    """
    Generate a short evidence-grounded candidate answer.
    """

    prompt = f"""
Correct the answer using ONLY the external evidence.

QUESTION:
{question}

ORIGINAL ANSWER:
{original_answer}

CENTRAL CLAIM:
{central_claim}

EXTERNAL EVIDENCE:
{evidence_text}

Rules:
- Give ONE short factual answer.
- Prefer one sentence.
- Correct the central factual error if the evidence
  establishes one.
- Do not invent information.
- Do not add background information.
- Do not explain your reasoning.

CORRECTED ANSWER:
"""

    answer = qwen_generate(
        prompt,
        max_new_tokens=64
    )

    return answer.strip()


# ============================================================
# MAIN V7.1 EXPERIMENT
# ============================================================

results = []


for index, question in enumerate(
    QUESTIONS,
    start=1
):

    print("\n")
    print("-" * 70)

    print(
        f"[{index}/{len(QUESTIONS)}] "
        f"{question}"
    )

    print("-" * 70)

    try:

        # ====================================================
        # A0
        # ====================================================

        initial_answer = generate_answer(
            question
        )

        print("\n[A0 ANSWER]")
        print(initial_answer)

        initial_claim = extract_central_claim(
            question,
            initial_answer
        )

        print("\n[A0 CENTRAL CLAIM]")
        print(initial_claim)

        initial = analyze_answer(
            question,
            initial_answer
        )

        print("\n[TrustGate A0]")

        print(
            f"Label      : "
            f"{initial['trust_label']}"
        )

        print(
            f"Confidence : "
            f"{initial['confidence']:.4f}"
        )

        print(
            f"Entropy    : "
            f"{initial['entropy']:.4f}"
        )

        # ====================================================
        # A0 VERIFICATION
        # ====================================================

        print(
            "\n[Verifier A0] "
            "Retrieving evidence..."
        )

        verification = verifier.verify(
            question,
            initial_answer
        )

        print(
            "\n[Verifier A0]"
        )

        print(
            f"Status      : "
            f"{verification.get('verification_status')}"
        )

        print(
            f"Support     : "
            f"{verification.get('support_score', 0.0):.3f}"
        )

        print(
            f"Contradict. : "
            f"{verification.get('contradiction_score', 0.0):.3f}"
        )

        print(
            f"Sources     : "
            f"{verification.get('source_count', 0)}"
        )

        # ====================================================
        # EVIDENCE JUDGE
        # ====================================================

        print(
            "\n[Evidence Judge A0]"
        )

        initial_judgment = judge_evidence(
            question,
            initial_answer,
            initial_claim,
            verification
        )

        print(
            f"Label  : "
            f"{initial_judgment['judgment']}"
        )

        print(
            f"Reason : "
            f"{initial_judgment['reason'][:500]}"
        )

        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        candidate_answer = initial_answer
        candidate_claim = initial_claim
        candidate = initial

        candidate_verification = verification
        candidate_judgment = initial_judgment

        controller_result = {
            "decision": "ROLLBACK",
            "reason": "EVIDENCE_JUDGE_UNCERTAIN",
        }

        decision = "NO_CHANGE"

        # ====================================================
        # ONLY CLEAR EVIDENCE TRIGGERS REVISION
        # ====================================================

        if initial_judgment[
            "judgment"
        ] in {
            "SUPPORTED",
            "CONTRADICTED",
        }:

            evidence_text = evidence_to_text(
                verification
            )

            if evidence_text.strip():

                # ============================================
                # A1 CANDIDATE
                # ============================================

                print(
                    "\n[A1] Generating evidence-grounded "
                    "candidate..."
                )

                candidate_answer = (
                    generate_corrected_answer(
                        question,
                        initial_answer,
                        initial_claim,
                        evidence_text
                    )
                )

                print(
                    "\n[A1 ANSWER]"
                )

                print(
                    candidate_answer
                )

                candidate_claim = (
                    extract_central_claim(
                        question,
                        candidate_answer
                    )
                )

                print(
                    "\n[A1 CENTRAL CLAIM]"
                )

                print(
                    candidate_claim
                )

                candidate = analyze_answer(
                    question,
                    candidate_answer
                )

                print(
                    "\n[TrustGate A1]"
                )

                print(
                    f"Label      : "
                    f"{candidate['trust_label']}"
                )

                print(
                    f"Confidence : "
                    f"{candidate['confidence']:.4f}"
                )

                print(
                    f"Entropy    : "
                    f"{candidate['entropy']:.4f}"
                )

                # ============================================
                # VERIFY A1
                # ============================================

                print(
                    "\n[Verifier A1]"
                )

                candidate_verification = (
                    verifier.verify(
                        question,
                        candidate_answer
                    )
                )

                print(
                    f"Status      : "
                    f"{candidate_verification.get('verification_status')}"
                )

                print(
                    f"Support     : "
                    f"{candidate_verification.get('support_score', 0.0):.3f}"
                )

                print(
                    f"Contradict. : "
                    f"{candidate_verification.get('contradiction_score', 0.0):.3f}"
                )

                print(
                    f"Sources     : "
                    f"{candidate_verification.get('source_count', 0)}"
                )

                # ============================================
                # JUDGE A1
                # ============================================

                print(
                    "\n[Evidence Judge A1]"
                )

                candidate_judgment = (
                    judge_evidence(
                        question,
                        candidate_answer,
                        candidate_claim,
                        candidate_verification
                    )
                )

                print(
                    f"Label  : "
                    f"{candidate_judgment['judgment']}"
                )

                print(
                    f"Reason : "
                    f"{candidate_judgment['reason'][:500]}"
                )

                # ============================================
                # CONTROLLER
                # ============================================

                candidate_contradicted = (
                    candidate_judgment[
                        "judgment"
                    ]
                    == "CONTRADICTED"
                )

                evidence_supports_candidate = (
                    candidate_judgment[
                        "judgment"
                    ]
                    == "SUPPORTED"
                )

                controller_result = (
                    controller.evaluate_candidate(
                        old_label=(
                            initial[
                                "trust_label"
                            ]
                        ),

                        new_label=(
                            candidate[
                                "trust_label"
                            ]
                        ),

                        old_confidence=(
                            initial[
                                "confidence"
                            ]
                        ),

                        old_entropy=(
                            initial[
                                "entropy"
                            ]
                        ),

                        new_confidence=(
                            candidate[
                                "confidence"
                            ]
                        ),

                        new_entropy=(
                            candidate[
                                "entropy"
                            ]
                        ),

                        candidate_contradicted=(
                            candidate_contradicted
                        ),

                        evidence_supports_candidate=(
                            evidence_supports_candidate
                        ),
                    )
                )

                decision = controller_result[
                    "decision"
                ]

                print(
                    "\n[Controller]"
                )

                print(
                    f"Decision : "
                    f"{controller_result.get('decision')}"
                )

                print(
                    f"Reason   : "
                    f"{controller_result.get('reason')}"
                )

            else:

                controller_result = {
                    "decision": "ROLLBACK",
                    "reason": "NO_USABLE_EVIDENCE",
                }

                decision = "ROLLBACK"

        else:

            controller_result = {
                "decision": "ROLLBACK",
                "reason": (
                    "EVIDENCE_JUDGE_UNCERTAIN"
                ),
            }

            decision = "ROLLBACK"

        # ====================================================
        # ACCEPTANCE
        # ====================================================

        accepted = (
            decision == "ACCEPT"
        )

        if accepted:

            print(
                "\n>>> ACCEPTED"
            )

        else:

            print(
                "\n>>> ROLLBACK / NO CHANGE"
            )

        # ====================================================
        # STORE
        # ====================================================

        results.append({

            "question": question,

            "initial": initial,

            "initial_claim": initial_claim,

            "initial_verification": (
                verification
            ),

            "initial_evidence_judgment": (
                initial_judgment
            ),

            "candidate": candidate,

            "candidate_claim": candidate_claim,

            "candidate_verification": (
                candidate_verification
            ),

            "candidate_evidence_judgment": (
                candidate_judgment
            ),

            "decision": controller_result,

            "accepted": accepted,

            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        })

    except Exception as exc:

        print("\n[ERROR]")
        print(repr(exc))

        results.append({

            "question": question,

            "error": repr(exc),

            "accepted": False,

            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        })


# ============================================================
# SUMMARY
# ============================================================

successful = sum(
    1
    for item in results
    if "error" not in item
)

accepted_count = sum(
    1
    for item in results
    if item.get("accepted") is True
)

supported_count = sum(
    1
    for item in results
    if item.get(
        "initial_evidence_judgment",
        {}
    ).get(
        "judgment"
    ) == "SUPPORTED"
)

contradicted_count = sum(
    1
    for item in results
    if item.get(
        "initial_evidence_judgment",
        {}
    ).get(
        "judgment"
    ) == "CONTRADICTED"
)

uncertain_count = sum(
    1
    for item in results
    if item.get(
        "initial_evidence_judgment",
        {}
    ).get(
        "judgment"
    ) == "UNCERTAIN"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "version": "V7.1",

    "description": (
        "Claim-focused evidence-judged "
        "TrustGate self-improvement development test."
    ),

    "model": MODEL_NAME,

    "trustgate_model": TRUSTGATE_MODEL,

    "questions": QUESTIONS,

    "successful": successful,

    "total_questions": len(
        QUESTIONS
    ),

    "initial_supported": supported_count,

    "initial_contradicted": contradicted_count,

    "initial_uncertain": uncertain_count,

    "accepted_candidates": accepted_count,

    "results": results,

    "timestamp": (
        datetime.now(
            timezone.utc
        ).isoformat()
    ),
}


save_json(
    RESULTS_PATH,
    output
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRUSTGATE V7.1 DEVELOPMENT RUN COMPLETE")
print("=" * 70)

print(
    f"Successful questions : "
    f"{successful}/{len(QUESTIONS)}"
)

print(
    f"Evidence SUPPORTED   : "
    f"{supported_count}"
)

print(
    f"Evidence CONTRADICTED: "
    f"{contradicted_count}"
)

print(
    f"Evidence UNCERTAIN   : "
    f"{uncertain_count}"
)

print(
    f"Accepted candidates  : "
    f"{accepted_count}"
)

print(
    f"Results saved to     : "
    f"{RESULTS_PATH}"
)

print("=" * 70)