import os
import re
import json
import math
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from features.trustgate_controller import TrustGateController
from features.external_verifier import ExternalVerifier


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_PATH = (
    "outputs/trustgate_v7_1_natural_270_results.json"
)

OUTPUT_PATH = (
    "outputs/trustgate_v8_correction_30_results.json"
)

MAX_NEW_TOKENS = 60


# ============================================================
# LOAD V7.1 RESULTS
# ============================================================

print("=" * 70)
print("TRUSTGATE V8 — CORRECTION-ONLY EXPERIMENT")
print("=" * 70)

print("\nLoading V7.1 results...")

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:
    v7_data = json.load(f)


v7_results = v7_data["results"]

correction_cases = [
    x for x in v7_results
    if x.get("decision", {}).get("decision") == "ROLLBACK"
    or (
        x.get("initial", {}).get("verification", {}).get(
            "verification_status"
        )
        == "CONTRADICTED"
    )
]


# Remove accidental duplicates by held-out ID.

unique_cases = {}

for item in correction_cases:
    unique_cases[item["heldout_id"]] = item

correction_cases = list(
    unique_cases.values()
)


print(
    "\nV7.1 total cases:",
    len(v7_results)
)

print(
    "Correction cases selected:",
    len(correction_cases)
)


if len(correction_cases) == 0:
    raise RuntimeError(
        "No correction cases were found in the V7.1 results."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen model...")

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

print("\nLoading TrustGate...")

controller = TrustGateController()

print("TrustGate loaded.")


# ============================================================
# LOAD EXTERNAL VERIFIER
# ============================================================

print("\nLoading external verifier...")

verifier = ExternalVerifier()

print("External verifier loaded.")


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    question,
    instruction
):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual question answering "
                "assistant. Answer directly and concisely. "
                "Give only the answer needed to answer the "
                "question. Do not provide background, "
                "reasoning, speculation, or multiple claims."
            )
        },
        {
            "role": "user",
            "content": instruction
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

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claim(
    question,
    answer
):

    prompt = f"""
Question:

{question}

Answer:

{answer}

Extract ONLY the single central factual claim
needed to answer the question.

Rules:

- Preserve negation.
- Preserve yes/no meaning.
- Preserve important numbers and names.
- Do not add facts.
- Do not add explanations.
- Do not combine multiple claims.
- Return exactly one short sentence.

Return only the claim.
"""

    claim = generate_answer(
        question,
        prompt
    ).strip()

    claim = re.sub(
        r"^(Claim|Central claim)\s*:\s*",
        "",
        claim,
        flags=re.IGNORECASE
    )

    claim = claim.split(
        "Explanation:"
    )[0].strip()

    lines = [
        x.strip()
        for x in claim.splitlines()
        if x.strip()
    ]

    if lines:
        claim = lines[0]

    return claim


# ============================================================
# EVIDENCE JUDGE
# ============================================================

def judge_evidence(
    question,
    answer,
    claim,
    verification
):

    status = verification.get(
        "verification_status",
        "INSUFFICIENT_EVIDENCE"
    )

    support = float(
        verification.get(
            "support_score",
            0.0
        )
    )

    contradiction = float(
        verification.get(
            "contradiction_score",
            0.0
        )
    )

    supporting = verification.get(
        "supporting_sentences",
        []
    )

    contradicting = verification.get(
        "contradicting_sentences",
        []
    )

    prompt = f"""
You are a strict factual evidence judge.

Question:
{question}

Answer:
{answer}

CENTRAL CLAIM:
{claim}

External verification status:
{status}

Support score:
{support:.3f}

Contradiction score:
{contradiction:.3f}

Supporting evidence:
{json.dumps(
    supporting[:6],
    ensure_ascii=False
)}

Contradicting evidence:
{json.dumps(
    contradicting[:6],
    ensure_ascii=False
)}

Determine whether the supplied evidence supports
or contradicts the CENTRAL CLAIM.

Rules:

1. Judge the claim, not keyword overlap.
2. Preserve negation exactly.
3. A claim and its negation are opposite.
4. Do not use outside knowledge.
5. If evidence is insufficient or ambiguous, use UNCERTAIN.
6. Do not infer support merely because words overlap.
7. The evidence must actually address the claim.

Return exactly one of:

LABEL: SUPPORTED

LABEL: CONTRADICTED

LABEL: UNCERTAIN

Then provide:

REASON: one short sentence
"""

    result = generate_answer(
        question,
        prompt
    )

    match = re.search(
        r"LABEL\s*:\s*"
        r"(SUPPORTED|CONTRADICTED|UNCERTAIN)",
        result,
        flags=re.IGNORECASE
    )

    if match:
        label = match.group(1).upper()
    else:
        label = "UNCERTAIN"

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        result,
        flags=re.IGNORECASE
    )

    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        reason = result.strip()

    return {
        "judgment": label,
        "reason": reason,
        "raw_output": result
    }


# ============================================================
# TRUST PRIORITY
# ============================================================

TRUST_PRIORITY = {
    "UNRELIABLE": 0,
    "UNCERTAIN": 1,
    "RELIABLE": 2
}


def trust_priority(label):

    return TRUST_PRIORITY.get(
        str(label),
        -1
    )


# ============================================================
# TRUST FEATURES
# ============================================================

def get_trust_features(answer):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual question answering "
                "assistant."
            )
        },
        {
            "role": "user",
            "content": answer
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

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model(
            **inputs,
            output_hidden_states=True
        )

        logits = output.logits.float()

        hidden = output.hidden_states[-1].float()

    input_ids = inputs["input_ids"]

    attention_mask = inputs.get(
        "attention_mask",
        torch.ones_like(input_ids)
    )

    sequence_length = input_ids.shape[1]

    if sequence_length < 2:

        hidden_vector = (
            hidden[0, -1, :]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        return {
            "hidden_state": hidden_vector,
            "confidence": 0.0,
            "entropy": 0.0
        }

    # Next-token probabilities.

    next_logits = logits[:, :-1, :]

    next_targets = input_ids[:, 1:]

    next_mask = attention_mask[:, 1:]

    probabilities = torch.softmax(
        next_logits,
        dim=-1
    )

    target_probabilities = probabilities.gather(
        2,
        next_targets.unsqueeze(-1)
    ).squeeze(-1)

    mask = next_mask.float()

    confidence = (
        target_probabilities * mask
    ).sum() / mask.sum().clamp_min(1.0)

    log_probabilities = torch.log(
        probabilities.clamp_min(1e-12)
    )

    entropy = -(
        probabilities * log_probabilities
    ).sum(dim=-1)

    entropy = (
        entropy * mask
    ).sum() / mask.sum().clamp_min(1.0)

    hidden_mask = attention_mask.float()

    hidden_vector = (
        hidden[0] * hidden_mask.unsqueeze(-1)
    ).sum(dim=0) / hidden_mask.sum().clamp_min(1.0)

    hidden_vector = (
        hidden_vector
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
        .tolist()
    )

    return {
        "hidden_state": hidden_vector,
        "confidence": float(
            confidence.detach().cpu().item()
        ),
        "entropy": float(
            entropy.detach().cpu().item()
        )
    }


# ============================================================
# BUILD CORRECTION
# ============================================================

def generate_correction(
    question,
    old_answer,
    old_claim,
    verification
):

    supporting = verification.get(
        "supporting_sentences",
        []
    )

    contradicting = verification.get(
        "contradicting_sentences",
        []
    )

    prompt = f"""
Question:

{question}

Original answer:

{old_answer}

Original central claim:

{old_claim}

The original claim was judged contradicted
by external evidence.

Contradicting evidence:

{json.dumps(
    contradicting[:6],
    ensure_ascii=False
)}

Potential supporting evidence:

{json.dumps(
    supporting[:6],
    ensure_ascii=False
)}

Write a corrected answer to the ORIGINAL question.

STRICT RULES:

- Correct only the factual error.
- Answer the original question directly.
- Use the evidence above.
- Preserve important negation.
- Do not mention evidence or verification.
- Do not explain your reasoning.
- Do not add background.
- Do not introduce unrelated facts.
- Use one short sentence whenever possible.
- Give one atomic factual answer.

Return ONLY the corrected answer.
"""

    return generate_answer(
        question,
        prompt
    )


# ============================================================
# PROCESS ONE CORRECTION CASE
# ============================================================

def process_case(
    item,
    index,
    total
):

    heldout_id = item["heldout_id"]

    question = item["question"]

    initial = item["initial"]

    a0 = initial["answer"]

    a0_claim = initial.get(
        "claim"
    )

    if not a0_claim:

        a0_claim = extract_claim(
            question,
            a0
        )

    a0_verification = initial.get(
        "verification"
    )

    if not a0_verification:

        a0_verification = verifier.verify(
            question,
            a0
        )

    a0_judgment = initial.get(
        "evidence_judgment"
    )

    if not a0_judgment:

        a0_judgment = judge_evidence(
            question,
            a0,
            a0_claim,
            a0_verification
        )

    print("\n" + "=" * 70)
    print(
        f"V8 CASE {index}/{total}"
    )
    print("=" * 70)

    print(
        "Held-out ID:",
        heldout_id
    )

    print(
        "\nQuestion:",
        question
    )

    print(
        "\nA0:",
        a0
    )

    print(
        "\nA0 Claim:",
        a0_claim
    )

    print(
        "A0 Evidence:",
        a0_judgment["judgment"]
    )

    # --------------------------------------------------------
    # A0 TRUST
    # --------------------------------------------------------

    a0_features = get_trust_features(
        a0
    )

    a0_trust = controller.classify(
        a0_features["hidden_state"],
        a0_features["confidence"],
        a0_features["entropy"]
    )

    print(
        "A0 TrustGate:",
        a0_trust
    )

    # --------------------------------------------------------
    # ONLY PROCESS CONTRADICTED CASES
    # --------------------------------------------------------

    if a0_judgment["judgment"] != "CONTRADICTED":

        return {
            "heldout_id": heldout_id,
            "question": question,
            "status": "SKIPPED_NOT_CONTRADICTED",
            "decision": {
                "decision": "KEEP_INITIAL",
                "reason": "NOT_CONTRADICTED"
            }
        }

    # --------------------------------------------------------
    # GENERATE CORRECTION
    # --------------------------------------------------------

    a1 = generate_correction(
        question,
        a0,
        a0_claim,
        a0_verification
    )

    print(
        "\nA1:",
        a1
    )

    # --------------------------------------------------------
    # A1 CLAIM
    # --------------------------------------------------------

    a1_claim = extract_claim(
        question,
        a1
    )

    print(
        "A1 Claim:",
        a1_claim
    )

    # --------------------------------------------------------
    # A1 VERIFICATION
    # --------------------------------------------------------

    a1_verification = verifier.verify(
        question,
        a1
    )

    a1_judgment = judge_evidence(
        question,
        a1,
        a1_claim,
        a1_verification
    )

    print(
        "A1 Evidence:",
        a1_judgment["judgment"]
    )

    # --------------------------------------------------------
    # A1 TRUST
    # --------------------------------------------------------

    a1_features = get_trust_features(
        a1
    )

    a1_trust = controller.classify(
        a1_features["hidden_state"],
        a1_features["confidence"],
        a1_features["entropy"]
    )

    print(
        "A1 TrustGate:",
        a1_trust
    )

    # --------------------------------------------------------
    # TRUST COMPARISON
    # --------------------------------------------------------

    p0 = trust_priority(
        a0_trust
    )

    p1 = trust_priority(
        a1_trust
    )

    trust_improved = (
        p1 > p0
    )

    trust_same = (
        p1 == p0
    )

    confidence_improved = (
        a1_features["confidence"]
        >
        a0_features["confidence"]
    )

    entropy_improved = (
        a1_features["entropy"]
        <
        a0_features["entropy"]
    )

    # --------------------------------------------------------
    # V8 ACCEPTANCE GATE
    # --------------------------------------------------------

    evidence_supported = (
        a1_judgment["judgment"]
        == "SUPPORTED"
    )

    if not evidence_supported:

        decision = {
            "decision": "ROLLBACK",
            "reason": "A1_EVIDENCE_NOT_SUPPORTED"
        }

    elif trust_improved:

        decision = {
            "decision": "ACCEPT",
            "reason": "EVIDENCE_SUPPORTED_AND_TRUST_IMPROVED"
        }

    elif trust_same and (
        confidence_improved
        and entropy_improved
    ):

        decision = {
            "decision": "ACCEPT",
            "reason": (
                "EVIDENCE_SUPPORTED_AND_"
                "CONFIDENCE_ENTROPY_IMPROVED"
            )
        }

    else:

        decision = {
            "decision": "ROLLBACK",
            "reason": (
                "EVIDENCE_SUPPORTED_BUT_"
                "TRUST_NOT_IMPROVED"
            )
        }

    print(
        "\nTrust comparison:",
        a0_trust,
        "->",
        a1_trust
    )

    print(
        "Confidence:",
        round(
            a0_features["confidence"],
            4
        ),
        "->",
        round(
            a1_features["confidence"],
            4
        )
    )

    print(
        "Entropy:",
        round(
            a0_features["entropy"],
            4
        ),
        "->",
        round(
            a1_features["entropy"],
            4
        )
    )

    print(
        "\nV8 Decision:",
        decision
    )

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "heldout_id": heldout_id,
        "question": question,

        "initial": {
            "answer": a0,
            "claim": a0_claim,
            "trust_label": a0_trust,
            "confidence": a0_features["confidence"],
            "entropy": a0_features["entropy"],
            "verification": a0_verification,
            "evidence_judgment": a0_judgment
        },

        "candidate": {
            "answer": a1,
            "claim": a1_claim,
            "trust_label": a1_trust,
            "confidence": a1_features["confidence"],
            "entropy": a1_features["entropy"],
            "verification": a1_verification,
            "evidence_judgment": a1_judgment
        },

        "comparison": {
            "initial_trust_priority": p0,
            "candidate_trust_priority": p1,
            "trust_improved": trust_improved,
            "trust_same": trust_same,
            "confidence_improved": confidence_improved,
            "entropy_improved": entropy_improved,
            "evidence_supported": evidence_supported
        },

        "decision": decision
    }


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    accepted = 0
    rollback = 0
    skipped = 0

    for i, item in enumerate(
        correction_cases,
        start=1
    ):

        try:

            result = process_case(
                item,
                i,
                len(correction_cases)
            )

            results.append(result)

            decision = result.get(
                "decision",
                {}
            ).get(
                "decision"
            )

            if decision == "ACCEPT":
                accepted += 1

            elif decision == "ROLLBACK":
                rollback += 1

            else:
                skipped += 1

        except Exception as e:

            print(
                "\nERROR:",
                repr(e)
            )

            results.append({
                "heldout_id": item.get(
                    "heldout_id"
                ),
                "question": item.get(
                    "question"
                ),
                "error": repr(e)
            })

    # ========================================================
    # SUMMARY
    # ========================================================

    output = {
        "experiment": (
            "TrustGate V8 Correction-Only "
            "30-Case Evaluation"
        ),

        "model": MODEL_NAME,

        "source": INPUT_PATH,

        "description": (
            "V8 re-evaluates only the V7.1 cases "
            "that entered the correction path. "
            "Acceptance requires evidence support "
            "plus TrustGate improvement, or equal "
            "TrustGate class with improved confidence "
            "and reduced entropy."
        ),

        "num_cases": len(results),

        "counts": {
            "accepted": accepted,
            "rollback": rollback,
            "skipped": skipped
        },

        "results": results
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n")
    print("=" * 70)
    print("TRUSTGATE V8 COMPLETE")
    print("=" * 70)

    print(
        "\nCases processed:",
        len(results)
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
        "Skipped:",
        skipped
    )

    print(
        "\nResults saved to:"
    )

    print(
        os.path.abspath(
            OUTPUT_PATH
        )
    )


if __name__ == "__main__":
    main()