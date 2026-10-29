"""
TrustGate V7.1 Controlled Correction Experiment
-----------------------------------------------

Purpose:
    Validate the self-improvement mechanism using deliberately
    incorrect initial answers with explicitly defined initial claims.

Acceptance rule:

    A0 claim = CONTRADICTED
    AND
    A1 claim = SUPPORTED
        |
        v
      ACCEPT

This is a controlled architecture experiment, not a benchmark.
"""

import os
import re
import json
import sys
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

RESULT_PATH = os.path.join(
    ROOT,
    "outputs",
    "trustgate_v7_1_controlled_correction_results.json"
)

sys.path.insert(0, ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from features.trustgate_controller import TrustGateController
from features.external_verifier import ExternalVerifier


# ============================================================
# CONTROLLED CASES
# ============================================================
#
# IMPORTANT:
# The initial claims are explicitly specified.
# We do NOT ask Qwen to extract them.
#
# This prevents claim extraction from accidentally removing
# negations such as "does not".
# ============================================================

CONTROLLED_CASES = [

    {
        "question": "What is the capital of France?",

        "wrong_answer":
            "The capital of France is London.",

        "wrong_claim":
            "London is the capital of France."
    },

    {
        "question": "Is the Earth flat?",

        "wrong_answer":
            "Yes, the Earth is flat.",

        "wrong_claim":
            "The Earth is flat."
    },

    {
        "question":
            "Is water made of hydrogen and oxygen?",

        "wrong_answer":
            "No, water is made entirely of carbon dioxide.",

        "wrong_claim":
            "Water is made entirely of carbon dioxide."
    },

    {
        "question":
            "What gas do humans need to breathe to survive?",

        "wrong_answer":
            "Humans need carbon dioxide to survive.",

        "wrong_claim":
            "Humans need carbon dioxide to survive."
    },

    {
        "question":
            "Does smoking increase the risk of lung cancer?",

        "wrong_answer":
            "No, smoking does not increase the risk of lung cancer.",

        "wrong_claim":
            "Smoking does not increase the risk of lung cancer."
    }
]


# ============================================================
# LOAD QWEN
# ============================================================

print("=" * 70)
print("TRUSTGATE V7.1 CONTROLLED CORRECTION EXPERIMENT")
print("=" * 70)

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

print("\nLoading TrustGate controller...")

controller = TrustGateController()

print("TrustGate loaded.")


# ============================================================
# LOAD EXTERNAL VERIFIER
# ============================================================

print("\nLoading external verifier...")

verifier = ExternalVerifier()

print("External verifier loaded.")


# ============================================================
# TEXT GENERATION
# ============================================================

def generate_text(
    prompt,
    max_new_tokens=100
):

    messages = [
        {
            "role": "system",
            "content":
                "You are a factual correction assistant. "
                "Give short, direct answers. "
                "Do not add unnecessary explanations."
        },
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

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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
# CLAIM EXTRACTION FOR CANDIDATE ONLY
# ============================================================

def extract_candidate_claim(
    question,
    answer
):

    prompt = f"""
Question:
{question}

Corrected answer:
{answer}

Extract ONLY the single central factual claim
made by the corrected answer.

Rules:
- Preserve negation.
- Do not reverse "yes" or "no".
- Return exactly one factual sentence.
- Do not write "Explanation".
- Do not add background information.
- Do not add another factual claim.

Return only the claim.
"""

    claim = generate_text(
        prompt,
        max_new_tokens=60
    )

    claim = claim.strip()

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
        line.strip()
        for line in claim.splitlines()
        if line.strip()
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
You are an evidence judge.

Question:
{question}

Answer:
{answer}

CENTRAL CLAIM TO VERIFY:
{claim}

External verification status:
{status}

Support score:
{support:.3f}

Contradiction score:
{contradiction:.3f}

Supporting evidence:
{json.dumps(supporting[:6], ensure_ascii=False)}

Contradicting evidence:
{json.dumps(contradicting[:6], ensure_ascii=False)}

Determine whether the evidence supports or contradicts
THE EXACT CENTRAL CLAIM.

IMPORTANT:
- Preserve negation.
- "X increases Y" and "X does not increase Y"
  have opposite meanings.
- Do not infer that an answer is correct merely because
  some words overlap with the evidence.
- Do not use your own background knowledge.
- Use only the supplied evidence.

Choose exactly one:

LABEL: SUPPORTED

LABEL: CONTRADICTED

LABEL: UNCERTAIN

Then give one short reason.

Required format:

LABEL: <label>
REASON: <short reason>
"""

    result = generate_text(
        prompt,
        max_new_tokens=100
    )

    match = re.search(
        r"LABEL\s*:\s*(SUPPORTED|CONTRADICTED|UNCERTAIN)",
        result,
        flags=re.IGNORECASE
    )

    if match:

        label = match.group(
            1
        ).upper()

    else:

        label = "UNCERTAIN"

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        result,
        flags=re.IGNORECASE
    )

    if reason_match:

        reason = reason_match.group(
            1
        ).strip()

    else:

        reason = result.strip()

    return {
        "judgment": label,
        "reason": reason,
        "raw_output": result
    }


# ============================================================
# TRUST FEATURES
# ============================================================

def get_trust_features(
    answer
):

    """
    Obtain hidden representation,
    confidence and entropy.

    Used only to exercise the existing TrustGate.
    """

    messages = [
        {
            "role": "system",
            "content":
                "Answer the question briefly and factually."
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

        outputs = model(
            **inputs,
            output_hidden_states=True
        )

    logits = outputs.logits.float()

    hidden = outputs.hidden_states[
        -1
    ].float()

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    entropy = -(
        probabilities *
        torch.log(
            probabilities.clamp(
                min=1e-12
            )
        )
    ).sum(
        dim=-1
    )

    hidden_vector = (
        hidden[0]
        .mean(dim=0)
        .detach()
        .cpu()
        .numpy()
    )

    confidence = float(
        probabilities
        .max(dim=-1)
        .values
        .mean()
        .item()
    )

    mean_entropy = float(
        entropy
        .mean()
        .item()
    )

    return {
        "hidden_state":
            hidden_vector.tolist(),

        "confidence":
            confidence,

        "entropy":
            mean_entropy
    }


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(
    case,
    index
):

    question = case[
        "question"
    ]

    a0 = case[
        "wrong_answer"
    ]

    a0_claim = case[
        "wrong_claim"
    ]

    print("\n")
    print("=" * 70)
    print(
        f"CASE {index}"
    )
    print("=" * 70)

    print("\nQUESTION:")
    print(question)

    print("\nA0 - INTENTIONALLY WRONG:")
    print(a0)

    print("\nA0 CLAIM:")
    print(a0_claim)

    # --------------------------------------------------------
    # A0 TRUSTGATE
    # --------------------------------------------------------

    a0_features = get_trust_features(
        a0
    )

    a0_trust = controller.classify(
        a0_features[
            "hidden_state"
        ],
        a0_features[
            "confidence"
        ],
        a0_features[
            "entropy"
        ]
    )

    print("\nA0 TRUSTGATE:")
    print(a0_trust)

    # --------------------------------------------------------
    # A0 VERIFICATION
    # --------------------------------------------------------

    print("\nVerifying A0...")

    a0_verification = verifier.verify(
        question,
        a0
    )

    print(
        "A0 VERIFIER:",
        a0_verification.get(
            "verification_status"
        )
    )

    print(
        "A0 SUPPORT:",
        round(
            float(
                a0_verification.get(
                    "support_score",
                    0
                )
            ),
            3
        )
    )

    print(
        "A0 CONTRADICTION:",
        round(
            float(
                a0_verification.get(
                    "contradiction_score",
                    0
                )
            ),
            3
        )
    )

    # --------------------------------------------------------
    # A0 EVIDENCE JUDGE
    # --------------------------------------------------------

    a0_judgment = judge_evidence(
        question,
        a0,
        a0_claim,
        a0_verification
    )

    print("\nA0 EVIDENCE JUDGE:")
    print(
        a0_judgment[
            "judgment"
        ]
    )

    print(
        "Reason:",
        a0_judgment[
            "reason"
        ]
    )

    # --------------------------------------------------------
    # GENERATE CORRECTION
    # --------------------------------------------------------

    correction_prompt = f"""
Question:
{question}

The initial answer is incorrect.

Initial answer:
{a0}

Initial claim:
{a0_claim}

External verification status:
{a0_verification.get("verification_status")}

Supporting evidence:
{json.dumps(
    a0_verification.get(
        "supporting_sentences",
        []
    )[:6],
    ensure_ascii=False
)}

Contradicting evidence:
{json.dumps(
    a0_verification.get(
        "contradicting_sentences",
        []
    )[:6],
    ensure_ascii=False
)}

Give the corrected answer.

Rules:
- Correct the factual error.
- Answer the original question directly.
- Use one or two short sentences.
- Preserve factual accuracy.
- Do not mention this verification process.
"""

    a1 = generate_text(
        correction_prompt,
        max_new_tokens=80
    )

    print("\nA1 - CORRECTED CANDIDATE:")
    print(a1)

    # --------------------------------------------------------
    # A1 CLAIM
    # --------------------------------------------------------

    a1_claim = extract_candidate_claim(
        question,
        a1
    )

    print("\nA1 CLAIM:")
    print(a1_claim)

    # --------------------------------------------------------
    # A1 VERIFICATION
    # --------------------------------------------------------

    print("\nVerifying A1...")

    a1_verification = verifier.verify(
        question,
        a1
    )

    print(
        "A1 VERIFIER:",
        a1_verification.get(
            "verification_status"
        )
    )

    print(
        "A1 SUPPORT:",
        round(
            float(
                a1_verification.get(
                    "support_score",
                    0
                )
            ),
            3
        )
    )

    print(
        "A1 CONTRADICTION:",
        round(
            float(
                a1_verification.get(
                    "contradiction_score",
                    0
                )
            ),
            3
        )
    )

    # --------------------------------------------------------
    # A1 EVIDENCE JUDGE
    # --------------------------------------------------------

    a1_judgment = judge_evidence(
        question,
        a1,
        a1_claim,
        a1_verification
    )

    print("\nA1 EVIDENCE JUDGE:")
    print(
        a1_judgment[
            "judgment"
        ]
    )

    print(
        "Reason:",
        a1_judgment[
            "reason"
        ]
    )

    # --------------------------------------------------------
    # A1 TRUSTGATE
    # --------------------------------------------------------

    a1_features = get_trust_features(
        a1
    )

    a1_trust = controller.classify(
        a1_features[
            "hidden_state"
        ],
        a1_features[
            "confidence"
        ],
        a1_features[
            "entropy"
        ]
    )

    print("\nA1 TRUSTGATE:")
    print(a1_trust)

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if (
        a0_judgment[
            "judgment"
        ] == "CONTRADICTED"
        and
        a1_judgment[
            "judgment"
        ] == "SUPPORTED"
    ):

        decision = {
            "decision": "ACCEPT",
            "reason":
                "EVIDENCE_CONFIRMS_CORRECTION"
        }

    else:

        decision = {
            "decision": "ROLLBACK",
            "reason":
                "CORRECTION_CRITERIA_NOT_MET"
        }

    print("\nFINAL DECISION:")
    print(decision)

    return {

        "question":
            question,

        "initial": {

            "answer":
                a0,

            "claim":
                a0_claim,

            "trust_label":
                a0_trust,

            "verification":
                a0_verification,

            "evidence_judgment":
                a0_judgment
        },

        "candidate": {

            "answer":
                a1,

            "claim":
                a1_claim,

            "trust_label":
                a1_trust,

            "verification":
                a1_verification,

            "evidence_judgment":
                a1_judgment
        },

        "decision":
            decision
    }


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    accepted = 0
    rejected = 0
    a0_contradicted = 0
    a0_supported = 0
    a0_uncertain = 0
    a1_supported = 0
    a1_uncertain = 0
    a1_contradicted = 0

    for index, case in enumerate(
        CONTROLLED_CASES,
        start=1
    ):

        try:

            result = run_case(
                case,
                index
            )

            results.append(
                result
            )

            if (
                result[
                    "decision"
                ]["decision"]
                == "ACCEPT"
            ):

                accepted += 1

            else:

                rejected += 1

            a0_label = result[
                "initial"
            ][
                "evidence_judgment"
            ][
                "judgment"
            ]

            a1_label = result[
                "candidate"
            ][
                "evidence_judgment"
            ][
                "judgment"
            ]

            if a0_label == "CONTRADICTED":
                a0_contradicted += 1

            elif a0_label == "SUPPORTED":
                a0_supported += 1

            else:
                a0_uncertain += 1

            if a1_label == "SUPPORTED":
                a1_supported += 1

            elif a1_label == "CONTRADICTED":
                a1_contradicted += 1

            else:
                a1_uncertain += 1

        except Exception as error:

            print(
                "\nERROR:",
                repr(error)
            )

            results.append(
                {
                    "question":
                        case["question"],

                    "error":
                        repr(error)
                }
            )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output = {

        "experiment":
            "TrustGate V7.1 Controlled Correction",

        "description":
            "Controlled test using intentionally incorrect "
            "answers and explicitly specified initial claims.",

        "model":
            MODEL_NAME,

        "num_cases":
            len(CONTROLLED_CASES),

        "accepted":
            accepted,

        "rejected":
            rejected,

        "initial_contradicted":
            a0_contradicted,

        "initial_supported":
            a0_supported,

        "initial_uncertain":
            a0_uncertain,

        "candidate_supported":
            a1_supported,

        "candidate_contradicted":
            a1_contradicted,

        "candidate_uncertain":
            a1_uncertain,

        "results":
            results
    }

    os.makedirs(
        os.path.dirname(
            RESULT_PATH
        ),
        exist_ok=True
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "CONTROLLED CORRECTION TEST COMPLETE"
    )
    print("=" * 70)

    print(
        f"Cases processed       : "
        f"{len(results)}/5"
    )

    print(
        f"Accepted              : "
        f"{accepted}"
    )

    print(
        f"Rejected              : "
        f"{rejected}"
    )

    print(
        f"A0 contradicted       : "
        f"{a0_contradicted}"
    )

    print(
        f"A0 supported          : "
        f"{a0_supported}"
    )

    print(
        f"A0 uncertain          : "
        f"{a0_uncertain}"
    )

    print(
        f"A1 supported          : "
        f"{a1_supported}"
    )

    print(
        f"A1 contradicted       : "
        f"{a1_contradicted}"
    )

    print(
        f"A1 uncertain          : "
        f"{a1_uncertain}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":
    main()