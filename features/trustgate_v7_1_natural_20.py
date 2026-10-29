"""
TrustGate V7.1 Natural Held-Out Evaluation
-------------------------------------------

Uses 20 questions from the existing 290-question held-out pool.

Important:
- TrustGate model remains frozen.
- No held-out labels are used for tuning.
- Initial answers are generated normally by Qwen.
- External verification is used during self-improvement.
- Evidence judge evaluates the central claim.
- Candidate is accepted only when evidence supports it.
- Results are saved for later analysis.

This is a development evaluation, not a final benchmark.
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

HELDOUT_PATH = os.path.join(
    ROOT,
    "outputs",
    "trustgate_heldout_290.json"
)

RESULT_PATH = os.path.join(
    ROOT,
    "outputs",
    "trustgate_v7_1_natural_20_results.json"
)

sys.path.insert(0, ROOT)

from features.trustgate_controller import TrustGateController
from features.external_verifier import ExternalVerifier


# ============================================================
# SETTINGS
# ============================================================

NUM_QUESTIONS = 20

MAX_NEW_TOKENS = 80

MAX_ATTEMPTS = 1


# ============================================================
# LOAD HELD-OUT DATA
# ============================================================

with open(
    HELDOUT_PATH,
    "r",
    encoding="utf-8"
) as f:

    heldout = json.load(f)


if not isinstance(heldout, list):
    raise ValueError(
        "Held-out dataset must be a list."
    )


if len(heldout) < NUM_QUESTIONS:
    raise ValueError(
        f"Expected at least {NUM_QUESTIONS} "
        f"held-out questions."
    )


cases = heldout[:NUM_QUESTIONS]


print("=" * 70)
print("TRUSTGATE V7.1 NATURAL HELD-OUT EVALUATION")
print("=" * 70)

print(
    f"\nHeld-out pool size : {len(heldout)}"
)

print(
    f"Evaluation size    : {len(cases)}"
)

print(
    "\nNo held-out correctness labels will be used "
    "for model decisions."
)


# ============================================================
# LOAD QWEN
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

print("\nLoading TrustGate controller...")

controller = TrustGateController()

print("TrustGate loaded.")


# ============================================================
# LOAD VERIFIER
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
                "Do not provide unnecessary background."
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

Extract the single central factual claim
made by this answer.

Rules:
- Preserve negation.
- Preserve the meaning of yes/no answers.
- Return one sentence only.
- Do not add an explanation.
- Do not introduce new facts.
- Do not reverse the meaning of the answer.

Return only the central claim.
"""

    claim = generate_answer(
        question,
        prompt
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
You are an evidence judge.

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

Judge ONLY whether the supplied evidence supports
or contradicts the central claim.

Important:
- Preserve negation.
- "X is true" and "X is not true" are opposite.
- Do not rely on outside knowledge.
- Do not treat keyword overlap alone as proof.
- If evidence is ambiguous, use UNCERTAIN.

Return exactly:

LABEL: SUPPORTED

or

LABEL: CONTRADICTED

or

LABEL: UNCERTAIN

Then:

REASON: one short sentence
"""

    result = generate_answer(
        question,
        prompt
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
# ONE QUESTION
# ============================================================

def run_case(
    case,
    index
):

    question = case[
        "question"
    ]

    print("\n")
    print("=" * 70)
    print(
        f"QUESTION {index}/{NUM_QUESTIONS}"
    )
    print("=" * 70)

    print(
        "Held-out ID:",
        case.get("heldout_id")
    )

    print(
        "Category:",
        case.get("category")
    )

    print(
        "\nQuestion:",
        question
    )

    # --------------------------------------------------------
    # INITIAL ANSWER
    # --------------------------------------------------------

    a0 = generate_answer(
        question,
        question
    )

    print(
        "\nA0:",
        a0
    )

    # --------------------------------------------------------
    # A0 TRUST
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

    print(
        "A0 TrustGate:",
        a0_trust
    )

    # --------------------------------------------------------
    # INITIAL CLAIM
    # --------------------------------------------------------

    a0_claim = extract_claim(
        question,
        a0
    )

    print(
        "A0 Claim:",
        a0_claim
    )

    # --------------------------------------------------------
    # VERIFY INITIAL ANSWER
    # --------------------------------------------------------

    a0_verification = verifier.verify(
        question,
        a0
    )

    a0_judgment = judge_evidence(
        question,
        a0,
        a0_claim,
        a0_verification
    )

    print(
        "A0 Evidence:",
        a0_judgment[
            "judgment"
        ]
    )

    # --------------------------------------------------------
    # DECIDE WHETHER TO TRY CORRECTION
    # --------------------------------------------------------

    if (
        a0_judgment[
            "judgment"
        ] == "CONTRADICTED"
    ):

        correction_prompt = f"""
Question:
{question}

The following answer has evidence contradicting
its central claim:

Answer:
{a0}

Central claim:
{a0_claim}

External supporting evidence:
{json.dumps(
    a0_verification.get(
        "supporting_sentences",
        []
    )[:6],
    ensure_ascii=False
)}

External contradicting evidence:
{json.dumps(
    a0_verification.get(
        "contradicting_sentences",
        []
    )[:6],
    ensure_ascii=False
)}

Produce a corrected answer.

Rules:
- Correct the factual error.
- Answer the original question.
- Keep the answer short.
- Do not mention this verification process.
"""

        a1 = generate_answer(
            question,
            correction_prompt
        )

        a1_claim = extract_claim(
            question,
            a1
        )

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

        if (
            a1_judgment[
                "judgment"
            ] == "SUPPORTED"
        ):

            decision = {
                "decision": "ACCEPT",
                "reason":
                    "EVIDENCE_SUPPORTS_CORRECTION"
            }

        else:

            decision = {
                "decision": "ROLLBACK",
                "reason":
                    "CORRECTION_NOT_SUPPORTED"
            }

        print(
            "A1:",
            a1
        )

        print(
            "A1 Claim:",
            a1_claim
        )

        print(
            "A1 Evidence:",
            a1_judgment[
                "judgment"
            ]
        )

        print(
            "A1 TrustGate:",
            a1_trust
        )

        print(
            "Decision:",
            decision
        )

    else:

        a1 = None
        a1_claim = None
        a1_verification = None
        a1_judgment = None
        a1_trust = None

        decision = {
            "decision": "KEEP_INITIAL",
            "reason":
                "INITIAL_NOT_EXTERNALLY_CONTRADICTED"
        }

        print(
            "No correction attempted."
        )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "heldout_id":
            case.get("heldout_id"),

        "truthfulqa_csv_index":
            case.get(
                "truthfulqa_csv_index"
            ),

        "question":
            question,

        "category":
            case.get("category"),

        "type":
            case.get("type"),

        "initial": {

            "answer":
                a0,

            "claim":
                a0_claim,

            "trust_label":
                a0_trust,

            "confidence":
                a0_features[
                    "confidence"
                ],

            "entropy":
                a0_features[
                    "entropy"
                ],

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

    counts = {

        "keep_initial":
            0,

        "accepted":
            0,

        "rollback":
            0,

        "initial_supported":
            0,

        "initial_contradicted":
            0,

        "initial_uncertain":
            0,

        "candidate_supported":
            0,

        "candidate_contradicted":
            0,

        "candidate_uncertain":
            0
    }

    for index, case in enumerate(
        cases,
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

            decision = result[
                "decision"
            ][
                "decision"
            ]

            if decision == "ACCEPT":
                counts[
                    "accepted"
                ] += 1

            elif decision == "ROLLBACK":
                counts[
                    "rollback"
                ] += 1

            else:
                counts[
                    "keep_initial"
                ] += 1

            initial_judgment = result[
                "initial"
            ][
                "evidence_judgment"
            ][
                "judgment"
            ]

            if initial_judgment == "SUPPORTED":

                counts[
                    "initial_supported"
                ] += 1

            elif initial_judgment == "CONTRADICTED":

                counts[
                    "initial_contradicted"
                ] += 1

            else:

                counts[
                    "initial_uncertain"
                ] += 1

            candidate_judgment = result[
                "candidate"
            ][
                "evidence_judgment"
            ]

            if candidate_judgment is not None:

                label = candidate_judgment[
                    "judgment"
                ]

                if label == "SUPPORTED":

                    counts[
                        "candidate_supported"
                    ] += 1

                elif label == "CONTRADICTED":

                    counts[
                        "candidate_contradicted"
                    ] += 1

                else:

                    counts[
                        "candidate_uncertain"
                    ] += 1

        except Exception as error:

            print(
                "\nERROR:",
                repr(error)
            )

            results.append(
                {
                    "heldout_id":
                        case.get(
                            "heldout_id"
                        ),

                    "question":
                        case.get(
                            "question"
                        ),

                    "error":
                        repr(error)
                }
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = {

        "experiment":
            "TrustGate V7.1 Natural Held-Out 20",

        "model":
            MODEL_NAME,

        "heldout_source":
            "outputs/trustgate_heldout_290.json",

        "num_cases":
            NUM_QUESTIONS,

        "counts":
            counts,

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
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "NATURAL HELD-OUT 20 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Cases processed       : "
        f"{len(results)}/{NUM_QUESTIONS}"
    )

    print(
        f"Initial supported     : "
        f"{counts['initial_supported']}"
    )

    print(
        f"Initial contradicted  : "
        f"{counts['initial_contradicted']}"
    )

    print(
        f"Initial uncertain     : "
        f"{counts['initial_uncertain']}"
    )

    print(
        f"Candidate supported   : "
        f"{counts['candidate_supported']}"
    )

    print(
        f"Candidate contradicted: "
        f"{counts['candidate_contradicted']}"
    )

    print(
        f"Candidate uncertain   : "
        f"{counts['candidate_uncertain']}"
    )

    print(
        f"Accepted              : "
        f"{counts['accepted']}"
    )

    print(
        f"Rollback              : "
        f"{counts['rollback']}"
    )

    print(
        f"Keep initial          : "
        f"{counts['keep_initial']}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":
    main()