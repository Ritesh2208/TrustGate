import os
import re
import json
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from features.trustgate_controller import TrustGateController
from features.external_verifier import ExternalVerifier


# ============================================================
# TRUSTGATE V9
# MULTI-SOURCE EVIDENCE CONSENSUS SELF-IMPROVEMENT
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_PATH = (
    "outputs/trustgate_v7_1_natural_270_results.json"
)

OUTPUT_PATH = (
    "outputs/trustgate_v9_multi_source_results.json"
)

MAX_NEW_TOKENS = 60

MIN_SUPPORTING_SOURCES = 2


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("TRUSTGATE V9")
print("MULTI-SOURCE EVIDENCE CONSENSUS SELF-IMPROVEMENT")
print("=" * 72)

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:
    v7_data = json.load(f)

v7_results = v7_data["results"]


# ============================================================
# SELECT THE SAME 30 CORRECTION CASES
# ============================================================

correction_cases = [
    x
    for x in v7_results
    if x.get("initial", {})
        .get("evidence_judgment", {})
        .get("judgment") == "CONTRADICTED"
]

unique_cases = {}

for item in correction_cases:
    unique_cases[item["heldout_id"]] = item

correction_cases = list(
    unique_cases.values()
)

print(
    "\nTotal V7.1 cases:",
    len(v7_results)
)

print(
    "V9 correction cases:",
    len(correction_cases)
)

if len(correction_cases) != 30:
    raise RuntimeError(
        "Expected 30 correction cases, "
        f"found {len(correction_cases)}."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen...")

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

def generate_text(
    instruction
):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict factual verification "
                "assistant. Follow the requested output "
                "format exactly. Do not invent evidence."
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
        k: v.to(model.device)
        for k, v in inputs.items()
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
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    instruction
):

    return generate_text(
        instruction
    )


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

Extract the SINGLE central factual claim contained
in the answer.

Rules:
- Preserve negation.
- Preserve yes/no meaning.
- Preserve names and numbers.
- Do not add information.
- Do not explain.
- Do not combine multiple claims.
- Return one short sentence only.

Return only the claim.
"""

    claim = generate_text(
        prompt
    )

    claim = re.sub(
        r"^(Claim|Central claim)\s*:\s*",
        "",
        claim,
        flags=re.IGNORECASE
    )

    lines = [
        x.strip()
        for x in claim.splitlines()
        if x.strip()
    ]

    if lines:
        claim = lines[0]

    return claim.strip()


# ============================================================
# SOURCE-LEVEL CONSENSUS JUDGE
# ============================================================

def judge_source_consensus(
    question,
    claim,
    verification
):

    sources = verification.get(
        "sources",
        []
    )

    if not sources:

        return {
            "consensus": "INSUFFICIENT",
            "supporting_source_count": 0,
            "contradicting_source_count": 0,
            "source_judgments": []
        }

    source_blocks = []

    for i, source in enumerate(
        sources,
        start=1
    ):

        title = source.get(
            "title",
            f"Source {i}"
        )

        sentences = source.get(
            "sentences",
            []
        )

        text = " ".join(
            str(x)
            for x in sentences
        )

        # Keep prompt size controlled.
        text = text[:5000]

        source_blocks.append(
            f"""
SOURCE {i}
TITLE:
{title}

EVIDENCE:
{text}
"""
        )

    evidence_text = "\n".join(
        source_blocks
    )

    prompt = f"""
You are a strict multi-source factual
consensus judge.

QUESTION:
{question}

CENTRAL CLAIM:
{claim}

Below are independent retrieved sources.

{evidence_text}

For EACH source determine whether it:

SUPPORTS the central claim,
CONTRADICTS the central claim,
or is IRRELEVANT/INSUFFICIENT.

Important rules:

1. Judge the complete meaning of the claim.
2. Preserve negation.
3. Do not infer "everyone", "always",
   "never", or universal claims unless the
   source actually supports that scope.
4. Shared keywords do NOT mean support.
5. A source is SUPPORTING only if its evidence
   actually establishes the claim.
6. Do not use outside knowledge.
7. Two sources count as independent only if
   they are separate retrieved sources.
8. If evidence is ambiguous, mark it
   IRRELEVANT or INSUFFICIENT.

After judging every source, determine:

CONSENSUS: SUPPORTED
only when at least two independent sources
support the SAME factual claim and there is
no strong contradiction.

CONSENSUS: CONTRADICTED
when reliable source evidence clearly
contradicts the claim.

CONSENSUS: INSUFFICIENT
when fewer than two sources clearly support
the claim, or evidence is ambiguous.

Return exactly this structure:

SOURCE 1: SUPPORTS
SOURCE 2: CONTRADICTS
...

SUPPORTING_SOURCES: number
CONTRADICTING_SOURCES: number

CONSENSUS: SUPPORTED

REASON: one short sentence
"""

    raw = generate_text(
        prompt
    )

    source_judgments = []

    for i in range(
        1,
        len(sources) + 1
    ):

        pattern = (
            rf"SOURCE\s*{i}\s*:\s*"
            r"(SUPPORTS|CONTRADICTS|"
            r"IRRELEVANT|INSUFFICIENT)"
        )

        match = re.search(
            pattern,
            raw,
            flags=re.IGNORECASE
        )

        if match:

            judgment = (
                match.group(1)
                .upper()
            )

        else:

            judgment = "INSUFFICIENT"

        source_judgments.append({
            "source_index": i,
            "judgment": judgment,
            "title": sources[i - 1].get(
                "title",
                f"Source {i}"
            )
        })

    supporting_count = sum(
        1
        for x in source_judgments
        if x["judgment"] == "SUPPORTS"
    )

    contradicting_count = sum(
        1
        for x in source_judgments
        if x["judgment"] == "CONTRADICTS"
    )

    consensus_match = re.search(
        r"CONSENSUS\s*:\s*"
        r"(SUPPORTED|CONTRADICTED|INSUFFICIENT)",
        raw,
        flags=re.IGNORECASE
    )

    if consensus_match:

        consensus = (
            consensus_match
            .group(1)
            .upper()
        )

    else:

        consensus = "INSUFFICIENT"

    # --------------------------------------------------------
    # HARD SAFETY RULE
    # --------------------------------------------------------
    #
    # The LLM cannot declare SUPPORTED unless
    # at least two actual sources were marked
    # SUPPORTS.
    #

    if supporting_count >= MIN_SUPPORTING_SOURCES:

        if contradicting_count == 0:

            consensus = "SUPPORTED"

        elif (
            supporting_count
            > contradicting_count
        ):

            consensus = "SUPPORTED"

        else:

            consensus = "INSUFFICIENT"

    else:

        if contradicting_count >= 1:

            consensus = "CONTRADICTED"

        else:

            consensus = "INSUFFICIENT"

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        raw,
        flags=re.IGNORECASE
    )

    if reason_match:

        reason = (
            reason_match
            .group(1)
            .strip()
        )

    else:

        reason = (
            "Multi-source consensus "
            "could not be established."
        )

    return {
        "consensus": consensus,
        "supporting_source_count": (
            supporting_count
        ),
        "contradicting_source_count": (
            contradicting_count
        ),
        "source_judgments": (
            source_judgments
        ),
        "reason": reason,
        "raw_output": raw
    }


# ============================================================
# TRUST REPRESENTATION
# ============================================================

def get_trust_features(
    question,
    answer
):

    system_text = (
        "You are a factual question answering "
        "assistant."
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
        k: v.to(model.device)
        for k, v in encoded.items()
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
            "entropy": 0.0
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

    probs = torch.softmax(
        prediction_logits,
        dim=-1
    )

    actual_probs = probs[
        torch.arange(
            usable,
            device=model.device
        ),
        generated_token_ids
    ]

    entropy_values = -torch.sum(
        probs
        * torch.log(
            probs.clamp_min(1e-12)
        ),
        dim=-1
    )

    confidence = float(
        actual_probs.mean()
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
            hidden_generated.mean(dim=0)
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
            "Expected hidden dimension 896, "
            f"got {hidden_vector.shape[0]}"
        )

    return {
        "hidden_state": hidden_vector.tolist(),
        "confidence": confidence,
        "entropy": entropy
    }


# ============================================================
# EVIDENCE-CALIBRATED TRUST
# ============================================================

def final_trust_label(
    raw_trust_label,
    consensus
):

    # Strong multi-source agreement can upgrade
    # the final evidence-calibrated trust state.

    if consensus == "SUPPORTED":

        return "RELIABLE"

    if consensus == "CONTRADICTED":

        return "UNRELIABLE"

    return "UNCERTAIN"


# ============================================================
# CORRECTION GENERATION
# ============================================================

def generate_correction(
    question,
    old_answer,
    old_claim,
    verification,
    consensus
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

The original claim did not pass
multi-source verification.

Multi-source consensus:

{consensus["consensus"]}

Number of supporting sources:

{consensus["supporting_source_count"]}

Number of contradicting sources:

{consensus["contradicting_source_count"]}

Supporting evidence:

{json.dumps(
    supporting[:8],
    ensure_ascii=False
)}

Contradicting evidence:

{json.dumps(
    contradicting[:8],
    ensure_ascii=False
)}

Write a corrected answer to the original
question.

Rules:

- Correct the factual error.
- Use only information supported by evidence.
- Preserve negation.
- Answer directly.
- One atomic factual claim.
- No reasoning.
- No explanation.
- No background.
- No unrelated facts.

Return only the corrected answer.
"""

    return generate_text(
        prompt
    )


# ============================================================
# VERIFY ANSWER
# ============================================================

def verify_answer(
    question,
    answer
):

    claim = extract_claim(
        question,
        answer
    )

    verification = verifier.verify(
        question,
        answer
    )

    consensus = judge_source_consensus(
        question,
        claim,
        verification
    )

    return {
        "claim": claim,
        "verification": verification,
        "consensus": consensus
    }


# ============================================================
# PROCESS CASE
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

    print("\n" + "=" * 72)
    print(
        f"V9 CASE {index}/{total}"
    )
    print("=" * 72)

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

    # --------------------------------------------------------
    # A0 TRUST
    # --------------------------------------------------------

    a0_features = get_trust_features(
        question,
        a0
    )

    a0_raw_trust = controller.classify(
        a0_features["hidden_state"],
        a0_features["confidence"],
        a0_features["entropy"]
    )

    # --------------------------------------------------------
    # A0 VERIFICATION
    # --------------------------------------------------------

    a0_verified = verify_answer(
        question,
        a0
    )

    a0_consensus = a0_verified[
        "consensus"
    ]

    a0_final_trust = final_trust_label(
        a0_raw_trust,
        a0_consensus["consensus"]
    )

    print(
        "\nA0 Raw TrustGate:",
        a0_raw_trust
    )

    print(
        "A0 Sources:",
        len(
            a0_verified[
                "verification"
            ].get("sources", [])
        )
    )

    print(
        "A0 Supporting sources:",
        a0_consensus[
            "supporting_source_count"
        ]
    )

    print(
        "A0 Consensus:",
        a0_consensus[
            "consensus"
        ]
    )

    print(
        "A0 Final Trust:",
        a0_final_trust
    )

    # --------------------------------------------------------
    # CORRECTION IS NEEDED IF:
    #
    # 1. Original V7.1 classified it as contradicted
    # OR
    # 2. Multi-source consensus contradicts it
    #
    # --------------------------------------------------------

    original_trigger = (
        initial.get(
            "evidence_judgment",
            {}
        ).get(
            "judgment"
        ) == "CONTRADICTED"
    )

    if not original_trigger:

        return {
            "heldout_id": heldout_id,
            "question": question,
            "status": "NO_CORRECTION_REQUIRED",
            "decision": {
                "decision": "KEEP_INITIAL",
                "reason": "NO_CORRECTION_TRIGGER"
            }
        }

    # --------------------------------------------------------
    # GENERATE A1
    # --------------------------------------------------------

    a1 = generate_correction(
        question,
        a0,
        a0_verified["claim"],
        a0_verified["verification"],
        a0_consensus
    )

    print(
        "\nA1:",
        a1
    )

    # --------------------------------------------------------
    # A1 VERIFICATION
    # --------------------------------------------------------

    a1_features = get_trust_features(
        question,
        a1
    )

    a1_raw_trust = controller.classify(
        a1_features["hidden_state"],
        a1_features["confidence"],
        a1_features["entropy"]
    )

    a1_verified = verify_answer(
        question,
        a1
    )

    a1_consensus = a1_verified[
        "consensus"
    ]

    a1_final_trust = final_trust_label(
        a1_raw_trust,
        a1_consensus["consensus"]
    )

    print(
        "\nA1 Raw TrustGate:",
        a1_raw_trust
    )

    print(
        "A1 Sources:",
        len(
            a1_verified[
                "verification"
            ].get("sources", [])
        )
    )

    print(
        "A1 Supporting sources:",
        a1_consensus[
            "supporting_source_count"
        ]
    )

    print(
        "A1 Consensus:",
        a1_consensus[
            "consensus"
        ]
    )

    print(
        "A1 Final Trust:",
        a1_final_trust
    )

    # --------------------------------------------------------
    # ACCEPTANCE LOGIC
    # --------------------------------------------------------

    a1_supported = (
        a1_consensus[
            "consensus"
        ] == "SUPPORTED"
        and
        a1_consensus[
            "supporting_source_count"
        ] >= MIN_SUPPORTING_SOURCES
        and
        a1_consensus[
            "contradicting_source_count"
        ] == 0
    )

    a1_contradicted = (
        a1_consensus[
            "consensus"
        ] == "CONTRADICTED"
    )

    raw_priority = {
        "UNRELIABLE": 0,
        "UNCERTAIN": 1,
        "RELIABLE": 2
    }

    raw_a0_priority = raw_priority.get(
        a0_raw_trust,
        -1
    )

    raw_a1_priority = raw_priority.get(
        a1_raw_trust,
        -1
    )

    final_a0_priority = raw_priority.get(
        a0_final_trust,
        -1
    )

    final_a1_priority = raw_priority.get(
        a1_final_trust,
        -1
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
    # V9 DECISION
    # --------------------------------------------------------

    if a1_supported:

        decision = {
            "decision": "ACCEPT",
            "reason": (
                "MULTI_SOURCE_CONSENSUS_SUPPORTED"
            )
        }

    elif a1_contradicted:

        decision = {
            "decision": "ROLLBACK",
            "reason": (
                "MULTI_SOURCE_CONTRADICTION"
            )
        }

    else:

        decision = {
            "decision": "ROLLBACK",
            "reason": (
                "INSUFFICIENT_MULTI_SOURCE_CONSENSUS"
            )
        }

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\nConfidence:",
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
        "\nFINAL V9 DECISION:",
        decision
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "heldout_id": heldout_id,

        "question": question,

        "initial": {

            "answer": a0,

            "claim": a0_verified[
                "claim"
            ],

            "raw_trust_label": (
                a0_raw_trust
            ),

            "final_trust_label": (
                a0_final_trust
            ),

            "confidence": (
                a0_features[
                    "confidence"
                ]
            ),

            "entropy": (
                a0_features[
                    "entropy"
                ]
            ),

            "verification": (
                a0_verified[
                    "verification"
                ]
            ),

            "multi_source_consensus": (
                a0_consensus
            )
        },

        "candidate": {

            "answer": a1,

            "claim": a1_verified[
                "claim"
            ],

            "raw_trust_label": (
                a1_raw_trust
            ),

            "final_trust_label": (
                a1_final_trust
            ),

            "confidence": (
                a1_features[
                    "confidence"
                ]
            ),

            "entropy": (
                a1_features[
                    "entropy"
                ]
            ),

            "verification": (
                a1_verified[
                    "verification"
                ]
            ),

            "multi_source_consensus": (
                a1_consensus
            )
        },

        "comparison": {

            "raw_trust_improved": (
                raw_a1_priority
                >
                raw_a0_priority
            ),

            "final_trust_improved": (
                final_a1_priority
                >
                final_a0_priority
            ),

            "confidence_improved": (
                confidence_improved
            ),

            "entropy_improved": (
                entropy_improved
            ),

            "supporting_sources": (
                a1_consensus[
                    "supporting_source_count"
                ]
            ),

            "contradicting_sources": (
                a1_consensus[
                    "contradicting_source_count"
                ]
            )
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
    errors = 0

    initial_reliable = 0
    candidate_reliable = 0

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

            results.append(
                result
            )

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

            if (
                result.get(
                    "initial",
                    {}
                ).get(
                    "final_trust_label"
                )
                == "RELIABLE"
            ):

                initial_reliable += 1

            if (
                result.get(
                    "candidate",
                    {}
                ).get(
                    "final_trust_label"
                )
                == "RELIABLE"
            ):

                candidate_reliable += 1

        except Exception as e:

            errors += 1

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
            "TrustGate V9 Multi-Source "
            "Evidence Consensus"
        ),

        "model": MODEL_NAME,

        "source": INPUT_PATH,

        "minimum_supporting_sources": (
            MIN_SUPPORTING_SOURCES
        ),

        "num_cases": len(
            correction_cases
        ),

        "counts": {

            "accepted": accepted,

            "rollback": rollback,

            "errors": errors,

            "initial_final_reliable": (
                initial_reliable
            ),

            "candidate_final_reliable": (
                candidate_reliable
            )
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
    print("=" * 72)
    print("TRUSTGATE V9 COMPLETE")
    print("=" * 72)

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
        "Errors:",
        errors
    )

    print(
        "Initial final RELIABLE:",
        initial_reliable
    )

    print(
        "Candidate final RELIABLE:",
        candidate_reliable
    )

    print(
        "\nAcceptance rate:",
        round(
            100 * accepted / max(
                len(results),
                1
            ),
            2
        ),
        "%"
    )

    print(
        "\nSaved:",
        os.path.abspath(
            OUTPUT_PATH
        )
    )


if __name__ == "__main__":
    main()