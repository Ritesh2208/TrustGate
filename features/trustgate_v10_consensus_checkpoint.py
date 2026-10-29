import os
import re
import json
import tempfile
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from features.trustgate_controller import TrustGateController
from features.external_verifier import ExternalVerifier


# ============================================================
# TRUSTGATE V10
# MULTI-SOURCE CONSENSUS + CHECKPOINTED SELF-IMPROVEMENT
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_PATH = (
    "outputs/trustgate_v7_1_natural_270_results.json"
)

OUTPUT_PATH = (
    "outputs/trustgate_v10_consensus_checkpoint_results.json"
)

CHECKPOINT_PATH = (
    "outputs/trustgate_v10_consensus_checkpoint.json"
)

MAX_NEW_TOKENS = 60

# Minimum number of source votes required for a
# majority decision.
MIN_SOURCES_FOR_MAJORITY = 2


# ============================================================
# SAFE ATOMIC JSON SAVE
# ============================================================

def atomic_json_save(
    data,
    path
):

    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    fd, temp_path = tempfile.mkstemp(
        prefix=".trustgate_tmp_",
        suffix=".json",
        dir=directory
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

            f.flush()

            os.fsync(
                f.fileno()
            )

        os.replace(
            temp_path,
            path
        )

    except Exception:

        try:

            os.remove(
                temp_path
            )

        except OSError:
            pass

        raise


# ============================================================
# LOAD INPUT
# ============================================================

print("=" * 72)
print("TRUSTGATE V10")
print("MULTI-SOURCE CONSENSUS + CHECKPOINTED SELF-IMPROVEMENT")
print("=" * 72)

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:

    v7_data = json.load(f)

v7_results = v7_data["results"]


# ============================================================
# SELECT EXACTLY THE 30 V7.1 CORRECTION CASES
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

    unique_cases[
        item["heldout_id"]
    ] = item

correction_cases = list(
    unique_cases.values()
)

print(
    "\nV7.1 total cases:",
    len(v7_results)
)

print(
    "V10 correction cases:",
    len(correction_cases)
)

if len(correction_cases) != 30:

    raise RuntimeError(
        "Expected exactly 30 correction cases, "
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
                "You are a strict factual assistant. "
                "Follow the requested format exactly. "
                "Do not invent facts or evidence."
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

Extract ONLY the single central factual claim.

Rules:

- Preserve negation.
- Preserve yes/no meaning.
- Preserve names and numbers.
- Do not add information.
- Do not explain.
- Do not combine multiple claims.
- Return one short sentence.

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
# MULTI-SOURCE JUDGE
# ============================================================

def judge_sources(
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
            "consensus": "UNCERTAIN",
            "supporting_source_count": 0,
            "contradicting_source_count": 0,
            "insufficient_source_count": 0,
            "total_sources": 0,
            "source_judgments": [],
            "reason": "No retrieved sources."
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

        evidence = " ".join(
            str(x)
            for x in sentences
        )

        evidence = evidence[:5000]

        source_blocks.append(
            f"""
SOURCE {i}

TITLE:
{title}

EVIDENCE:
{evidence}
"""
        )

    evidence_text = "\n".join(
        source_blocks
    )

    prompt = f"""
You are a strict multi-source factual
evidence judge.

QUESTION:
{question}

CENTRAL CLAIM:
{claim}

Retrieved sources:

{evidence_text}

For each source classify its evidence as exactly
one of:

SUPPORTS
CONTRADICTS
INSUFFICIENT

Rules:

1. Judge the complete meaning of the claim.
2. Preserve negation.
3. Do not use keyword overlap alone.
4. A source supports the claim only if its
   evidence actually supports the factual claim.
5. A source contradicts the claim only if its
   evidence actually contradicts the claim.
6. Otherwise mark INSUFFICIENT.
7. Do not assume universal statements such as
   "everyone", "always", or "never" unless the
   source actually establishes that scope.
8. Do not use outside knowledge.

Return exactly:

SOURCE 1: SUPPORTS
SOURCE 2: INSUFFICIENT
SOURCE 3: CONTRADICTS

Do this for every source.
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
            r"INSUFFICIENT)"
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
            "title": sources[
                i - 1
            ].get(
                "title",
                f"Source {i}"
            ),
            "judgment": judgment
        })

    supporting = sum(
        x["judgment"] == "SUPPORTS"
        for x in source_judgments
    )

    contradicting = sum(
        x["judgment"] == "CONTRADICTS"
        for x in source_judgments
    )

    insufficient = sum(
        x["judgment"] == "INSUFFICIENT"
        for x in source_judgments
    )

    total = len(
        source_judgments
    )

    # ========================================================
    # MAJORITY CONSENSUS
    # ========================================================
    #
    # 2/3 support       -> SUPPORTED
    # 2/3 contradiction -> CONTRADICTED
    # 1/3 vs 1/3        -> UNCERTAIN
    #
    # More generally:
    #
    # supporting > contradicting
    #     and supporting >= 2
    #     -> SUPPORTED
    #
    # contradicting > supporting
    #     and contradicting >= 2
    #     -> CONTRADICTED
    #
    # Otherwise -> UNCERTAIN
    # ========================================================

    if (
        supporting >= MIN_SOURCES_FOR_MAJORITY
        and
        supporting > contradicting
    ):

        consensus = "SUPPORTED"

    elif (
        contradicting >= MIN_SOURCES_FOR_MAJORITY
        and
        contradicting > supporting
    ):

        consensus = "CONTRADICTED"

    else:

        consensus = "UNCERTAIN"

    if consensus == "SUPPORTED":

        reason = (
            f"{supporting} of {total} sources "
            "support the claim, forming a "
            "supporting majority."
        )

    elif consensus == "CONTRADICTED":

        reason = (
            f"{contradicting} of {total} sources "
            "contradict the claim, forming a "
            "contradicting majority."
        )

    else:

        reason = (
            f"Evidence is not sufficiently "
            f"consistent: {supporting} supporting, "
            f"{contradicting} contradicting, "
            f"{insufficient} insufficient."
        )

    return {
        "consensus": consensus,
        "supporting_source_count": supporting,
        "contradicting_source_count": contradicting,
        "insufficient_source_count": insufficient,
        "total_sources": total,
        "source_judgments": source_judgments,
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

def evidence_trust_label(
    consensus
):

    if consensus == "SUPPORTED":

        return "RELIABLE"

    if consensus == "CONTRADICTED":

        return "UNRELIABLE"

    return "UNCERTAIN"


# ============================================================
# VERIFY ONE ANSWER
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

    consensus = judge_sources(
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
# GENERATE CORRECTION
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

The original answer requires correction.

Multi-source consensus:

{consensus["consensus"]}

Supporting source count:

{consensus["supporting_source_count"]}

Contradicting source count:

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

Produce a corrected answer to the original
question.

Rules:

- Correct the factual error.
- Use only supported information.
- Preserve negation.
- Answer the original question directly.
- Give one atomic factual answer.
- Do not explain reasoning.
- Do not mention verification.
- Do not add unrelated facts.

Return ONLY the corrected answer.
"""

    return generate_text(
        prompt
    )


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(
    item,
    index,
    total
):

    heldout_id = item[
        "heldout_id"
    ]

    question = item[
        "question"
    ]

    initial = item[
        "initial"
    ]

    a0 = initial[
        "answer"
    ]

    print("\n" + "=" * 72)
    print(
        f"V10 CASE {index}/{total}"
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

    # ========================================================
    # A0 TRUST
    # ========================================================

    a0_features = get_trust_features(
        question,
        a0
    )

    a0_raw_trust = controller.classify(
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

    # ========================================================
    # A0 EVIDENCE
    # ========================================================

    a0_verified = verify_answer(
        question,
        a0
    )

    a0_consensus = a0_verified[
        "consensus"
    ]

    a0_evidence_trust = (
        evidence_trust_label(
            a0_consensus[
                "consensus"
            ]
        )
    )

    print(
        "\nA0 Raw TrustGate:",
        a0_raw_trust
    )

    print(
        "A0 Sources:",
        a0_consensus[
            "total_sources"
        ]
    )

    print(
        "A0 Supported:",
        a0_consensus[
            "supporting_source_count"
        ]
    )

    print(
        "A0 Contradicted:",
        a0_consensus[
            "contradicting_source_count"
        ]
    )

    print(
        "A0 Consensus:",
        a0_consensus[
            "consensus"
        ]
    )

    print(
        "A0 Evidence-calibrated Trust:",
        a0_evidence_trust
    )

    # ========================================================
    # A0 MUST BE A CORRECTION CASE
    # ========================================================

    original_trigger = (
        initial.get(
            "evidence_judgment",
            {}
        ).get(
            "judgment"
        )
        == "CONTRADICTED"
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

    # ========================================================
    # GENERATE A1
    # ========================================================

    a1 = generate_correction(
        question,
        a0,
        a0_verified[
            "claim"
        ],
        a0_verified[
            "verification"
        ],
        a0_consensus
    )

    print(
        "\nA1:",
        a1
    )

    # ========================================================
    # A1 TRUST
    # ========================================================

    a1_features = get_trust_features(
        question,
        a1
    )

    a1_raw_trust = controller.classify(
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

    # ========================================================
    # A1 EVIDENCE
    # ========================================================

    a1_verified = verify_answer(
        question,
        a1
    )

    a1_consensus = a1_verified[
        "consensus"
    ]

    a1_evidence_trust = (
        evidence_trust_label(
            a1_consensus[
                "consensus"
            ]
        )
    )

    print(
        "\nA1 Raw TrustGate:",
        a1_raw_trust
    )

    print(
        "A1 Sources:",
        a1_consensus[
            "total_sources"
        ]
    )

    print(
        "A1 Supported:",
        a1_consensus[
            "supporting_source_count"
        ]
    )

    print(
        "A1 Contradicted:",
        a1_consensus[
            "contradicting_source_count"
        ]
    )

    print(
        "A1 Consensus:",
        a1_consensus[
            "consensus"
        ]
    )

    print(
        "A1 Evidence-calibrated Trust:",
        a1_evidence_trust
    )

    # ========================================================
    # CONFIDENCE / ENTROPY
    # ========================================================

    confidence_delta = (
        a1_features[
            "confidence"
        ]
        -
        a0_features[
            "confidence"
        ]
    )

    entropy_delta = (
        a1_features[
            "entropy"
        ]
        -
        a0_features[
            "entropy"
        ]
    )

    print(
        "\nConfidence:",
        round(
            a0_features[
                "confidence"
            ],
            4
        ),
        "->",
        round(
            a1_features[
                "confidence"
            ],
            4
        ),
        "delta:",
        round(
            confidence_delta,
            4
        )
    )

    print(
        "Entropy:",
        round(
            a0_features[
                "entropy"
            ],
            4
        ),
        "->",
        round(
            a1_features[
                "entropy"
            ],
            4
        ),
        "delta:",
        round(
            entropy_delta,
            4
        )
    )

    # ========================================================
    # V10 ACCEPTANCE
    # ========================================================
    #
    # Evidence consensus is the PRIMARY factual gate.
    #
    # SUPPORTED majority:
    #       ACCEPT
    #
    # CONTRADICTED majority:
    #       ROLLBACK
    #
    # TIE / insufficient:
    #       ROLLBACK
    #
    # Confidence and entropy are recorded but do NOT
    # veto strong multi-source factual consensus.
    # ========================================================

    if (
        a1_consensus[
            "consensus"
        ]
        == "SUPPORTED"
    ):

        decision = {
            "decision": "ACCEPT",
            "reason": (
                "MULTI_SOURCE_SUPPORTING_MAJORITY"
            )
        }

    elif (
        a1_consensus[
            "consensus"
        ]
        == "CONTRADICTED"
    ):

        decision = {
            "decision": "ROLLBACK",
            "reason": (
                "MULTI_SOURCE_CONTRADICTING_MAJORITY"
            )
        }

    else:

        decision = {
            "decision": "ROLLBACK",
            "reason": (
                "MULTI_SOURCE_CONSENSUS_UNCERTAIN"
            )
        }

    print(
        "\nV10 FINAL DECISION:",
        decision
    )

    # ========================================================
    # RETURN COMPLETE RESULT
    # ========================================================

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

            "evidence_calibrated_trust": (
                a0_evidence_trust
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

            "evidence_calibrated_trust": (
                a1_evidence_trust
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

            "confidence_delta": (
                confidence_delta
            ),

            "entropy_delta": (
                entropy_delta
            ),

            "raw_trust_changed": (
                a0_raw_trust
                !=
                a1_raw_trust
            ),

            "evidence_trust_changed": (
                a0_evidence_trust
                !=
                a1_evidence_trust
            ),

            "evidence_trust_improved": (
                {
                    "UNRELIABLE": 0,
                    "UNCERTAIN": 1,
                    "RELIABLE": 2
                }[
                    a1_evidence_trust
                ]
                >
                {
                    "UNRELIABLE": 0,
                    "UNCERTAIN": 1,
                    "RELIABLE": 2
                }[
                    a0_evidence_trust
                ]
            )
        },

        "decision": decision
    }


# ============================================================
# CHECKPOINT DATA
# ============================================================

def build_checkpoint(
    results,
    next_index,
    accepted,
    rollback,
    errors
):

    return {

        "experiment": (
            "TrustGate V10 Multi-Source "
            "Consensus Checkpointed"
        ),

        "model": MODEL_NAME,

        "input": INPUT_PATH,

        "total_cases": len(
            correction_cases
        ),

        "completed_cases": len(
            results
        ),

        "next_index": next_index,

        "counts": {

            "accepted": accepted,

            "rollback": rollback,

            "errors": errors
        },

        "results": results
    }


# ============================================================
# RESUME CHECKPOINT
# ============================================================

results = []

start_index = 0

accepted = 0
rollback = 0
errors = 0


if os.path.exists(
    CHECKPOINT_PATH
):

    try:

        with open(
            CHECKPOINT_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            checkpoint = json.load(
                f
            )

        previous_results = checkpoint.get(
            "results",
            []
        )

        if (
            checkpoint.get(
                "total_cases"
            )
            == len(correction_cases)
        ):

            results = previous_results

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

            print(
                "\nCHECKPOINT FOUND"
            )

            print(
                "Completed cases:",
                len(results)
            )

            print(
                "Resuming from case:",
                start_index + 1
            )

        else:

            print(
                "\nCheckpoint does not match "
                "current experiment. Starting fresh."
            )

    except Exception as e:

        print(
            "\nCheckpoint could not be loaded:"
        )

        print(
            repr(e)
        )

        print(
            "Starting fresh."
        )


# ============================================================
# MAIN CHECKPOINTED LOOP
# ============================================================

for i in range(
    start_index,
    len(correction_cases)
):

    item = correction_cases[i]

    try:

        result = process_case(
            item,
            i + 1,
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

    except Exception as e:

        errors += 1

        print(
            "\nCASE ERROR:",
            repr(e)
        )

        result = {
            "heldout_id": item.get(
                "heldout_id"
            ),

            "question": item.get(
                "question"
            ),

            "error": repr(e)
        }

        results.append(
            result
        )

    # ========================================================
    # SAVE AFTER EVERY SINGLE CASE
    # ========================================================

    checkpoint = build_checkpoint(
        results,
        i + 1,
        accepted,
        rollback,
        errors
    )

    atomic_json_save(
        checkpoint,
        CHECKPOINT_PATH
    )

    print(
        "\nCHECKPOINT SAVED:"
    )

    print(
        f"{i + 1}/{len(correction_cases)}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

final_output = {

    "experiment": (
        "TrustGate V10 Multi-Source "
        "Consensus Self-Improvement"
    ),

    "model": MODEL_NAME,

    "input": INPUT_PATH,

    "total_cases": len(
        correction_cases
    ),

    "completed_cases": len(
        results
    ),

    "counts": {

        "accepted": accepted,

        "rollback": rollback,

        "errors": errors,

        "acceptance_rate_percent": round(
            100.0
            * accepted
            / max(
                len(results),
                1
            ),
            2
        )
    },

    "consensus_rule": {

        "support_majority": (
            "At least 2 supporting sources "
            "and more supporting than "
            "contradicting sources."
        ),

        "contradiction_majority": (
            "At least 2 contradicting sources "
            "and more contradicting than "
            "supporting sources."
        ),

        "otherwise": (
            "UNCERTAIN"
        )
    },

    "results": results
}


# ============================================================
# ATOMIC FINAL SAVE
# ============================================================

atomic_json_save(
    final_output,
    OUTPUT_PATH
)


print("\n")
print("=" * 72)
print("TRUSTGATE V10 COMPLETE")
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
    "Acceptance rate:",
    round(
        100.0
        * accepted
        / max(
            len(results),
            1
        ),
        2
    ),
    "%"
)

print(
    "\nFinal results:"
)

print(
    os.path.abspath(
        OUTPUT_PATH
    )
)

print(
    "\nCheckpoint:"
)

print(
    os.path.abspath(
        CHECKPOINT_PATH
    )
)