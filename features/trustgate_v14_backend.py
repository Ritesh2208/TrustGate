import re
import json
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# TRUSTGATE V14 UI BACKEND
# ============================================================
#
# Question
#    ↓
# Qwen generates A0
#    ↓
# 896-D Trust Representation
#    ↓
# TrustGate
#    ↓
# External Evidence Retrieval
#    ↓
# Evidence Source Judging
#    ↓
# If needed: generate A1
#    ↓
# Verify A1 with multiple sources
#    ↓
# Multi-source consensus
#    ↓
# ACCEPT / ROLLBACK
#
# IMPORTANT:
# - Raw TrustGate prediction is preserved.
# - Evidence consensus is a separate signal.
# - Irrelevant sources do NOT count as votes.
# - Strong contradiction blocks acceptance.
#
# ============================================================


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 80

# Minimum lexical support score retained as a secondary signal.
SUPPORT_THRESHOLD = 0.45

# Maximum number of evidence sources used by the judge.
MAX_EVIDENCE_SOURCES = 5


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

_TOKENIZER = None
_MODEL = None


# ============================================================
# LOAD QWEN
# ============================================================

def load_model():

    global _TOKENIZER
    global _MODEL

    if (
        _TOKENIZER is not None
        and _MODEL is not None
    ):
        return _TOKENIZER, _MODEL

    _TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    _MODEL.eval()

    return _TOKENIZER, _MODEL


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(instruction):

    tokenizer, model = load_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual answering assistant. "
                "Answer questions directly and concisely. "
                "Do not invent facts."
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

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# INITIAL ANSWER
# ============================================================

def generate_initial_answer(question):

    prompt = f"""
Answer the following question directly.

Question:

{question}

Rules:

- Give one concise factual answer.
- Do not provide unnecessary explanation.
- Do not mention these instructions.

Answer:
"""

    return generate_text(prompt)


# ============================================================
# TRUST FEATURE EXTRACTION
# ============================================================

def get_trust_features(
    question,
    answer
):

    tokenizer, model = load_model()

    system_text = (
        "You are a factual question answering assistant."
    )

    prompt = (
        system_text
        + "\n\n"
        + question
        + "\n\n"
    )

    full_text = (
        prompt
        + answer
    )

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
        encoded[
            "input_ids"
        ][
            0,
            prompt_len:
        ]
    )

    end_index = (
        first_prediction_index
        +
        len(generated_token_ids)
    )

    prediction_logits = logits[
        first_prediction_index:end_index
    ]

    usable = min(
        len(generated_token_ids),
        prediction_logits.shape[0]
    )

    generated_token_ids = (
        generated_token_ids[
            :usable
        ]
    )

    prediction_logits = (
        prediction_logits[
            :usable
        ]
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
        *
        torch.log(
            probabilities.clamp_min(
                1e-12
            )
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
        "hidden_state": (
            hidden_vector.tolist()
        ),
        "confidence": confidence,
        "entropy": entropy,
    }


# ============================================================
# TRUSTGATE CLASSIFICATION
# ============================================================

def classify_with_trustgate(
    question,
    answer,
    controller
):

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
# EXTRACT CENTRAL CLAIM
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

Extract the single central factual claim.

Rules:

- Preserve negation.
- Preserve names.
- Preserve numbers.
- Do not add facts.
- Do not explain.
- Return one short sentence.
- Return ONLY the claim.

Claim:
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
        line.strip()
        for line in claim.splitlines()
        if line.strip()
    ]

    if lines:
        claim = lines[0]

    return claim.strip()


# ============================================================
# VERIFY ANSWER WITH EXTERNAL VERIFIER
# ============================================================

def verify_answer(
    question,
    answer,
    verifier
):

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
# BUILD EVIDENCE TEXT
# ============================================================

def build_evidence_text(
    verification
):

    sources = verification.get(
        "sources",
        []
    )

    blocks = []

    for index, source in enumerate(
        sources[:MAX_EVIDENCE_SOURCES],
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
            str(sentence)
            for sentence in sentences
        )

        evidence = evidence[:2500]

        blocks.append(
            f"SOURCE {index}: {title}\n"
            f"{evidence}"
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# GENERATE CORRECTION
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

        prompt = f"""
The previous candidate answer was not acceptable.

Answer the ORIGINAL QUESTION again.

Original question:

{question}

Retrieved evidence:

{evidence_text}

Previous answer:

{old_answer}

Rules:

- Do not repeat the previous answer if it is wrong.
- Determine the answer from the evidence.
- Answer the original question directly.
- Do not mention the correction process.
- Do not mention sources.
- Keep the answer short.
- Return ONLY the answer.

Correct answer:
"""

    else:

        prompt = f"""
Answer the ORIGINAL QUESTION using the retrieved evidence.

Original question:

{question}

Retrieved evidence:

{evidence_text}

The current answer is considered unreliable.

Rules:

- Independently determine the answer.
- Do not blindly copy the current answer.
- Use relevant evidence.
- Answer the original question directly.
- Do not mention the correction process.
- Do not mention sources.
- Keep the answer short.
- Return ONLY the answer.

Correct answer:
"""

    return generate_text(
        prompt
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
# SAME ANSWER DETECTION
# ============================================================

def answers_effectively_same(
    a0,
    a1
):

    first = normalize_text(
        a0
    )

    second = normalize_text(
        a1
    )

    if first == second:
        return True

    first_tokens = set(
        first.split()
    )

    second_tokens = set(
        second.split()
    )

    if not first_tokens or not second_tokens:
        return False

    intersection = len(
        first_tokens
        &
        second_tokens
    )

    union = len(
        first_tokens
        |
        second_tokens
    )

    similarity = (
        intersection
        /
        max(union, 1)
    )

    return similarity >= 0.90


# ============================================================
# SOURCE-LEVEL EVIDENCE JUDGE
# ============================================================

def judge_single_source(
    question,
    claim,
    source_title,
    source_sentences
):

    evidence = " ".join(
        str(sentence)
        for sentence in source_sentences
    )

    evidence = evidence[:3500]

    prompt = f"""
You are an evidence evaluation system.

Question:
{question}

Central claim:
{claim}

Source title:
{source_title}

Source evidence:
{evidence}

Determine whether THIS SOURCE actually provides evidence
about the central claim.

Use semantic meaning, not keyword overlap.

Return exactly TWO lines:

LABEL: SUPPORTS
REASON: short explanation

OR:

LABEL: CONTRADICTS
REASON: short explanation

OR:

LABEL: IRRELEVANT
REASON: short explanation

OR:

LABEL: INSUFFICIENT
REASON: short explanation

Rules:

- SUPPORTS means the source directly supports the claim.
- CONTRADICTS means the source directly conflicts with the claim.
- IRRELEVANT means the source is about a different subject.
- INSUFFICIENT means the source is related but does not establish the claim.
- Do not guess.
- Do not use the fact that the source was retrieved as evidence.
"""

    result = generate_text(
        prompt
    )

    label_match = re.search(
        r"LABEL\s*:\s*(SUPPORTS|CONTRADICTS|IRRELEVANT|INSUFFICIENT)",
        result,
        flags=re.IGNORECASE
    )

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        result,
        flags=re.IGNORECASE
    )

    if label_match:

        label = (
            label_match.group(1)
            .upper()
        )

    else:

        label = "INSUFFICIENT"

    if reason_match:

        reason = (
            reason_match.group(1)
            .strip()
        )

    else:

        reason = result.strip()

    return {
        "label": label,
        "reason": reason,
        "source_title": source_title,
        "evidence": evidence,
    }


# ============================================================
# MULTI-SOURCE CONSENSUS
# ============================================================

def calculate_source_consensus(
    source_judgments
):

    supports = [
        item
        for item in source_judgments
        if item["label"] == "SUPPORTS"
    ]

    contradicts = [
        item
        for item in source_judgments
        if item["label"] == "CONTRADICTS"
    ]

    useful = [
        item
        for item in source_judgments
        if item["label"]
        in
        [
            "SUPPORTS",
            "CONTRADICTS"
        ]
    ]

    support_count = len(
        supports
    )

    contradiction_count = len(
        contradicts
    )

    useful_count = len(
        useful
    )

    # --------------------------------------------------------
    # Strong support
    # --------------------------------------------------------

    if (
        support_count >= 2
        and
        contradiction_count == 0
    ):

        return {
            "status": "SUPPORTED",
            "strength": "STRONG",
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "useful_source_count": useful_count,
        }

    # --------------------------------------------------------
    # Majority support
    #
    # Example:
    # 2 support + 1 contradiction
    # --------------------------------------------------------

    if (
        support_count > contradiction_count
        and
        support_count >= 2
    ):

        return {
            "status": "SUPPORTED",
            "strength": "MAJORITY",
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "useful_source_count": useful_count,
        }

    # --------------------------------------------------------
    # Single strong supporting source
    #
    # This is allowed only when there is no contradiction.
    # --------------------------------------------------------

    if (
        support_count == 1
        and
        contradiction_count == 0
    ):

        return {
            "status": "SUPPORTED",
            "strength": "SINGLE_SOURCE",
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "useful_source_count": useful_count,
        }

    # --------------------------------------------------------
    # Majority contradiction
    # --------------------------------------------------------

    if (
        contradiction_count > support_count
        and
        contradiction_count >= 1
    ):

        return {
            "status": "CONTRADICTED",
            "strength": "CONTRADICTION",
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "useful_source_count": useful_count,
        }

    # --------------------------------------------------------
    # Mixed evidence
    #
    # Example:
    # 1 support + 1 contradiction
    # --------------------------------------------------------

    if (
        support_count > 0
        and
        contradiction_count > 0
    ):

        return {
            "status": "UNCERTAIN",
            "strength": "CONFLICTING",
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "useful_source_count": useful_count,
        }

    # --------------------------------------------------------
    # No useful evidence
    # --------------------------------------------------------

    return {
        "status": "UNCERTAIN",
        "strength": "INSUFFICIENT",
        "support_count": support_count,
        "contradiction_count": contradiction_count,
        "useful_source_count": useful_count,
    }


# ============================================================
# MULTI-SOURCE EVIDENCE ANALYSIS
# ============================================================

def analyze_evidence_consensus(
    question,
    claim,
    verification
):

    sources = verification.get(
        "sources",
        []
    )

    source_judgments = []

    for source in sources[
        :MAX_EVIDENCE_SOURCES
    ]:

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

        judgment = judge_single_source(
            question,
            claim,
            title,
            sentences
        )

        source_judgments.append(
            judgment
        )

    consensus = calculate_source_consensus(
        source_judgments
    )

    return {
        "consensus": consensus,
        "source_judgments": source_judgments,
        "sources_examined": len(
            source_judgments
        ),
    }


# ============================================================
# EVIDENCE GATE
# ============================================================

def candidate_support_result(
    verification,
    consensus
):

    status = verification.get(
        "verification_status",
        ""
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

    consensus_status = consensus.get(
        "status",
        "UNCERTAIN"
    )

    support_count = consensus.get(
        "support_count",
        0
    )

    contradiction_count = consensus.get(
        "contradiction_count",
        0
    )

    # --------------------------------------------------------
    # HARD CONTRADICTION
    # --------------------------------------------------------

    if consensus_status == "CONTRADICTED":

        return {
            "supported": False,
            "status": "CONTRADICTED",
            "reason": "MULTI_SOURCE_CONTRADICTION",
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": support_count,
            "contradicting_sources": contradiction_count,
        }

    # --------------------------------------------------------
    # MULTI-SOURCE SUPPORT
    # --------------------------------------------------------

    if consensus_status == "SUPPORTED":

        return {
            "supported": True,
            "status": "SUPPORTED",
            "reason": (
                "MULTI_SOURCE_EVIDENCE_SUPPORT"
            ),
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": support_count,
            "contradicting_sources": contradiction_count,
        }

    # --------------------------------------------------------
    # FALLBACK:
    # Strong lexical evidence with no contradiction.
    #
    # This is intentionally secondary to source consensus.
    # --------------------------------------------------------

    if (
        support >= SUPPORT_THRESHOLD
        and
        support > contradiction
        and
        contradiction_count == 0
    ):

        return {
            "supported": True,
            "status": "SUPPORTED",
            "reason": (
                "LEXICAL_SUPPORT_WITHOUT_CONTRADICTION"
            ),
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": support_count,
            "contradicting_sources": contradiction_count,
        }

    # --------------------------------------------------------
    # VERIFIER EXPLICITLY SUPPORTED
    # --------------------------------------------------------

    if (
        status == "SUPPORTED"
        and
        support > contradiction
        and
        contradiction_count == 0
    ):

        return {
            "supported": True,
            "status": "SUPPORTED",
            "reason": "VERIFIER_SUPPORTED",
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": support_count,
            "contradicting_sources": contradiction_count,
        }

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    return {
        "supported": False,
        "status": "UNCERTAIN",
        "reason": "INSUFFICIENT_OR_CONFLICTING_EVIDENCE",
        "support_score": support,
        "contradiction_score": contradiction,
        "supporting_sources": support_count,
        "contradicting_sources": contradiction_count,
    }


# ============================================================
# TRUST COMPARISON
# ============================================================

def compare_trust(
    a0_trust,
    a1_trust
):

    trust_priority = {
        "RELIABLE": 2,
        "UNCERTAIN": 1,
        "UNRELIABLE": 0
    }

    a0_label = a0_trust.get(
        "label",
        "UNKNOWN"
    )

    a1_label = a1_trust.get(
        "label",
        "UNKNOWN"
    )

    a0_priority = trust_priority.get(
        a0_label,
        -1
    )

    a1_priority = trust_priority.get(
        a1_label,
        -1
    )

    return {
        "a0_label": a0_label,
        "a1_label": a1_label,
        "trust_improved": (
            a1_priority > a0_priority
        ),
        "trust_same": (
            a1_priority == a0_priority
        ),
        "trust_worsened": (
            a1_priority < a0_priority
        ),
        "confidence_improved": (
            a1_trust["confidence"]
            >
            a0_trust["confidence"]
        ),
        "entropy_improved": (
            a1_trust["entropy"]
            <
            a0_trust["entropy"]
        ),
    }


# ============================================================
# MAIN UI BACKEND
# ============================================================

def answer_question(
    question,
    controller,
    verifier
):

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    # ========================================================
    # A0
    # ========================================================

    a0 = generate_initial_answer(
        question
    )

    a0_trust = classify_with_trustgate(
        question,
        a0,
        controller
    )

    a0_verified = verify_answer(
        question,
        a0,
        verifier
    )

    a0_verification = (
        a0_verified["verification"]
    )

    # --------------------------------------------------------
    # Analyze A0 evidence
    # --------------------------------------------------------

    a0_evidence = analyze_evidence_consensus(
        question,
        a0_verified["claim"],
        a0_verification
    )

    a0_consensus = (
        a0_evidence["consensus"]
    )

    a0_status = a0_consensus.get(
        "status",
        "UNCERTAIN"
    )

    # ========================================================
    # SHOULD WE IMPROVE?
    # ========================================================

    should_improve = (

        a0_status == "CONTRADICTED"

        or

        a0_trust["label"]
        in
        [
            "UNRELIABLE",
            "UNCERTAIN"
        ]
    )

    # ========================================================
    # KEEP INITIAL
    # ========================================================

    if not should_improve:

        return {

            "initial": {

                "answer": a0,

                "claim": a0_verified[
                    "claim"
                ],

                "trustgate": a0_trust,

                "verification": (
                    a0_verification
                ),

                "evidence_consensus": (
                    a0_evidence
                ),

            },

            "candidate": None,

            "decision": {

                "decision": "KEEP_INITIAL",

                "reason": (
                    "INITIAL_ANSWER_ACCEPTED"
                ),

            },

            "final": {

                "answer": a0,

                "source": "INITIAL",

            },
        }

    # ========================================================
    # GENERATE A1
    # ========================================================

    a1 = generate_correction(
        question,
        a0,
        a0_verification,
        retry=False
    )

    retry_used = False

    # ========================================================
    # RETRY IF A1 COPIES A0
    # ========================================================

    if answers_effectively_same(
        a0,
        a1
    ):

        retry_used = True

        a1 = generate_correction(
            question,
            a0,
            a0_verification,
            retry=True
        )

    # ========================================================
    # A1 TRUSTGATE
    # ========================================================

    a1_trust = classify_with_trustgate(
        question,
        a1,
        controller
    )

    # ========================================================
    # A1 VERIFICATION
    # ========================================================

    a1_verified = verify_answer(
        question,
        a1,
        verifier
    )

    a1_verification = (
        a1_verified["verification"]
    )

    # ========================================================
    # A1 MULTI-SOURCE EVIDENCE
    # ========================================================

    a1_evidence = analyze_evidence_consensus(
        question,
        a1_verified["claim"],
        a1_verification
    )

    a1_consensus = (
        a1_evidence["consensus"]
    )

    # ========================================================
    # CANDIDATE EVIDENCE GATE
    # ========================================================

    candidate_gate = candidate_support_result(
        a1_verification,
        a1_consensus
    )

    # ========================================================
    # TRUST COMPARISON
    # ========================================================

    trust_comparison = compare_trust(
        a0_trust,
        a1_trust
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================
    #
    # Evidence is the primary correction gate.
    #
    # Raw TrustGate is preserved separately.
    #
    # A candidate can be accepted even if the raw TrustGate
    # remains UNRELIABLE, provided strong external evidence
    # supports the candidate and there is no meaningful
    # contradiction.
    #
    # ========================================================

    if candidate_gate["supported"]:

        decision = {

            "decision": "ACCEPT",

            "reason": (
                candidate_gate["reason"]
            ),

            "candidate_gate_status": (
                candidate_gate["status"]
            ),

            "support_score": (
                candidate_gate[
                    "support_score"
                ]
            ),

            "contradiction_score": (
                candidate_gate[
                    "contradiction_score"
                ]
            ),

            "supporting_sources": (
                candidate_gate[
                    "supporting_sources"
                ]
            ),

            "contradicting_sources": (
                candidate_gate[
                    "contradicting_sources"
                ]
            ),

            "trustgate_improved": (
                trust_comparison[
                    "trust_improved"
                ]
            ),

            "confidence_improved": (
                trust_comparison[
                    "confidence_improved"
                ]
            ),

            "entropy_improved": (
                trust_comparison[
                    "entropy_improved"
                ]
            ),

        }

        final_answer = a1

        final_source = "CANDIDATE"

    else:

        decision = {

            "decision": "ROLLBACK",

            "reason": (
                candidate_gate["reason"]
            ),

            "candidate_gate_status": (
                candidate_gate["status"]
            ),

            "support_score": (
                candidate_gate[
                    "support_score"
                ]
            ),

            "contradiction_score": (
                candidate_gate[
                    "contradiction_score"
                ]
            ),

            "supporting_sources": (
                candidate_gate[
                    "supporting_sources"
                ]
            ),

            "contradicting_sources": (
                candidate_gate[
                    "contradicting_sources"
                ]
            ),

            "trustgate_improved": (
                trust_comparison[
                    "trust_improved"
                ]
            ),

            "confidence_improved": (
                trust_comparison[
                    "confidence_improved"
                ]
            ),

            "entropy_improved": (
                trust_comparison[
                    "entropy_improved"
                ]
            ),

        }

        final_answer = a0

        final_source = "INITIAL"

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "initial": {

            "answer": a0,

            "claim": a0_verified[
                "claim"
            ],

            "trustgate": a0_trust,

            "verification": (
                a0_verification
            ),

            "evidence_consensus": (
                a0_evidence
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

            "evidence_consensus": (
                a1_evidence
            ),

            "candidate_gate": (
                candidate_gate
            ),

            "trust_comparison": (
                trust_comparison
            ),

            "retry_used": retry_used,

        },

        "decision": decision,

        "final": {

            "answer": final_answer,

            "source": final_source,

        },

    }