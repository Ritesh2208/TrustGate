import re
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# TRUSTGATE V16
# ============================================================
#
# Question
#    ↓
# A0 generation
#    ↓
# 896-D Trust Representation
#    ↓
# TrustGate
#    ↓
# External Evidence
#    ↓
# Need improvement?
#    ↓
# Generate A1
#    ↓
# Candidate validation
#    ↓
# Genuine-change check
#    ↓
# A1 TrustGate
#    ↓
# A1 evidence retrieval
#    ↓
# Source-level evidence checking
#    ↓
# Multi-source consensus
#    ↓
# Deterministic evidence gate
#    ↓
# ACCEPT / ROLLBACK
#
# ============================================================


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 80

MAX_EVIDENCE_SOURCES = 5

SUPPORT_THRESHOLD = 0.45


# ============================================================
# MODEL CACHE
# ============================================================

_TOKENIZER = None
_MODEL = None


# ============================================================
# LOAD MODEL
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
                "Never output instructions or prompt text "
                "instead of an answer. "
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
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower()

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
# CLEAN ANSWER
# ============================================================

def clean_generated_answer(answer):

    answer = str(answer).strip()

    answer = re.sub(
        r"^(Answer|Final answer|Correct answer)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )

    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip()
    ]

    if lines:

        answer = lines[0]

    return answer.strip()


# ============================================================
# INITIAL ANSWER
# ============================================================

def generate_initial_answer(question):

    prompt = f"""
Answer this question directly.

QUESTION:
{question}

Rules:

- Return the actual factual answer.
- Keep it concise.
- Do not explain the answering process.
- Do not output instructions.
- Do not output prompt text.
- Return ONLY the answer.

ANSWER:
"""

    return clean_generated_answer(
        generate_text(prompt)
    )


# ============================================================
# INVALID CANDIDATE PHRASES
# ============================================================

INVALID_CANDIDATE_PHRASES = [

    "the current answer is considered unreliable",

    "the previous answer is considered unreliable",

    "the current answer is unreliable",

    "the previous answer is unreliable",

    "answer the original question",

    "original question:",

    "retrieved evidence:",

    "previous answer:",

    "correct answer:",

    "return only the answer",

    "do not mention",

    "rules:",

    "important:",

    "the candidate answer",

]


# ============================================================
# CANDIDATE VALIDATION
# ============================================================

def validate_candidate_answer(
    question,
    candidate
):

    candidate = str(
        candidate
    ).strip()

    normalized = normalize_text(
        candidate
    )

    question_normalized = normalize_text(
        question
    )

    if not normalized:

        return {
            "valid": False,
            "reason": "EMPTY_CANDIDATE",
        }

    for phrase in INVALID_CANDIDATE_PHRASES:

        if phrase in normalized:

            return {
                "valid": False,
                "reason": "INSTRUCTION_LEAKAGE",
                "matched_phrase": phrase,
            }

    if normalized == question_normalized:

        return {
            "valid": False,
            "reason": "CANDIDATE_REPEATS_QUESTION",
        }

    return {
        "valid": True,
        "reason": "VALID_CANDIDATE",
    }


# ============================================================
# TRUST FEATURES
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
# TRUSTGATE
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

Extract the single central factual claim.

Rules:

- Preserve negation.
- Preserve names.
- Preserve numbers.
- Do not add facts.
- Return one short factual sentence.
- Do not explain.
- Return ONLY the claim.

CLAIM:
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
# VERIFY ANSWER
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
# EVIDENCE TEXT
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

        evidence = evidence[:3000]

        if evidence.strip():

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

    if not evidence_text:

        evidence_text = (
            "No reliable evidence was retrieved."
        )

    prompt = f"""
Answer the QUESTION below.

QUESTION:
{question}

CURRENT ANSWER:
{old_answer}

AVAILABLE EVIDENCE:
{evidence_text}

Your task is to produce a new factual answer to the
QUESTION.

Rules:

- Return an actual answer.
- Do not describe the current answer.
- Do not describe the correction process.
- Do not output instructions.
- Do not repeat prompt text.
- Do not write "The current answer is considered unreliable."
- Do not write "Correct answer:".
- Do not write "Previous answer:".
- Do not write "Retrieved evidence:".
- If the evidence is insufficient, give the most
  cautious factual answer possible.
- Return ONLY the answer.

ANSWER:
"""

    return clean_generated_answer(
        generate_text(prompt)
    )


# ============================================================
# SAME ANSWER / MEANINGFUL CHANGE
# ============================================================

def answers_effectively_same(
    a0,
    a1
):

    first = normalize_text(a0)
    second = normalize_text(a1)

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
# SOURCE JUDGE
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

    if not evidence.strip():

        return {
            "label": "INSUFFICIENT",
            "reason": "No evidence available.",
            "source_title": source_title,
            "evidence": "",
        }

    prompt = f"""
Evaluate the source below.

QUESTION:
{question}

CLAIM:
{claim}

SOURCE:
{source_title}

EVIDENCE:
{evidence}

Return exactly:

LABEL: SUPPORTS
REASON: ...

OR

LABEL: CONTRADICTS
REASON: ...

OR

LABEL: IRRELEVANT
REASON: ...

OR

LABEL: INSUFFICIENT
REASON: ...

Rules:

- SUPPORTS requires evidence that directly supports
  the claim.
- CONTRADICTS requires evidence that directly conflicts
  with the claim.
- IRRELEVANT means the source does not actually address
  the claim.
- INSUFFICIENT means the source is related but does not
  establish the claim.
- Never use keyword overlap alone.
- Never guess.
"""

    result = generate_text(
        prompt
    )

    label_match = re.search(
        r"LABEL\s*:\s*"
        r"(SUPPORTS|CONTRADICTS|IRRELEVANT|INSUFFICIENT)",
        result,
        flags=re.IGNORECASE
    )

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        result,
        flags=re.IGNORECASE
    )

    label = (
        label_match.group(1).upper()
        if label_match
        else "INSUFFICIENT"
    )

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else result.strip()
    )

    # ========================================================
    # V16 SAFETY CHECK
    # ========================================================
    #
    # If the model says SUPPORTS but its own reason contains
    # contradiction language, do NOT trust SUPPORTS.
    #
    # ========================================================

    contradiction_phrases = [
        "contradict",
        "conflict",
        "does not support",
        "not support",
        "opposes",
        "disagrees",
        "false",
        "incorrect",
    ]

    if label == "SUPPORTS":

        reason_lower = reason.lower()

        contradiction_detected = any(
            phrase in reason_lower
            for phrase in contradiction_phrases
        )

        if contradiction_detected:

            label = "CONTRADICTS"

            reason = (
                "Safety override: source judge labeled "
                "SUPPORTS but its reasoning indicates "
                "contradiction or lack of support. "
                + reason
            )

    return {
        "label": label,
        "reason": reason,
        "source_title": source_title,
        "evidence": evidence,
    }


# ============================================================
# CONSENSUS
# ============================================================

def calculate_source_consensus(
    source_judgments
):

    supports = [
        x
        for x in source_judgments
        if x["label"] == "SUPPORTS"
    ]

    contradicts = [
        x
        for x in source_judgments
        if x["label"] == "CONTRADICTS"
    ]

    support_count = len(supports)

    contradiction_count = len(
        contradicts
    )

    useful_count = (
        support_count
        +
        contradiction_count
    )

    # No useful evidence
    if useful_count == 0:

        return {
            "status": "UNCERTAIN",
            "strength": "NO_USEFUL_EVIDENCE",
            "support_count": 0,
            "contradiction_count": 0,
            "useful_source_count": 0,
        }

    # Only contradictions
    if (
        contradiction_count > 0
        and
        support_count == 0
    ):

        return {
            "status": "CONTRADICTED",
            "strength": "CONTRADICTION",
            "support_count": 0,
            "contradiction_count": contradiction_count,
            "useful_source_count": useful_count,
        }

    # Only support
    if (
        support_count > 0
        and
        contradiction_count == 0
    ):

        strength = (
            "STRONG"
            if support_count >= 2
            else
            "SINGLE_SOURCE"
        )

        return {
            "status": "SUPPORTED",
            "strength": strength,
            "support_count": support_count,
            "contradiction_count": 0,
            "useful_source_count": useful_count,
        }

    # Support majority
    if support_count > contradiction_count:

        return {
            "status": "SUPPORTED",
            "strength": "MAJORITY",
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "useful_source_count": useful_count,
        }

    # Contradiction majority
    if contradiction_count > support_count:

        return {
            "status": "CONTRADICTED",
            "strength": "MAJORITY_CONTRADICTION",
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "useful_source_count": useful_count,
        }

    # Equal conflict
    return {
        "status": "UNCERTAIN",
        "strength": "CONFLICTING",
        "support_count": support_count,
        "contradiction_count": contradiction_count,
        "useful_source_count": useful_count,
    }


# ============================================================
# EVIDENCE ANALYSIS
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
# DETERMINISTIC CANDIDATE GATE
# ============================================================

def candidate_support_result(
    verification,
    consensus,
    candidate_validation,
    meaningful_change
):

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

    support_count = consensus.get(
        "support_count",
        0
    )

    contradiction_count = consensus.get(
        "contradiction_count",
        0
    )

    consensus_status = consensus.get(
        "status",
        "UNCERTAIN"
    )

    # --------------------------------------------------------
    # INVALID CANDIDATE
    # --------------------------------------------------------

    if not candidate_validation.get(
        "valid",
        False
    ):

        return {
            "supported": False,
            "status": "INVALID_CANDIDATE",
            "reason": (
                candidate_validation.get(
                    "reason",
                    "INVALID_CANDIDATE"
                )
            ),
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": support_count,
            "contradicting_sources": contradiction_count,
        }

    # --------------------------------------------------------
    # NO MEANINGFUL CHANGE
    # --------------------------------------------------------

    if not meaningful_change:

        return {
            "supported": False,
            "status": "NO_IMPROVEMENT",
            "reason": "NO_MEANINGFUL_SELF_IMPROVEMENT",
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": support_count,
            "contradicting_sources": contradiction_count,
        }

    # --------------------------------------------------------
    # NO USEFUL EVIDENCE
    # --------------------------------------------------------

    if (
        support_count == 0
        and
        contradiction_count == 0
    ):

        return {
            "supported": False,
            "status": "UNCERTAIN",
            "reason": "NO_USEFUL_SOURCE_EVIDENCE",
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": 0,
            "contradicting_sources": 0,
        }

    # --------------------------------------------------------
    # STRONG CONTRADICTION
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
    # ZERO LEXICAL SUPPORT
    #
    # Prevents the exact V15 failure:
    #
    # support = 0
    # contradiction = 0
    # source judge says 3 SUPPORTS
    #
    # --------------------------------------------------------

    if support <= 0:

        return {
            "supported": False,
            "status": "UNCERTAIN",
            "reason": "ZERO_EVIDENCE_SUPPORT_SCORE",
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": support_count,
            "contradicting_sources": contradiction_count,
        }

    # --------------------------------------------------------
    # SUPPORT WITH NO CONTRADICTION
    # --------------------------------------------------------

    if (
        consensus_status == "SUPPORTED"
        and
        support > contradiction
        and
        support_count > 0
        and
        contradiction_count == 0
        and
        support >= SUPPORT_THRESHOLD
    ):

        return {
            "supported": True,
            "status": "SUPPORTED",
            "reason": (
                "EVIDENCE_AND_MULTI_SOURCE_SUPPORT"
            ),
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": support_count,
            "contradicting_sources": contradiction_count,
        }

    # --------------------------------------------------------
    # MAJORITY SUPPORT
    # --------------------------------------------------------

    if (
        consensus_status == "SUPPORTED"
        and
        support_count > contradiction_count
        and
        support > contradiction
        and
        support > 0
    ):

        return {
            "supported": True,
            "status": "SUPPORTED",
            "reason": (
                "MULTI_SOURCE_MAJORITY_SUPPORT"
            ),
            "support_score": support,
            "contradiction_score": contradiction,
            "supporting_sources": support_count,
            "contradicting_sources": contradiction_count,
        }

    # --------------------------------------------------------
    # OTHERWISE UNCERTAIN
    # --------------------------------------------------------

    return {
        "supported": False,
        "status": "UNCERTAIN",
        "reason": (
            "INSUFFICIENT_OR_CONFLICTING_EVIDENCE"
        ),
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

    priorities = {
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

    a0_priority = priorities.get(
        a0_label,
        -1
    )

    a1_priority = priorities.get(
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
# MAIN PIPELINE
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

    a0_evidence = analyze_evidence_consensus(
        question,
        a0_verified["claim"],
        a0_verification
    )

    a0_consensus = (
        a0_evidence["consensus"]
    )

    # ========================================================
    # IMPROVEMENT TRIGGER
    # ========================================================

    should_improve = (

        a0_consensus.get(
            "status"
        ) == "CONTRADICTED"

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
                "claim": a0_verified["claim"],
                "trustgate": a0_trust,
                "verification": a0_verification,
                "evidence_consensus": a0_evidence,
            },

            "candidate": None,

            "decision": {
                "decision": "KEEP_INITIAL",
                "reason": "INITIAL_ANSWER_ACCEPTED",
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
        a0_verification
    )

    candidate_validation = (
        validate_candidate_answer(
            question,
            a1
        )
    )

    retry_used = False

    # ========================================================
    # RETRY INVALID OR IDENTICAL
    # ========================================================

    if (
        not candidate_validation["valid"]
        or
        answers_effectively_same(
            a0,
            a1
        )
    ):

        retry_used = True

        a1 = generate_correction(
            question,
            a0,
            a0_verification,
            retry=True
        )

        candidate_validation = (
            validate_candidate_answer(
                question,
                a1
            )
        )

    # ========================================================
    # MEANINGFUL CHANGE
    # ========================================================

    meaningful_change = not answers_effectively_same(
        a0,
        a1
    )

    # ========================================================
    # HARD INVALID / NO-CHANGE STOP
    # ========================================================

    if (
        not candidate_validation["valid"]
        or
        not meaningful_change
    ):

        reason = (
            "INVALID_CANDIDATE"
            if not candidate_validation["valid"]
            else
            "NO_MEANINGFUL_SELF_IMPROVEMENT"
        )

        return {

            "initial": {
                "answer": a0,
                "claim": a0_verified["claim"],
                "trustgate": a0_trust,
                "verification": a0_verification,
                "evidence_consensus": a0_evidence,
            },

            "candidate": {

                "answer": a1,

                "validation": candidate_validation,

                "meaningful_change": meaningful_change,

                "retry_used": retry_used,
            },

            "decision": {

                "decision": "ROLLBACK",

                "reason": reason,
            },

            "final": {

                "answer": a0,

                "source": "INITIAL",
            },
        }

    # ========================================================
    # A1 TRUSTGATE
    # ========================================================

    a1_trust = classify_with_trustgate(
        question,
        a1,
        controller
    )

    # ========================================================
    # A1 VERIFY
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
    # A1 EVIDENCE
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
    # CANDIDATE GATE
    # ========================================================

    candidate_gate = candidate_support_result(
        a1_verification,
        a1_consensus,
        candidate_validation,
        meaningful_change
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

    if candidate_gate["supported"]:

        decision = {

            "decision": "ACCEPT",

            "reason": candidate_gate["reason"],

            "candidate_gate_status": (
                candidate_gate["status"]
            ),

            "support_score": (
                candidate_gate["support_score"]
            ),

            "contradiction_score": (
                candidate_gate["contradiction_score"]
            ),

            "supporting_sources": (
                candidate_gate["supporting_sources"]
            ),

            "contradicting_sources": (
                candidate_gate["contradicting_sources"]
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

            "reason": candidate_gate["reason"],

            "candidate_gate_status": (
                candidate_gate["status"]
            ),

            "support_score": (
                candidate_gate["support_score"]
            ),

            "contradiction_score": (
                candidate_gate["contradiction_score"]
            ),

            "supporting_sources": (
                candidate_gate["supporting_sources"]
            ),

            "contradicting_sources": (
                candidate_gate["contradicting_sources"]
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

            "verification": a0_verification,

            "evidence_consensus": a0_evidence,
        },

        "candidate": {

            "answer": a1,

            "claim": a1_verified[
                "claim"
            ],

            "validation": candidate_validation,

            "meaningful_change": meaningful_change,

            "trustgate": a1_trust,

            "verification": a1_verification,

            "evidence_consensus": a1_evidence,

            "candidate_gate": candidate_gate,

            "trust_comparison": trust_comparison,

            "retry_used": retry_used,
        },

        "decision": decision,

        "final": {

            "answer": final_answer,

            "source": final_source,
        },
    }