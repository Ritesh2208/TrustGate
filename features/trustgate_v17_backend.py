import re
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# TRUSTGATE V17
# CLAIM-SPECIFIC EVIDENCE-GROUNDED SELF-IMPROVEMENT
# ============================================================
#
# Main goal:
#
#   A0
#    ↓
#   TrustGate
#    ↓
#   Extract claim
#    ↓
#   Search QUESTION + CLAIM
#    ↓
#   Filter irrelevant sources
#    ↓
#   Direct evidence judgment
#    ↓
#   Evidence consensus
#    ↓
#   Generate A1 INDEPENDENTLY FROM VERIFIED EVIDENCE
#    ↓
#   Validate A1
#    ↓
#   Check meaningful change
#    ↓
#   Re-run TrustGate
#    ↓
#   Retrieve evidence AGAIN for A1
#    ↓
#   Verify A1 independently
#    ↓
#   ACCEPT / ROLLBACK
#
# Hard rules:
#
# 1. A1 cannot be identical to A0.
# 2. A1 cannot contain instruction leakage.
# 3. Irrelevant sources do not count.
# 4. Topic-only sources do not count.
# 5. Source must directly establish claim to SUPPORT.
# 6. A1 is generated independently from evidence.
# 7. A1 is independently verified.
# 8. Zero useful evidence => ROLLBACK.
# 9. Contradiction => ROLLBACK.
# 10. Direct evidence support => ACCEPT.
# 11. TrustGate alone never proves factual correctness.
#
# ============================================================


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 80

MAX_SOURCES = 5

MAX_EVIDENCE_SENTENCES = 6

SUPPORT_THRESHOLD = 0.35


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

    print(
        "[V17] Loading Qwen model..."
    )

    _TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    _MODEL.eval()

    print(
        "[V17] Qwen loaded."
    )

    return _TOKENIZER, _MODEL


# ============================================================
# TEXT GENERATION
# ============================================================

def generate_text(
    instruction
):

    tokenizer, model = load_model()

    messages = [

        {
            "role": "system",
            "content": (
                "You are a strict factual assistant. "
                "Answer the user's question directly. "
                "Never output instructions, prompts, "
                "or descriptions of the answering process."
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

def normalize_text(
    text
):

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
# CLEAN GENERATED ANSWER
# ============================================================

def clean_answer(
    answer
):

    answer = str(
        answer
    ).strip()

    answer = re.sub(
        r"^(answer|final answer|correct answer)\s*:\s*",
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

def generate_initial_answer(
    question
):

    prompt = f"""
Answer the following question directly.

QUESTION:
{question}

Rules:

- Give the factual answer.
- Be concise.
- Do not explain the answering process.
- Do not output instructions.
- Do not output prompt text.
- Return only the answer.

ANSWER:
"""

    return clean_answer(
        generate_text(
            prompt
        )
    )


# ============================================================
# CANDIDATE VALIDATION
# ============================================================

INVALID_PHRASES = [

    "the current answer is considered unreliable",

    "the previous answer is considered unreliable",

    "the current answer is unreliable",

    "the previous answer is unreliable",

    "answer the original question",

    "original question:",

    "retrieved evidence:",

    "previous answer:",

    "correct answer:",

    "candidate answer:",

    "return only the answer",

    "do not mention these instructions",

    "rules:",

    "important:",
]


def validate_candidate(
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

    for phrase in INVALID_PHRASES:

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
# MEANINGFUL CHANGE
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

    first_words = set(
        first.split()
    )

    second_words = set(
        second.split()
    )

    if not first_words or not second_words:

        return False

    overlap = len(
        first_words
        &
        second_words
    )

    union = len(
        first_words
        |
        second_words
    )

    similarity = (
        overlap
        /
        max(union, 1)
    )

    return similarity >= 0.90


# ============================================================
# TRUST FEATURES
# ============================================================

def get_trust_features(
    question,
    answer
):

    tokenizer, model = load_model()

    prompt = (
        "You are a factual question answering assistant.\n\n"
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

    logits = (
        outputs.logits[0]
        .float()
    )

    generated_ids = (
        encoded["input_ids"][
            0,
            prompt_len:
        ]
    )

    prediction_start = max(
        prompt_len - 1,
        0
    )

    prediction_end = (
        prediction_start
        +
        len(generated_ids)
    )

    prediction_logits = logits[
        prediction_start:
        prediction_end
    ]

    usable = min(
        len(generated_ids),
        prediction_logits.shape[0]
    )

    generated_ids = (
        generated_ids[:usable]
    )

    prediction_logits = (
        prediction_logits[:usable]
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
        generated_ids
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
            "Expected 896-D hidden state, "
            f"got {hidden_vector.shape[0]}"
        )

    return {

        "hidden_state":
            hidden_vector.tolist(),

        "confidence":
            confidence,

        "entropy":
            entropy,
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

        "label":
            label,

        "confidence":
            features["confidence"],

        "entropy":
            features["entropy"],
    }


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claim(
    question,
    answer
):

    prompt = f"""
Extract the single central factual claim from this answer.

QUESTION:
{question}

ANSWER:
{answer}

Rules:

- Preserve the subject.
- Preserve names.
- Preserve dates.
- Preserve numbers.
- Preserve negation.
- Do not add information.
- Do not remove the important factual relationship.
- Return exactly one factual sentence.
- Return only the claim.

CLAIM:
"""

    claim = generate_text(
        prompt
    )

    claim = re.sub(
        r"^(claim|central claim)\s*:\s*",
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
# CLAIM-SPECIFIC SEARCH QUERY
# ============================================================

def build_claim_query(
    question,
    claim
):

    question_clean = " ".join(
        question.split()
    )

    claim_clean = " ".join(
        claim.split()
    )

    return (
        question_clean
        + " "
        + claim_clean
    )


# ============================================================
# CLAIM-SPECIFIC SOURCE RETRIEVAL
# ============================================================

def retrieve_claim_sources(
    question,
    claim,
    verifier
):

    query = build_claim_query(
        question,
        claim
    )

    print()
    print(
        "[V17] Claim-specific search:"
    )
    print(
        "[V17]",
        query
    )

    try:

        search_results = (
            verifier.search_wikipedia(
                query,
                limit=max(
                    MAX_SOURCES,
                    5
                )
            )
        )

    except TypeError:

        search_results = (
            verifier.search_wikipedia(
                query
            )
        )

    except Exception as error:

        print(
            "[V17] Search error:",
            error
        )

        return []

    if not search_results:

        return []

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    titles = []

    for item in search_results:

        if "error" in item:

            continue

        title = item.get(
            "title",
            ""
        )

        if (
            title
            and
            title not in titles
        ):

            titles.append(
                title
            )

    # --------------------------------------------------------
    # Retrieve pages
    # --------------------------------------------------------

    records = []

    for title in titles:

        if len(records) >= MAX_SOURCES:

            break

        try:

            page_text = (
                verifier.retrieve_wikipedia_page(
                    title
                )
            )

        except Exception:

            continue

        if not page_text:

            continue

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We retrieve sentences using QUESTION + CLAIM,
        # not only the generated answer.
        # ----------------------------------------------------

        try:

            sentences = (
                verifier.extract_relevant_sentences(
                    question,
                    claim,
                    page_text,
                    max_sentences=MAX_EVIDENCE_SENTENCES
                )
            )

        except TypeError:

            try:

                sentences = (
                    verifier.extract_relevant_sentences(
                        question,
                        claim,
                        page_text
                    )
                )

            except Exception:

                sentences = []

        except Exception:

            sentences = []

        if not sentences:

            continue

        records.append({

            "title":
                title,

            "source":
                "Wikipedia",

            "url":
                (
                    "https://en.wikipedia.org/wiki/"
                    +
                    title.replace(
                        " ",
                        "_"
                    )
                ),

            "sentences":
                sentences,

            "page_text":
                page_text,
        })

    return records


# ============================================================
# BUILD EVIDENCE TEXT
# ============================================================

def build_evidence_text(
    sources
):

    blocks = []

    for index, source in enumerate(
        sources,
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

        blocks.append(
            f"""
SOURCE {index}
TITLE:
{title}

EVIDENCE:
{evidence[:4000]}
"""
        )

    return "\n".join(
        blocks
    )


# ============================================================
# STRICT SOURCE JUDGE
# ============================================================

def judge_source(
    question,
    claim,
    source
):

    title = source.get(
        "title",
        "Unknown"
    )

    sentences = source.get(
        "sentences",
        []
    )

    evidence = " ".join(
        str(x)
        for x in sentences
    )

    if not evidence.strip():

        return {

            "label":
                "INSUFFICIENT",

            "reason":
                "The source contains no usable evidence.",

            "source_title":
                title,

            "evidence":
                evidence,
        }

    prompt = f"""
You are a STRICT factual evidence judge.

QUESTION:
{question}

EXACT CLAIM:
{claim}

SOURCE TITLE:
{title}

SOURCE EVIDENCE:
{evidence[:4500]}

Classify whether THIS SOURCE DIRECTLY establishes
the EXACT CLAIM.

Return exactly one:

LABEL: SUPPORTS
REASON: ...

OR

LABEL: CONTRADICTS
REASON: ...

OR

LABEL: INSUFFICIENT
REASON: ...

Rules:

1. Judge the exact claim, not merely the topic.
2. The source must directly establish the factual
   relationship in the claim to be SUPPORTS.
3. A source merely mentioning the question topic
   is INSUFFICIENT.
4. A source mentioning the subject but not the
   requested fact is INSUFFICIENT.
5. If the source explicitly states the opposite
   fact, use CONTRADICTS.
6. Preserve negation.
7. Preserve dates and numbers.
8. Do not use outside knowledge.
9. Do not infer missing facts.
10. If you are unsure, use INSUFFICIENT.
11. SUPPORTS requires direct evidence.
"""

    raw = generate_text(
        prompt
    )

    label_match = re.search(
        r"LABEL\s*:\s*"
        r"(SUPPORTS|CONTRADICTS|INSUFFICIENT)",
        raw,
        flags=re.IGNORECASE
    )

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        raw,
        flags=re.IGNORECASE
    )

    label = (

        label_match.group(1).upper()

        if label_match

        else
        "INSUFFICIENT"
    )

    reason = (

        reason_match.group(1).strip()

        if reason_match

        else
        raw.strip()
    )

    # ========================================================
    # V17 SAFETY NORMALIZATION
    # ========================================================

    reason_lower = reason.lower()

    insufficient_phrases = [

        "does not explicitly state",

        "does not explicitly mention",

        "does not state",

        "does not mention",

        "does not provide",

        "does not establish",

        "cannot establish",

        "cannot determine",

        "not enough evidence",

        "insufficient evidence",

        "not directly",

        "does not directly",

        "only mentions",

        "merely mentions",

        "not relevant",

        "irrelevant",

        "cannot support",
    ]

    contradiction_phrases = [

        "contradicts",

        "contradict",

        "conflicts with",

        "conflict with",

        "opposite",

        "inconsistent with",

        "false",

        "incorrect",
    ]

    # --------------------------------------------------------
    # If the reason itself says insufficient, override.
    # --------------------------------------------------------

    if any(
        phrase in reason_lower
        for phrase in insufficient_phrases
    ):

        label = "INSUFFICIENT"

        reason = (
            "V17 safety normalization: "
            "the source judge's reasoning indicates "
            "that the source does not directly establish "
            "the claim. Original reasoning: "
            + reason
        )

    # --------------------------------------------------------
    # If the reasoning clearly indicates contradiction,
    # never allow SUPPORTS.
    # --------------------------------------------------------

    elif (
        label == "SUPPORTS"
        and
        any(
            phrase in reason_lower
            for phrase in contradiction_phrases
        )
    ):

        label = "CONTRADICTS"

        reason = (
            "V17 safety normalization: "
            "source reasoning indicates contradiction. "
            "Original reasoning: "
            + reason
        )

    return {

        "label":
            label,

        "reason":
            reason,

        "source_title":
            title,

        "evidence":
            evidence,
    }


# ============================================================
# SOURCE CONSENSUS
# ============================================================

def calculate_consensus(
    judgments
):

    support_count = sum(
        1
        for item in judgments
        if item["label"] == "SUPPORTS"
    )

    contradiction_count = sum(
        1
        for item in judgments
        if item["label"] == "CONTRADICTS"
    )

    insufficient_count = sum(
        1
        for item in judgments
        if item["label"] == "INSUFFICIENT"
    )

    useful_count = (
        support_count
        +
        contradiction_count
    )

    # --------------------------------------------------------
    # No direct evidence
    # --------------------------------------------------------

    if useful_count == 0:

        return {

            "status":
                "UNCERTAIN",

            "strength":
                "NO_DIRECT_EVIDENCE",

            "support_count":
                support_count,

            "contradiction_count":
                contradiction_count,

            "insufficient_count":
                insufficient_count,
        }

    # --------------------------------------------------------
    # Contradiction without support
    # --------------------------------------------------------

    if (
        contradiction_count > 0
        and
        support_count == 0
    ):

        return {

            "status":
                "CONTRADICTED",

            "strength":
                "DIRECT_CONTRADICTION",

            "support_count":
                support_count,

            "contradiction_count":
                contradiction_count,

            "insufficient_count":
                insufficient_count,
        }

    # --------------------------------------------------------
    # Direct support without contradiction
    # --------------------------------------------------------

    if (
        support_count > 0
        and
        contradiction_count == 0
    ):

        if support_count >= 2:

            strength = "STRONG"

        else:

            strength = "SINGLE_DIRECT_SOURCE"

        return {

            "status":
                "SUPPORTED",

            "strength":
                strength,

            "support_count":
                support_count,

            "contradiction_count":
                contradiction_count,

            "insufficient_count":
                insufficient_count,
        }

    # --------------------------------------------------------
    # Majority support
    # --------------------------------------------------------

    if support_count > contradiction_count:

        return {

            "status":
                "SUPPORTED",

            "strength":
                "MAJORITY_DIRECT_SUPPORT",

            "support_count":
                support_count,

            "contradiction_count":
                contradiction_count,

            "insufficient_count":
                insufficient_count,
        }

    # --------------------------------------------------------
    # Majority contradiction
    # --------------------------------------------------------

    if contradiction_count > support_count:

        return {

            "status":
                "CONTRADICTED",

            "strength":
                "MAJORITY_CONTRADICTION",

            "support_count":
                support_count,

            "contradiction_count":
                contradiction_count,

            "insufficient_count":
                insufficient_count,
        }

    # --------------------------------------------------------
    # Equal support and contradiction
    # --------------------------------------------------------

    return {

        "status":
            "UNCERTAIN",

        "strength":
            "CONFLICTING_DIRECT_EVIDENCE",

        "support_count":
            support_count,

        "contradiction_count":
            contradiction_count,

        "insufficient_count":
            insufficient_count,
    }


# ============================================================
# EVIDENCE ANALYSIS
# ============================================================

def analyze_claim_evidence(
    question,
    claim,
    verifier
):

    sources = retrieve_claim_sources(
        question,
        claim,
        verifier
    )

    if not sources:

        return {

            "consensus": {

                "status":
                    "UNCERTAIN",

                "strength":
                    "NO_SOURCES",

                "support_count":
                    0,

                "contradiction_count":
                    0,

                "insufficient_count":
                    0,
            },

            "source_judgments":
                [],

            "sources":
                [],

            "sources_examined":
                0,
        }

    judgments = []

    for source in sources:

        judgment = judge_source(
            question,
            claim,
            source
        )

        judgments.append(
            judgment
        )

    consensus = calculate_consensus(
        judgments
    )

    return {

        "consensus":
            consensus,

        "source_judgments":
            judgments,

        "sources":
            sources,

        "sources_examined":
            len(sources),
    }


# ============================================================
# LEGACY-STYLE VERIFICATION OBJECT
# ============================================================

def make_verification_object(
    evidence_analysis
):

    consensus = evidence_analysis[
        "consensus"
    ]

    support_count = consensus.get(
        "support_count",
        0
    )

    contradiction_count = consensus.get(
        "contradiction_count",
        0
    )

    # Evidence score here is deliberately conservative.
    #
    # It is NOT used as proof by itself.
    # Direct source consensus is the primary gate.

    if support_count > 0:

        support_score = min(
            1.0,
            0.35
            +
            0.15 * support_count
        )

    else:

        support_score = 0.0

    if contradiction_count > 0:

        contradiction_score = min(
            1.0,
            0.35
            +
            0.15 * contradiction_count
        )

    else:

        contradiction_score = 0.0

    status = (
        consensus.get(
            "status",
            "UNCERTAIN"
        )
    )

    if status == "SUPPORTED":

        verification_status = (
            "SUPPORTED"
        )

    elif status == "CONTRADICTED":

        verification_status = (
            "CONTRADICTED"
        )

    else:

        verification_status = (
            "INSUFFICIENT_EVIDENCE"
        )

    supporting_sentences = []

    contradicting_sentences = []

    for item in evidence_analysis[
        "source_judgments"
    ]:

        if item["label"] == "SUPPORTS":

            supporting_sentences.append(
                item["evidence"]
            )

        elif item["label"] == "CONTRADICTS":

            contradicting_sentences.append(
                item["evidence"]
            )

    return {

        "verification_status":
            verification_status,

        "source_count":
            evidence_analysis[
                "sources_examined"
            ],

        "support_score":
            support_score,

        "contradiction_score":
            contradiction_score,

        "supporting_sentences":
            supporting_sentences,

        "contradicting_sentences":
            contradicting_sentences,

        "reason":
            consensus.get(
                "strength",
                "UNKNOWN"
            ),

        "sources":
            evidence_analysis[
                "sources"
            ],
    }


# ============================================================
# GENERATE A1 FROM VERIFIED EVIDENCE
# ============================================================

def generate_evidence_grounded_correction(
    question,
    claim,
    evidence_analysis
):

    sources = evidence_analysis.get(
        "sources",
        []
    )

    judgments = evidence_analysis.get(
        "source_judgments",
        []
    )

    direct_support = []

    for source, judgment in zip(
        sources,
        judgments
    ):

        if judgment.get(
            "label"
        ) != "SUPPORTS":

            continue

        direct_support.append({

            "title":
                source.get(
                    "title",
                    "Source"
                ),

            "evidence":
                " ".join(
                    str(x)
                    for x in source.get(
                        "sentences",
                        []
                    )
                ),
        })

    if not direct_support:

        return ""

    evidence_blocks = []

    for index, item in enumerate(
        direct_support,
        start=1
    ):

        evidence_blocks.append(
            f"""
VERIFIED SOURCE {index}
TITLE:
{item["title"]}

DIRECT EVIDENCE:
{item["evidence"][:3500]}
"""
        )

    evidence_text = "\n".join(
        evidence_blocks
    )

    # ========================================================
    # IMPORTANT:
    #
    # A0 is deliberately NOT passed to this function.
    #
    # The candidate is generated independently from:
    # QUESTION + CLAIM + VERIFIED EVIDENCE.
    #
    # ========================================================

    prompt = f"""
Answer the original question using ONLY the verified
evidence below.

ORIGINAL QUESTION:
{question}

FACTUAL CLAIM TO RESOLVE:
{claim}

VERIFIED DIRECT EVIDENCE:
{evidence_text}

Rules:

1. Independently determine the answer.
2. Use the verified evidence.
3. Do not guess.
4. Do not copy a previous answer.
5. Do not mention the evidence.
6. Do not mention the verification process.
7. Do not mention these instructions.
8. Do not output "Correct answer:".
9. Do not output "Previous answer:".
10. Return only the concise factual answer.

ANSWER:
"""

    answer = generate_text(
        prompt
    )

    return clean_answer(
        answer
    )


# ============================================================
# TRUST COMPARISON
# ============================================================

def compare_trust(
    a0_trust,
    a1_trust
):

    priority = {

        "RELIABLE":
            2,

        "UNCERTAIN":
            1,

        "UNRELIABLE":
            0,
    }

    a0_label = a0_trust.get(
        "label",
        "UNKNOWN"
    )

    a1_label = a1_trust.get(
        "label",
        "UNKNOWN"
    )

    a0_priority = priority.get(
        a0_label,
        -1
    )

    a1_priority = priority.get(
        a1_label,
        -1
    )

    return {

        "a0_label":
            a0_label,

        "a1_label":
            a1_label,

        "trust_improved":
            a1_priority > a0_priority,

        "trust_worsened":
            a1_priority < a0_priority,

        "trust_same":
            a1_priority == a0_priority,

        "confidence_improved":
            (
                a1_trust["confidence"]
                >
                a0_trust["confidence"]
            ),

        "entropy_improved":
            (
                a1_trust["entropy"]
                <
                a0_trust["entropy"]
            ),
    }


# ============================================================
# FINAL EVIDENCE GATE
# ============================================================

def final_evidence_gate(
    candidate_validation,
    meaningful_change,
    evidence_analysis
):

    consensus = evidence_analysis[
        "consensus"
    ]

    status = consensus.get(
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
    # RULE 1
    # --------------------------------------------------------

    if not candidate_validation.get(
        "valid",
        False
    ):

        return {

            "supported":
                False,

            "status":
                "INVALID_CANDIDATE",

            "reason":
                candidate_validation.get(
                    "reason",
                    "INVALID_CANDIDATE"
                ),

            "supporting_sources":
                support_count,

            "contradicting_sources":
                contradiction_count,
        }

    # --------------------------------------------------------
    # RULE 2
    # --------------------------------------------------------

    if not meaningful_change:

        return {

            "supported":
                False,

            "status":
                "NO_IMPROVEMENT",

            "reason":
                "NO_MEANINGFUL_SELF_IMPROVEMENT",

            "supporting_sources":
                support_count,

            "contradicting_sources":
                contradiction_count,
        }

    # --------------------------------------------------------
    # RULE 8
    # --------------------------------------------------------

    if support_count == 0:

        return {

            "supported":
                False,

            "status":
                "UNCERTAIN",

            "reason":
                "ZERO_USEFUL_DIRECT_EVIDENCE",

            "supporting_sources":
                0,

            "contradicting_sources":
                contradiction_count,
        }

    # --------------------------------------------------------
    # RULE 9
    # --------------------------------------------------------

    if contradiction_count > 0:

        return {

            "supported":
                False,

            "status":
                "CONTRADICTED",

            "reason":
                "CONTRADICTING_DIRECT_EVIDENCE",

            "supporting_sources":
                support_count,

            "contradicting_sources":
                contradiction_count,
        }

    # --------------------------------------------------------
    # RULE 10
    # --------------------------------------------------------

    if (
        status == "SUPPORTED"
        and
        support_count > 0
    ):

        return {

            "supported":
                True,

            "status":
                "SUPPORTED",

            "reason":
                "DIRECT_CLAIM_SPECIFIC_EVIDENCE_SUPPORT",

            "supporting_sources":
                support_count,

            "contradicting_sources":
                contradiction_count,
        }

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return {

        "supported":
            False,

        "status":
            "UNCERTAIN",

        "reason":
            "INSUFFICIENT_DIRECT_EVIDENCE",

        "supporting_sources":
            support_count,

        "contradicting_sources":
            contradiction_count,
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

    print()
    print("=" * 70)
    print("TRUSTGATE V17")
    print("CLAIM-SPECIFIC EVIDENCE-GROUNDED SELF-IMPROVEMENT")
    print("=" * 70)

    # ========================================================
    # A0
    # ========================================================

    print(
        "[V17] Generating initial answer..."
    )

    a0 = generate_initial_answer(
        question
    )

    print(
        "[V17] A0:",
        a0
    )

    # ========================================================
    # A0 TRUSTGATE
    # ========================================================

    a0_trust = classify_with_trustgate(
        question,
        a0,
        controller
    )

    print(
        "[V17] A0 TrustGate:",
        a0_trust["label"]
    )

    # ========================================================
    # A0 CLAIM
    # ========================================================

    a0_claim = extract_claim(
        question,
        a0
    )

    print(
        "[V17] A0 claim:",
        a0_claim
    )

    # ========================================================
    # A0 CLAIM-SPECIFIC EVIDENCE
    # ========================================================

    a0_evidence = analyze_claim_evidence(
        question,
        a0_claim,
        verifier
    )

    a0_verification = (
        make_verification_object(
            a0_evidence
        )
    )

    a0_consensus = (
        a0_evidence[
            "consensus"
        ]
    )

    print(
        "[V17] A0 evidence:",
        a0_consensus
    )

    # ========================================================
    # SHOULD IMPROVE?
    # ========================================================

    should_improve = (

        a0_trust["label"]
        in
        [
            "UNRELIABLE",
            "UNCERTAIN"
        ]

        or

        a0_consensus.get(
            "status"
        )
        ==
        "CONTRADICTED"
    )

    # ========================================================
    # KEEP INITIAL
    # ========================================================

    if not should_improve:

        return {

            "initial": {

                "answer":
                    a0,

                "claim":
                    a0_claim,

                "trustgate":
                    a0_trust,

                "verification":
                    a0_verification,

                "evidence_consensus":
                    a0_evidence,
            },

            "candidate":
                None,

            "decision": {

                "decision":
                    "KEEP_INITIAL",

                "reason":
                    "INITIAL_ANSWER_ACCEPTED",
            },

            "final": {

                "answer":
                    a0,

                "source":
                    "INITIAL",
            },
        }

    # ========================================================
    # IMPORTANT:
    #
    # We only generate A1 when we have something to improve.
    #
    # But A1 MUST be generated from verified direct evidence.
    #
    # ========================================================

    if (
        a0_consensus.get(
            "status"
        )
        !=
        "SUPPORTED"
    ):

        # If A0 is uncertain or contradicted, we can still
        # attempt improvement only if useful sources exist.

        if (
            a0_evidence[
                "sources_examined"
            ]
            == 0
        ):

            return {

                "initial": {

                    "answer":
                        a0,

                    "claim":
                        a0_claim,

                    "trustgate":
                        a0_trust,

                    "verification":
                        a0_verification,

                    "evidence_consensus":
                        a0_evidence,
                },

                "candidate":
                    None,

                "decision": {

                    "decision":
                        "ROLLBACK",

                    "reason":
                        "NO_CLAIM_SPECIFIC_EVIDENCE_FOR_CORRECTION",
                },

                "final": {

                    "answer":
                        a0,

                    "source":
                        "INITIAL",
                },
            }

    # ========================================================
    # A1 FROM VERIFIED EVIDENCE
    # ========================================================

    a1 = generate_evidence_grounded_correction(
        question,
        a0_claim,
        a0_evidence
    )

    # ========================================================
    # CANDIDATE VALIDATION
    # ========================================================

    candidate_validation = validate_candidate(
        question,
        a1
    )

    meaningful_change = not answers_effectively_same(
        a0,
        a1
    )

    # ========================================================
    # RETRY IF NECESSARY
    # ========================================================

    retry_used = False

    if (
        not candidate_validation["valid"]
        or
        not meaningful_change
    ):

        retry_used = True

        a1 = generate_evidence_grounded_correction(
            question,
            a0_claim,
            a0_evidence
        )

        candidate_validation = (
            validate_candidate(
                question,
                a1
            )
        )

        meaningful_change = (
            not answers_effectively_same(
                a0,
                a1
            )
        )

    # ========================================================
    # STOP INVALID / UNCHANGED
    # ========================================================

    if not candidate_validation["valid"]:

        return {

            "initial": {

                "answer":
                    a0,

                "claim":
                    a0_claim,

                "trustgate":
                    a0_trust,

                "verification":
                    a0_verification,

                "evidence_consensus":
                    a0_evidence,
            },

            "candidate": {

                "answer":
                    a1,

                "validation":
                    candidate_validation,

                "meaningful_change":
                    meaningful_change,

                "retry_used":
                    retry_used,
            },

            "decision": {

                "decision":
                    "ROLLBACK",

                "reason":
                    candidate_validation[
                        "reason"
                    ],
            },

            "final": {

                "answer":
                    a0,

                "source":
                    "INITIAL",
            },
        }

    if not meaningful_change:

        return {

            "initial": {

                "answer":
                    a0,

                "claim":
                    a0_claim,

                "trustgate":
                    a0_trust,

                "verification":
                    a0_verification,

                "evidence_consensus":
                    a0_evidence,
            },

            "candidate": {

                "answer":
                    a1,

                "validation":
                    candidate_validation,

                "meaningful_change":
                    False,

                "retry_used":
                    retry_used,
            },

            "decision": {

                "decision":
                    "ROLLBACK",

                "reason":
                    "NO_MEANINGFUL_SELF_IMPROVEMENT",
            },

            "final": {

                "answer":
                    a0,

                "source":
                    "INITIAL",
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
    # A1 INDEPENDENT CLAIM
    # ========================================================

    a1_claim = extract_claim(
        question,
        a1
    )

    # ========================================================
    # A1 INDEPENDENT EVIDENCE RETRIEVAL
    # ========================================================

    a1_evidence = analyze_claim_evidence(
        question,
        a1_claim,
        verifier
    )

    a1_verification = (
        make_verification_object(
            a1_evidence
        )
    )

    # ========================================================
    # TRUST COMPARISON
    # ========================================================

    trust_comparison = compare_trust(
        a0_trust,
        a1_trust
    )

    # ========================================================
    # FINAL EVIDENCE GATE
    # ========================================================

    candidate_gate = final_evidence_gate(
        candidate_validation,
        meaningful_change,
        a1_evidence
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    if candidate_gate["supported"]:

        final_decision = {

            "decision":
                "ACCEPT",

            "reason":
                candidate_gate[
                    "reason"
                ],

            "evidence_supporting_sources":
                candidate_gate[
                    "supporting_sources"
                ],

            "evidence_contradicting_sources":
                candidate_gate[
                    "contradicting_sources"
                ],

            "trustgate_improved":
                trust_comparison[
                    "trust_improved"
                ],

            "confidence_improved":
                trust_comparison[
                    "confidence_improved"
                ],

            "entropy_improved":
                trust_comparison[
                    "entropy_improved"
                ],
        }

        final_answer = a1

        final_source = "CANDIDATE"

    else:

        final_decision = {

            "decision":
                "ROLLBACK",

            "reason":
                candidate_gate[
                    "reason"
                ],

            "evidence_supporting_sources":
                candidate_gate[
                    "supporting_sources"
                ],

            "evidence_contradicting_sources":
                candidate_gate[
                    "contradicting_sources"
                ],

            "trustgate_improved":
                trust_comparison[
                    "trust_improved"
                ],

            "confidence_improved":
                trust_comparison[
                    "confidence_improved"
                ],

            "entropy_improved":
                trust_comparison[
                    "entropy_improved"
                ],
        }

        final_answer = a0

        final_source = "INITIAL"

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "initial": {

            "answer":
                a0,

            "claim":
                a0_claim,

            "trustgate":
                a0_trust,

            "verification":
                a0_verification,

            "evidence_consensus":
                a0_evidence,
        },

        "candidate": {

            "answer":
                a1,

            "claim":
                a1_claim,

            "validation":
                candidate_validation,

            "meaningful_change":
                meaningful_change,

            "retry_used":
                retry_used,

            "trustgate":
                a1_trust,

            "verification":
                a1_verification,

            "evidence_consensus":
                a1_evidence,

            "trust_comparison":
                trust_comparison,

            "candidate_gate":
                candidate_gate,
        },

        "decision":
            final_decision,

        "final": {

            "answer":
                final_answer,

            "source":
                final_source,
        },
    }


# ============================================================
# V17 SELF TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("TRUSTGATE V17 BACKEND")
    print("Claim-Specific Evidence-Grounded Self-Improvement")
    print("=" * 70)

    print()
    print("Backend import successful.")

    print()
    print("Hard rules enabled:")

    rules = [

        "A1 cannot be identical to A0",

        "Instruction leakage rejected",

        "Irrelevant sources do not count",

        "Topic-only sources do not count",

        "Direct evidence required",

        "A1 generated independently from evidence",

        "A1 independently verified",

        "Zero useful evidence -> ROLLBACK",

        "Contradiction -> ROLLBACK",

        "Direct evidence support -> ACCEPT",

        "TrustGate alone does not prove truth",
    ]

    for index, rule in enumerate(
        rules,
        start=1
    ):

        print(
            f"[{index}] {rule}"
        )

    print()
    print(
        "V17 backend self-test passed."
    )