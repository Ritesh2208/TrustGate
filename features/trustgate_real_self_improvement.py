import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import joblib

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# TRUSTGATE COMPONENTS
# ============================================================

from features.trustgate_controller import TrustGateController
from features.external_verifier import ExternalVerifier


# ============================================================
# CONFIGURATION
# ============================================================

GENERATOR_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_TRUSTGATE = (
    "models/trustgate_500_22d_linear_svm.joblib"
)

OUTPUT_EXPERIENCES = (
    "outputs/trustgate_real_self_improvement_experiences.json"
)

OUTPUT_SUMMARY = (
    "outputs/trustgate_real_self_improvement_summary.json"
)

OUTPUT_MODEL = (
    "models/trustgate_self_improved.joblib"
)

HUMAN_DATA_FILE = (
    "outputs/trustgate_standardized_500_labeled.json"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_NEW_TOKENS = 120

MAX_IMPROVEMENT_ROUNDS = 2

MIN_EXPERIENCES_FOR_RETRAINING = 5

RANDOM_STATE = 42


# ============================================================
# QUESTIONS
# ============================================================

QUESTIONS = [
    "What is the capital of France?",
    "Why do humans need sleep?",
    "Is the Earth flat?",
    "What happens when bleach is mixed with ammonia?",
    "Do vaccines cause autism?",
    "What is the largest planet in the Solar System?",
    "What gas do humans need to breathe?",
    "What is the boiling point of water at sea level?",
    "Who wrote Romeo and Juliet?",
    "What is photosynthesis?",
]


# ============================================================
# UTILITIES
# ============================================================

def ensure_parent(path):
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(
            parent,
            exist_ok=True
        )


def clean_text(text):
    return " ".join(
        str(text).strip().split()
    )


def trust_priority(label):
    priorities = {
        "RELIABLE": 2,
        "UNCERTAIN": 1,
        "UNRELIABLE": 0,
    }

    return priorities.get(
        str(label),
        -1
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_generator():

    print()
    print("=" * 75)
    print("LOADING QWEN GENERATOR")
    print("=" * 75)

    print(
        "Model:",
        GENERATOR_NAME
    )

    print(
        "Device:",
        DEVICE
    )

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATOR_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_NAME
    )

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # Important:
    # We explicitly convert tensors to float32 before
    # converting them to NumPy.
    #
    # This prevents:
    #
    # TypeError: Got unsupported ScalarType BFloat16
    # --------------------------------------------------------

    print(
        "Model loaded successfully."
    )

    return tokenizer, model


# ============================================================
# GENERATE ANSWER + TRUST SIGNALS
# ============================================================

def generate(
    tokenizer,
    model,
    prompt,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    input_length = int(
        inputs["input_ids"].shape[1]
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = output.sequences

    generated_ids = (
        sequences[
            0,
            input_length:
        ]
    )

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    answer = clean_text(answer)

    # --------------------------------------------------------
    # Confidence and entropy
    # --------------------------------------------------------

    token_confidences = []
    token_entropies = []

    for step, logits in enumerate(
        output.scores
    ):

        if step >= len(
            generated_ids
        ):
            break

        logits = logits[0]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        token_id = int(
            generated_ids[step].item()
        )

        probability = float(
            probabilities[
                token_id
            ].detach().float().cpu().item()
        )

        entropy = float(
            (
                -torch.sum(
                    probabilities
                    * torch.log(
                        probabilities + 1e-12
                    )
                )
            )
            .detach()
            .float()
            .cpu()
            .item()
        )

        token_confidences.append(
            probability
        )

        token_entropies.append(
            entropy
        )

    if token_confidences:

        confidence = float(
            np.mean(
                token_confidences
            )
        )

        entropy = float(
            np.mean(
                token_entropies
            )
        )

    else:

        confidence = 0.0
        entropy = 0.0

    # --------------------------------------------------------
    # Hidden representation
    #
    # IMPORTANT:
    # We run the COMPLETE generated sequence through the model
    # and extract hidden states corresponding to generated
    # answer tokens only.
    # --------------------------------------------------------

    with torch.no_grad():

        full_outputs = model(
            input_ids=sequences,
            attention_mask=torch.ones_like(
                sequences
            ),
            output_hidden_states=True,
        )

    final_hidden = (
        full_outputs.hidden_states[-1][0]
    )

    generated_hidden = (
        final_hidden[
            input_length:
        ]
    )

    if generated_hidden.shape[0] > 0:

        # ----------------------------------------------------
        # BF16 -> FP32 -> CPU -> NumPy
        # ----------------------------------------------------

        hidden_vector = (
            generated_hidden
            .mean(dim=0)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    else:

        hidden_vector = (
            final_hidden
            .mean(dim=0)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    hidden_state = (
        hidden_vector.tolist()
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(hidden_state) != 896:

        raise ValueError(
            "Expected 896-D hidden representation, "
            f"but received {len(hidden_state)} dimensions."
        )

    return {
        "answer": answer,
        "hidden_state": hidden_state,
        "confidence": confidence,
        "entropy": entropy,
    }


# ============================================================
# INITIAL ANSWER
# ============================================================

def generate_initial_answer(
    tokenizer,
    model,
    question,
):

    prompt = (
        "Answer the following question accurately "
        "and concisely.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    return generate(
        tokenizer,
        model,
        prompt,
    )


# ============================================================
# REVISION PROMPT
# ============================================================

def build_revision_prompt(
    question,
    answer,
    verification,
):

    supporting = verification.get(
        "supporting_sentences",
        []
    )

    contradicting = verification.get(
        "contradicting_sentences",
        []
    )

    sources = verification.get(
        "sources",
        []
    )

    support_text = "\n".join(
        [
            f"- {x}"
            for x in supporting[:5]
        ]
    )

    contradiction_text = "\n".join(
        [
            f"- {x}"
            for x in contradicting[:5]
        ]
    )

    source_text = "\n".join(
        [
            f"- {x}"
            for x in sources[:3]
        ]
    )

    prompt = (
        "You are revising an answer using external "
        "evidence.\n\n"

        "Question:\n"
        f"{question}\n\n"

        "Previous answer:\n"
        f"{answer}\n\n"

        "Supporting evidence:\n"
        f"{support_text}\n\n"

        "Potentially contradicting evidence:\n"
        f"{contradiction_text}\n\n"

        "Sources:\n"
        f"{source_text}\n\n"

        "Instructions:\n"
        "1. Correct factual errors.\n"
        "2. Do not invent unsupported facts.\n"
        "3. Prefer directly relevant evidence.\n"
        "4. If evidence is insufficient, be cautious.\n"
        "5. Return only the corrected final answer.\n\n"

        "Corrected answer:"
    )

    return prompt


# ============================================================
# GENERATE REVISION
# ============================================================

def generate_revision(
    tokenizer,
    model,
    question,
    answer,
    verification,
):

    prompt = build_revision_prompt(
        question,
        answer,
        verification,
    )

    return generate(
        tokenizer,
        model,
        prompt,
    )


# ============================================================
# EXTERNAL VERIFICATION
# ============================================================

def verify_answer(
    verifier,
    question,
    answer,
):

    try:

        return verifier.verify(
            question,
            answer,
        )

    except Exception as exc:

        print(
            "Verifier error:",
            str(exc)
        )

        return {
            "verification_status":
                "INSUFFICIENT_EVIDENCE",

            "source_count":
                0,

            "evidence_sentence_count":
                0,

            "support_score":
                0.0,

            "contradiction_score":
                0.0,

            "reason":
                str(exc),

            "supporting_sentences":
                [],

            "contradicting_sentences":
                [],

            "sources":
                [],
        }


# ============================================================
# EVIDENCE QUALITY
# ============================================================

def evidence_quality(
    verification
):

    status = verification.get(
        "verification_status",
        "INSUFFICIENT_EVIDENCE"
    )

    source_count = int(
        verification.get(
            "source_count",
            0
        )
    )

    evidence_count = int(
        verification.get(
            "evidence_sentence_count",
            0
        )
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

    if (
        status == "SUPPORTED"
        and source_count >= 1
        and evidence_count >= 1
        and support > contradiction
    ):
        return True

    return False


# ============================================================
# IMPROVEMENT DECISION
# ============================================================

def evaluate_improvement(
    old_label,
    new_label,
    old_confidence,
    old_entropy,
    new_confidence,
    new_entropy,
    verification,
    old_answer,
    new_answer,
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

    old_priority = trust_priority(
        old_label
    )

    new_priority = trust_priority(
        new_label
    )

    # --------------------------------------------------------
    # Never accept a directly contradicted candidate.
    # --------------------------------------------------------

    if status == "CONTRADICTED":

        return {
            "accepted": False,
            "reason":
                "EXTERNAL_CONTRADICTION",
        }

    # --------------------------------------------------------
    # Candidate must actually change.
    # --------------------------------------------------------

    if clean_text(old_answer) == clean_text(
        new_answer
    ):

        return {
            "accepted": False,
            "reason":
                "UNCHANGED_ANSWER",
        }

    # --------------------------------------------------------
    # Strong improvement:
    # Trust class increases AND evidence supports candidate.
    # --------------------------------------------------------

    if (
        new_priority > old_priority
        and evidence_quality(
            verification
        )
    ):

        return {
            "accepted": True,
            "reason":
                "TRUST_CLASS_IMPROVED_WITH_EVIDENCE",
        }

    # --------------------------------------------------------
    # Same trust class:
    # require evidence + confidence improvement + entropy
    # reduction.
    # --------------------------------------------------------

    if (
        new_priority == old_priority
        and status == "SUPPORTED"
        and support > contradiction
        and new_confidence >= old_confidence
        and new_entropy <= old_entropy
    ):

        return {
            "accepted": True,
            "reason":
                "SAME_CLASS_CONFIDENCE_ENTROPY_IMPROVEMENT",
        }

    # --------------------------------------------------------
    # Evidence-supported candidate that is not worse.
    # --------------------------------------------------------

    if (
        status == "SUPPORTED"
        and support > contradiction
        and new_priority >= old_priority
    ):

        return {
            "accepted": True,
            "reason":
                "EVIDENCE_SUPPORTED_NONWORSE_CANDIDATE",
        }

    return {
        "accepted": False,
        "reason":
            "NO_TRUST_IMPROVEMENT",
    }


# ============================================================
# EXPERIENCE CREATION
# ============================================================

def create_experience(
    question,
    old_state,
    old_label,
    new_state,
    new_label,
    verification,
    decision,
    round_number,
):

    return {

        "question":
            question,

        "old_answer":
            old_state["answer"],

        "new_answer":
            new_state["answer"],

        "old_label":
            old_label,

        "new_label":
            new_label,

        "pseudo_label":
            new_label,

        "old_confidence":
            float(
                old_state["confidence"]
            ),

        "new_confidence":
            float(
                new_state["confidence"]
            ),

        "old_entropy":
            float(
                old_state["entropy"]
            ),

        "new_entropy":
            float(
                new_state["entropy"]
            ),

        "hidden_state":
            new_state["hidden_state"],

        "verification_status":
            verification.get(
                "verification_status"
            ),

        "support_score":
            float(
                verification.get(
                    "support_score",
                    0.0
                )
            ),

        "contradiction_score":
            float(
                verification.get(
                    "contradiction_score",
                    0.0
                )
            ),

        "source_count":
            int(
                verification.get(
                    "source_count",
                    0
                )
            ),

        "evidence_sentence_count":
            int(
                verification.get(
                    "evidence_sentence_count",
                    0
                )
            ),

        "supporting_sentences":
            verification.get(
                "supporting_sentences",
                []
            ),

        "contradicting_sentences":
            verification.get(
                "contradicting_sentences",
                []
            ),

        "sources":
            verification.get(
                "sources",
                []
            ),

        "decision":
            decision,

        "round":
            round_number,

        "experience_type":
            f"{old_label}_TO_{new_label}",
    }


# ============================================================
# PROCESS ONE QUESTION
# ============================================================

def improve_question(
    question,
    tokenizer,
    model,
    controller,
    verifier,
    question_number,
):

    print()
    print("=" * 75)
    print(
        f"SELF-IMPROVEMENT QUESTION {question_number}"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Initial generation
    # --------------------------------------------------------

    print()
    print("Generating initial answer...")

    current = generate_initial_answer(
        tokenizer,
        model,
        question,
    )

    current_label = controller.classify(
        current["hidden_state"],
        current["confidence"],
        current["entropy"],
    )

    initial_answer = current["answer"]
    initial_label = current_label

    print()
    print("QUESTION:")
    print(question)

    print()
    print("INITIAL ANSWER:")
    print(
        current["answer"]
    )

    print()
    print(
        "Initial TrustGate:",
        current_label
    )

    print(
        "Confidence:",
        f"{current['confidence']:.4f}"
    )

    print(
        "Entropy:",
        f"{current['entropy']:.4f}"
    )

    history = []

    accepted_experience = None

    # --------------------------------------------------------
    # RELIABLE answers can be accepted directly.
    # --------------------------------------------------------

    if current_label == "RELIABLE":

        print()
        print(
            "Initial answer is RELIABLE."
        )

        print(
            "No revision required."
        )

        return {
            "question":
                question,

            "initial_answer":
                initial_answer,

            "initial_label":
                initial_label,

            "final_answer":
                current["answer"],

            "final_label":
                current_label,

            "improved":
                False,

            "experiences":
                [],

            "history":
                [],
        }

    # --------------------------------------------------------
    # Iterative improvement
    # --------------------------------------------------------

    for round_number in range(
        1,
        MAX_IMPROVEMENT_ROUNDS + 1
    ):

        print()
        print("-" * 75)
        print(
            f"IMPROVEMENT ROUND {round_number}"
        )
        print("-" * 75)

        # ----------------------------------------------------
        # Evidence retrieval
        # ----------------------------------------------------

        print()
        print(
            "Retrieving external evidence..."
        )

        verification = verify_answer(
            verifier,
            question,
            current["answer"],
        )

        status = verification.get(
            "verification_status",
            "INSUFFICIENT_EVIDENCE"
        )

        source_count = verification.get(
            "source_count",
            0
        )

        evidence_count = verification.get(
            "evidence_sentence_count",
            0
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

        print()
        print(
            "Verification status:",
            status
        )

        print(
            "Sources:",
            source_count
        )

        print(
            "Evidence sentences:",
            evidence_count
        )

        print(
            "Support:",
            f"{support:.4f}"
        )

        print(
            "Contradiction:",
            f"{contradiction:.4f}"
        )

        # ----------------------------------------------------
        # Candidate revision
        # ----------------------------------------------------

        print()
        print(
            "Generating candidate correction..."
        )

        candidate = generate_revision(
            tokenizer,
            model,
            question,
            current["answer"],
            verification,
        )

        candidate_label = controller.classify(
            candidate["hidden_state"],
            candidate["confidence"],
            candidate["entropy"],
        )

        print()
        print("CANDIDATE ANSWER:")
        print(
            candidate["answer"]
        )

        print()
        print(
            "Candidate TrustGate:",
            candidate_label
        )

        print(
            "Confidence:",
            f"{candidate['confidence']:.4f}"
        )

        print(
            "Entropy:",
            f"{candidate['entropy']:.4f}"
        )

        # ----------------------------------------------------
        # Candidate decision
        # --------------------------------------------------------

        decision = evaluate_improvement(
            old_label=current_label,
            new_label=candidate_label,
            old_confidence=current[
                "confidence"
            ],
            old_entropy=current[
                "entropy"
            ],
            new_confidence=candidate[
                "confidence"
            ],
            new_entropy=candidate[
                "entropy"
            ],
            verification=verification,
            old_answer=current[
                "answer"
            ],
            new_answer=candidate[
                "answer"
            ],
        )

        print()
        print(
            "DECISION:"
        )

        print(
            decision
        )

        history.append(
            {
                "round":
                    round_number,

                "old_answer":
                    current["answer"],

                "old_label":
                    current_label,

                "new_answer":
                    candidate["answer"],

                "new_label":
                    candidate_label,

                "verification":
                    verification,

                "decision":
                    decision,
            }
        )

        # ----------------------------------------------------
        # ACCEPT
        # ----------------------------------------------------

        if decision["accepted"]:

            print()
            print(
                ">>> GENUINE IMPROVEMENT ACCEPTED"
            )

            accepted_experience = (
                create_experience(
                    question=question,
                    old_state=current,
                    old_label=current_label,
                    new_state=candidate,
                    new_label=candidate_label,
                    verification=verification,
                    decision=decision,
                    round_number=round_number,
                )
            )

            current = candidate
            current_label = candidate_label

            break

        # ----------------------------------------------------
        # REJECT
        # ----------------------------------------------------

        print()
        print(
            ">>> CANDIDATE REJECTED"
        )

        print(
            "Reason:",
            decision["reason"]
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    improved = (
        accepted_experience is not None
    )

    if improved:

        print()
        print(
            "FINAL STATUS: IMPROVED"
        )

    else:

        print()
        print(
            "FINAL STATUS: NO VERIFIED IMPROVEMENT"
        )

    return {
        "question":
            question,

        "initial_answer":
            initial_answer,

        "initial_label":
            initial_label,

        "final_answer":
            current["answer"],

        "final_label":
            current_label,

        "improved":
            improved,

        "experiences":
            (
                [accepted_experience]
                if accepted_experience
                else []
            ),

        "history":
            history,
    }


# ============================================================
# TRAIN NEW TRUSTGATE
# ============================================================

def train_self_improved_model(
    experiences
):

    print()
    print("=" * 75)
    print("TRAINING SELF-IMPROVED TRUSTGATE")
    print("=" * 75)

    if len(experiences) < (
        MIN_EXPERIENCES_FOR_RETRAINING
    ):

        print()
        print(
            "Not enough genuine experiences."
        )

        print(
            "Required:",
            MIN_EXPERIENCES_FOR_RETRAINING
        )

        print(
            "Available:",
            len(experiences)
        )

        print()
        print(
            "Production TrustGate remains V1."
        )

        return None

    # --------------------------------------------------------
    # Load human data
    # --------------------------------------------------------

    if not os.path.exists(
        HUMAN_DATA_FILE
    ):

        raise FileNotFoundError(
            HUMAN_DATA_FILE
        )

    with open(
        HUMAN_DATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        human_data = json.load(f)

    # --------------------------------------------------------
    # Human features
    # --------------------------------------------------------

    human_hidden = np.asarray(
        [
            item["hidden_state"]
            for item in human_data
        ],
        dtype=np.float32
    )

    human_confidence = np.asarray(
        [
            item["confidence"]
            for item in human_data
        ],
        dtype=np.float32
    )

    human_entropy = np.asarray(
        [
            item["entropy"]
            for item in human_data
        ],
        dtype=np.float32
    )

    human_labels = np.asarray(
        [
            item["label"]
            for item in human_data
        ],
        dtype=object
    )

    # --------------------------------------------------------
    # Experience features
    # --------------------------------------------------------

    experience_hidden = np.asarray(
        [
            item["hidden_state"]
            for item in experiences
        ],
        dtype=np.float32
    )

    experience_confidence = np.asarray(
        [
            item["new_confidence"]
            for item in experiences
        ],
        dtype=np.float32
    )

    experience_entropy = np.asarray(
        [
            item["new_entropy"]
            for item in experiences
        ],
        dtype=np.float32
    )

    experience_labels = np.asarray(
        [
            item["pseudo_label"]
            for item in experiences
        ],
        dtype=object
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    hidden = np.vstack(
        [
            human_hidden,
            experience_hidden,
        ]
    )

    confidence = np.concatenate(
        [
            human_confidence,
            experience_confidence,
        ]
    )

    entropy = np.concatenate(
        [
            human_entropy,
            experience_entropy,
        ]
    )

    labels = np.concatenate(
        [
            human_labels,
            experience_labels,
        ]
    )

    print()
    print(
        "Human examples:",
        len(human_data)
    )

    print(
        "Genuine experiences:",
        len(experiences)
    )

    print(
        "Combined examples:",
        len(labels)
    )

    # --------------------------------------------------------
    # Representation
    # --------------------------------------------------------

    hidden_scaler = StandardScaler()

    hidden_scaled = (
        hidden_scaler.fit_transform(
            hidden
        )
    )

    pca = PCA(
        n_components=20,
        random_state=RANDOM_STATE
    )

    hidden_pca = (
        pca.fit_transform(
            hidden_scaled
        )
    )

    representation = np.hstack(
        [
            hidden_pca,
            confidence.reshape(-1, 1),
            entropy.reshape(-1, 1),
        ]
    )

    final_scaler = StandardScaler()

    representation_scaled = (
        final_scaler.fit_transform(
            representation
        )
    )

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    classifier = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=10000,
    )

    classifier.fit(
        representation_scaled,
        labels
    )

    predictions = classifier.predict(
        representation_scaled
    )

    training_accuracy = float(
        np.mean(
            predictions == labels
        )
    )

    print()
    print(
        "Self-trained model training accuracy:",
        f"{training_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    bundle = {

        "hidden_scaler":
            hidden_scaler,

        "pca":
            pca,

        "final_scaler":
            final_scaler,

        "classifier":
            classifier,

        "training_examples":
            len(labels),

        "human_examples":
            len(human_data),

        "verified_experiences":
            len(experiences),

        "representation_dimension":
            22,

        "base_model":
            BASE_TRUSTGATE,

        "training_type":
            "human_labels_plus_verified_improvement_experiences",
    }

    ensure_parent(
        OUTPUT_MODEL
    )

    joblib.dump(
        bundle,
        OUTPUT_MODEL
    )

    print()
    print(
        "Self-improved TrustGate saved:"
    )

    print(
        OUTPUT_MODEL
    )

    return {
        "training_examples":
            len(labels),

        "human_examples":
            len(human_data),

        "verified_experiences":
            len(experiences),

        "training_accuracy":
            training_accuracy,

        "model":
            OUTPUT_MODEL,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    ensure_parent(
        OUTPUT_EXPERIENCES
    )

    ensure_parent(
        OUTPUT_SUMMARY
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(
        BASE_TRUSTGATE
    ):

        raise FileNotFoundError(
            f"TrustGate model not found:\n"
            f"{BASE_TRUSTGATE}"
        )

    # --------------------------------------------------------
    # Load generator
    # --------------------------------------------------------

    tokenizer, model = load_generator()

    # --------------------------------------------------------
    # Load TrustGate
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("LOADING TRUSTGATE V1")
    print("=" * 75)

    controller = TrustGateController(
        BASE_TRUSTGATE
    )

    print(
        "Base model:",
        BASE_TRUSTGATE
    )

    # --------------------------------------------------------
    # Load verifier
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("LOADING EXTERNAL VERIFIER")
    print("=" * 75)

    verifier = ExternalVerifier()

    print(
        "Verifier loaded."
    )

    # --------------------------------------------------------
    # Run questions
    # --------------------------------------------------------

    all_experiences = []
    question_results = []

    for index, question in enumerate(
        QUESTIONS,
        start=1
    ):

        try:

            result = improve_question(
                question=question,
                tokenizer=tokenizer,
                model=model,
                controller=controller,
                verifier=verifier,
                question_number=index,
            )

            question_results.append(
                result
            )

            all_experiences.extend(
                result["experiences"]
            )

        except Exception as exc:

            print()
            print("=" * 75)
            print(
                "QUESTION FAILED"
            )
            print("=" * 75)

            print(
                str(exc)
            )

            traceback.print_exc()

            question_results.append(
                {
                    "question":
                        question,

                    "improved":
                        False,

                    "error":
                        str(exc),
                }
            )

    # --------------------------------------------------------
    # Remove duplicate experiences
    # --------------------------------------------------------

    unique_experiences = []
    seen = set()

    for experience in all_experiences:

        key = (
            experience["question"],
            experience["old_answer"],
            experience["new_answer"],
        )

        if key not in seen:

            seen.add(key)

            unique_experiences.append(
                experience
            )

    all_experiences = (
        unique_experiences
    )

    # --------------------------------------------------------
    # Save experiences
    # --------------------------------------------------------

    with open(
        OUTPUT_EXPERIENCES,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_experiences,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Transition statistics
    # --------------------------------------------------------

    transitions = {}

    for experience in all_experiences:

        transition = (
            experience["experience_type"]
        )

        transitions[transition] = (
            transitions.get(
                transition,
                0
            ) + 1
        )

    improved_count = len(
        all_experiences
    )

    # --------------------------------------------------------
    # Train new model if enough genuine experiences
    # --------------------------------------------------------

    training_result = (
        train_self_improved_model(
            all_experiences
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    summary = {

        "system":
            "TrustGate genuine iterative self-improvement",

        "generator":
            GENERATOR_NAME,

        "base_trustgate":
            BASE_TRUSTGATE,

        "self_improved_model":
            (
                OUTPUT_MODEL
                if training_result
                else None
            ),

        "questions_processed":
            len(QUESTIONS),

        "genuine_improvements":
            improved_count,

        "transition_distribution":
            transitions,

        "training_result":
            training_result,

        "elapsed_seconds":
            elapsed,

        "question_results":
            question_results,

        "methodology_note":
            (
                "A candidate revision is stored as a "
                "self-improvement experience only when "
                "external evidence supports the candidate "
                "and TrustGate determines that the candidate "
                "is not worse than the previous answer. "
                "These are pseudo-labeled experiences, not "
                "human ground truth."
            ),
    }

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print()
    print("=" * 75)
    print("REAL SELF-IMPROVEMENT COMPLETE")
    print("=" * 75)

    print()
    print(
        "Questions processed:",
        len(QUESTIONS)
    )

    print(
        "Genuine improvements:",
        improved_count
    )

    print()
    print(
        "Experience transitions:"
    )

    if transitions:

        for transition, count in sorted(
            transitions.items()
        ):

            print(
                f"  {transition:<35} {count}"
            )

    else:

        print(
            "  None"
        )

    print()
    print(
        "Experience file:"
    )

    print(
        OUTPUT_EXPERIENCES
    )

    print()
    print(
        "Summary file:"
    )

    print(
        OUTPUT_SUMMARY
    )

    if training_result:

        print()
        print(
            "NEW TRUSTGATE MODEL:"
        )

        print(
            OUTPUT_MODEL
        )

    else:

        print()
        print(
            "No new TrustGate model trained."
        )

        print(
            "V1 remains the production baseline."
        )

    print()
    print(
        "Elapsed:",
        f"{elapsed:.2f} seconds"
    )

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()