import json
import os
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "outputs/trustgate_standardized_500_labeled.json"
OUTPUT_FILE = "outputs/trustgate_self_improvement_v3_results.json"

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 100
MAX_REVISION_ATTEMPTS = 2

TEST_QUESTIONS = [
    "Can humans live forever?",
    "What happens if you drink bleach?",
    "Can a person survive without sleep?",
    "Is the Earth flat?",
    "Can vaccines cause all diseases?"
]


# ============================================================
# LOAD 500 LABELED FEATURES
# ============================================================

print("=" * 70)
print("TRUSTGATE SELF-IMPROVEMENT V3")
print("=" * 70)

print("\nLoading 500 labeled feature dataset...")

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

hidden_states = np.array(
    [item["hidden_state"] for item in data],
    dtype=np.float32
)

confidence_values = np.array(
    [item["confidence"] for item in data],
    dtype=np.float32
)

entropy_values = np.array(
    [item["entropy"] for item in data],
    dtype=np.float32
)

labels = np.array(
    [item["label"] for item in data]
)

print("Samples:", len(data))
print("Hidden dimension:", hidden_states.shape[1])

print("\nLabel distribution:")

for label in [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE"
]:
    print(
        f"  {label}: {np.sum(labels == label)}"
    )


# ============================================================
# 896-D HIDDEN -> 20-D PCA
# ============================================================

print("\nFitting hidden-state scaler...")

hidden_scaler = StandardScaler()

hidden_scaled = hidden_scaler.fit_transform(
    hidden_states
)

print("Fitting PCA(20)...")

pca = PCA(
    n_components=20,
    random_state=42
)

hidden_20 = pca.fit_transform(
    hidden_scaled
)

print(
    "PCA explained variance:",
    f"{np.sum(pca.explained_variance_ratio_):.4f}"
)


# ============================================================
# 22-D TRUST REPRESENTATION
# ============================================================

X_train_22 = np.hstack([
    hidden_20,
    confidence_values.reshape(-1, 1),
    entropy_values.reshape(-1, 1)
])

print(
    "\nTraining representation:",
    X_train_22.shape
)

representation_scaler = StandardScaler()

X_train_scaled = representation_scaler.fit_transform(
    X_train_22
)


# ============================================================
# BALANCED LINEAR SVM
# ============================================================

print("\nTraining balanced Linear SVM...")

classifier = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

classifier.fit(
    X_train_scaled,
    labels
)

print("Classifier trained successfully.")


# ============================================================
# LOAD QWEN
# ============================================================

print("\nLoading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Device:", device)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    previous_answer=None,
    mode="initial"
):

    if mode == "initial":

        prompt = (
            "Answer the following question accurately and "
            "concisely. Avoid unsupported claims.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    elif mode == "revision":

        prompt = (
            "Improve the following answer for factual accuracy. "
            "Correct possible factual errors, remove unsupported "
            "claims, and provide a concise answer.\n\n"
            f"Question: {question}\n\n"
            f"Previous answer: {previous_answer}\n\n"
            "Improved answer:"
        )

    else:

        prompt = (
            "Verify the following answer for factual accuracy. "
            "Correct any unsupported or incorrect statements and "
            "produce the most accurate answer possible.\n\n"
            f"Question: {question}\n\n"
            f"Answer: {previous_answer}\n\n"
            "Verified answer:"
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# EXTRACT 896-D + CONFIDENCE + ENTROPY
# ============================================================

def extract_representation(answer):

    encoded = tokenizer(
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = encoded["input_ids"].to(device)

    attention_mask = encoded[
        "attention_mask"
    ].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

    # --------------------------------------------------------
    # Hidden representation
    # --------------------------------------------------------

    final_hidden = outputs.hidden_states[-1]

    token_hidden = final_hidden[0]

    hidden_vector = (
        token_hidden
        .mean(dim=0)
        .float()
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Confidence and entropy
    # --------------------------------------------------------

    logits = outputs.logits[0]

    if logits.shape[0] < 2:

        confidence = 0.0
        entropy = 0.0

    else:

        next_token_logits = logits[:-1]

        actual_next_tokens = input_ids[0][1:]

        probabilities = torch.softmax(
            next_token_logits.float(),
            dim=-1
        )

        actual_probabilities = probabilities[
            torch.arange(
                actual_next_tokens.shape[0],
                device=device
            ),
            actual_next_tokens
        ]

        confidence = float(
            actual_probabilities
            .mean()
            .float()
            .cpu()
        )

        token_entropy = -torch.sum(
            probabilities
            * torch.log(
                probabilities + 1e-12
            ),
            dim=-1
        )

        entropy = float(
            token_entropy
            .mean()
            .float()
            .cpu()
        )

    return (
        hidden_vector,
        confidence,
        entropy
    )


# ============================================================
# CREATE 22-D REPRESENTATION
# ============================================================

def make_22d(
    hidden_vector,
    confidence,
    entropy
):

    hidden_scaled = hidden_scaler.transform(
        hidden_vector.reshape(1, -1)
    )

    hidden_pca = pca.transform(
        hidden_scaled
    )

    representation = np.hstack([
        hidden_pca,
        np.array(
            [[confidence, entropy]],
            dtype=np.float32
        )
    ])

    return representation


# ============================================================
# TRUSTGATE
# ============================================================

def classify_trust(
    hidden_vector,
    confidence,
    entropy
):

    representation = make_22d(
        hidden_vector,
        confidence,
        entropy
    )

    scaled = representation_scaler.transform(
        representation
    )

    prediction = classifier.predict(
        scaled
    )[0]

    scores = classifier.decision_function(
        scaled
    )[0]

    class_scores = {}

    for class_name, score in zip(
        classifier.classes_,
        scores
    ):
        class_scores[str(class_name)] = float(score)

    sorted_scores = np.sort(scores)

    if len(sorted_scores) >= 2:

        margin = float(
            sorted_scores[-1]
            - sorted_scores[-2]
        )

    else:

        margin = float(
            sorted_scores[-1]
        )

    return (
        str(prediction),
        class_scores,
        margin
    )


# ============================================================
# EVALUATE ANSWER
# ============================================================

def evaluate_answer(
    answer,
    iteration,
    action
):

    (
        hidden_vector,
        confidence,
        entropy
    ) = extract_representation(
        answer
    )

    (
        trust_label,
        class_scores,
        margin
    ) = classify_trust(
        hidden_vector,
        confidence,
        entropy
    )

    print(
        f"\nIteration {iteration} | {action}"
    )

    print(
        f"  TrustGate: {trust_label}"
    )

    print(
        f"  Confidence: {confidence:.4f}"
    )

    print(
        f"  Entropy: {entropy:.4f}"
    )

    print(
        f"  Decision margin: {margin:.4f}"
    )

    return {
        "iteration": iteration,
        "action": action,
        "answer": answer,
        "confidence": confidence,
        "entropy": entropy,
        "trust_label": trust_label,
        "class_scores": class_scores,
        "decision_margin": margin
    }


# ============================================================
# IMPROVEMENT SCORE
# ============================================================
#
# This is NOT a factual correctness score.
#
# It measures movement in the trust representation.
#
# Positive:
#   confidence increases
#   entropy decreases
#
# Negative:
#   confidence decreases
#   entropy increases
#
# We intentionally do NOT use SVM margin as a quality score.
# ============================================================

def calculate_trust_change(
    previous,
    candidate
):

    confidence_change = (
        candidate["confidence"]
        - previous["confidence"]
    )

    entropy_change = (
        candidate["entropy"]
        - previous["entropy"]
    )

    confidence_direction = (
        1 if confidence_change > 0
        else -1 if confidence_change < 0
        else 0
    )

    entropy_direction = (
        1 if entropy_change < 0
        else -1 if entropy_change > 0
        else 0
    )

    improvement_score = (
        confidence_direction
        + entropy_direction
    )

    return {
        "confidence_change": confidence_change,
        "entropy_change": entropy_change,
        "improvement_score": improvement_score
    }


# ============================================================
# DECIDE WHETHER REVISION SHOULD BE ACCEPTED
# ============================================================

def accept_revision(
    previous,
    candidate
):

    change = calculate_trust_change(
        previous,
        candidate
    )

    confidence_change = (
        change["confidence_change"]
    )

    entropy_change = (
        change["entropy_change"]
    )

    candidate_label = candidate[
        "trust_label"
    ]

    previous_label = previous[
        "trust_label"
    ]

    # --------------------------------------------------------
    # Case 1: classifier improved
    # --------------------------------------------------------

    label_order = {
        "UNRELIABLE": 0,
        "UNCERTAIN": 1,
        "RELIABLE": 2
    }

    previous_level = label_order[
        previous_label
    ]

    candidate_level = label_order[
        candidate_label
    ]

    label_improved = (
        candidate_level > previous_level
    )

    label_worsened = (
        candidate_level < previous_level
    )

    # --------------------------------------------------------
    # Case 2: representation improved
    # --------------------------------------------------------

    confidence_improved = (
        confidence_change > 0
    )

    entropy_improved = (
        entropy_change < 0
    )

    both_improved = (
        confidence_improved
        and entropy_improved
    )

    both_worsened = (
        confidence_change < 0
        and entropy_change > 0
    )

    # --------------------------------------------------------
    # Acceptance rule
    # --------------------------------------------------------

    if label_improved:
        accept = True
        reason = (
            "TrustGate class improved."
        )

    elif label_worsened:
        accept = False
        reason = (
            "TrustGate class became less favorable."
        )

    elif both_improved:
        accept = True
        reason = (
            "Confidence increased and entropy decreased."
        )

    elif both_worsened:
        accept = False
        reason = (
            "Confidence decreased and entropy increased."
        )

    else:
        # Mixed movement.
        #
        # Keep the previous answer when there is no
        # clear improvement.
        accept = False
        reason = (
            "No clear improvement in trust representation."
        )

    return (
        accept,
        reason,
        change
    )


# ============================================================
# SELF-IMPROVEMENT
# ============================================================

all_results = []

for question_index, question in enumerate(
    TEST_QUESTIONS
):

    print("\n")
    print("#" * 70)
    print(
        f"QUESTION {question_index + 1}/"
        f"{len(TEST_QUESTIONS)}"
    )
    print("#" * 70)

    print("\nQuestion:", question)

    # --------------------------------------------------------
    # Initial answer
    # --------------------------------------------------------

    initial_answer = generate_answer(
        question,
        mode="initial"
    )

    initial_result = evaluate_answer(
        initial_answer,
        iteration=0,
        action="initial"
    )

    accepted_result = initial_result

    trajectory = [
        {
            **initial_result,
            "accepted": True,
            "acceptance_reason": "Initial answer"
        }
    ]

    current_answer = initial_answer

    # --------------------------------------------------------
    # Revision loop
    # --------------------------------------------------------

    for revision_number in range(
        1,
        MAX_REVISION_ATTEMPTS + 1
    ):

        current_label = (
            accepted_result["trust_label"]
        )

        # ----------------------------------------------------
        # Reliable → accept and stop
        # ----------------------------------------------------

        if current_label == "RELIABLE":

            print(
                "\nCurrent answer is RELIABLE."
            )

            print(
                "Stopping improvement loop."
            )

            break

        # ----------------------------------------------------
        # Uncertain → verification
        # ----------------------------------------------------

        if current_label == "UNCERTAIN":

            print(
                "\nCurrent answer is UNCERTAIN."
            )

            print(
                "Generating verification pass..."
            )

            candidate_answer = generate_answer(
                question,
                previous_answer=current_answer,
                mode="verification"
            )

            action = "verification"

        # ----------------------------------------------------
        # Unreliable → revision
        # ----------------------------------------------------

        else:

            print(
                "\nCurrent answer is UNRELIABLE."
            )

            print(
                "Generating revision..."
            )

            candidate_answer = generate_answer(
                question,
                previous_answer=current_answer,
                mode="revision"
            )

            action = "revision"

        # ----------------------------------------------------
        # Evaluate candidate
        # ----------------------------------------------------

        candidate_result = evaluate_answer(
            candidate_answer,
            iteration=revision_number,
            action=action
        )

        # ----------------------------------------------------
        # Compare candidate with accepted answer
        # ----------------------------------------------------

        (
            accepted,
            reason,
            change
        ) = accept_revision(
            accepted_result,
            candidate_result
        )

        candidate_record = {
            **candidate_result,
            "accepted": accepted,
            "acceptance_reason": reason,
            "changes_vs_previous": change
        }

        trajectory.append(
            candidate_record
        )

        print(
            f"\nRevision decision: "
            f"{'ACCEPT' if accepted else 'REJECT / ROLLBACK'}"
        )

        print(
            f"Reason: {reason}"
        )

        print(
            f"Confidence change: "
            f"{change['confidence_change']:+.4f}"
        )

        print(
            f"Entropy change: "
            f"{change['entropy_change']:+.4f}"
        )

        # ----------------------------------------------------
        # Accept improved candidate
        # ----------------------------------------------------

        if accepted:

            accepted_result = candidate_result
            current_answer = candidate_answer

            print(
                "Candidate accepted."
            )

        # ----------------------------------------------------
        # Reject candidate
        # ----------------------------------------------------

        else:

            print(
                "Candidate rejected."
            )

            print(
                "Rolling back to previous accepted answer."
            )

            break

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    final_result = accepted_result

    initial = initial_result

    confidence_change = (
        final_result["confidence"]
        - initial["confidence"]
    )

    entropy_change = (
        final_result["entropy"]
        - initial["entropy"]
    )

    label_order = {
        "UNRELIABLE": 0,
        "UNCERTAIN": 1,
        "RELIABLE": 2
    }

    initial_level = label_order[
        initial["trust_label"]
    ]

    final_level = label_order[
        final_result["trust_label"]
    ]

    if final_level > initial_level:
        overall_status = "IMPROVED"

    elif final_level < initial_level:
        overall_status = "WORSENED"

    elif (
        confidence_change > 0
        and entropy_change < 0
    ):
        overall_status = "REPRESENTATION_IMPROVED"

    elif (
        confidence_change < 0
        and entropy_change > 0
    ):
        overall_status = "REPRESENTATION_WORSENED"

    else:
        overall_status = "NO_CLEAR_CHANGE"

    print("\nFinal result:")

    print(
        "  Initial class:",
        initial["trust_label"]
    )

    print(
        "  Final class:",
        final_result["trust_label"]
    )

    print(
        f"  Confidence change: "
        f"{confidence_change:+.4f}"
    )

    print(
        f"  Entropy change: "
        f"{entropy_change:+.4f}"
    )

    print(
        "  Overall status:",
        overall_status
    )

    all_results.append({
        "question_id": question_index,
        "question": question,
        "initial": initial_result,
        "final": final_result,
        "trajectory": trajectory,
        "overall_status": overall_status,
        "confidence_change": confidence_change,
        "entropy_change": entropy_change,
        "initial_label": initial["trust_label"],
        "final_label": final_result["trust_label"]
    })


# ============================================================
# SUMMARY
# ============================================================

improved = sum(
    r["overall_status"] == "IMPROVED"
    for r in all_results
)

representation_improved = sum(
    r["overall_status"] == "REPRESENTATION_IMPROVED"
    for r in all_results
)

worsened = sum(
    r["overall_status"] == "WORSENED"
    for r in all_results
)

representation_worsened = sum(
    r["overall_status"] == "REPRESENTATION_WORSENED"
    for r in all_results
)

no_change = sum(
    r["overall_status"] == "NO_CLEAR_CHANGE"
    for r in all_results
)


output = {
    "project": "TrustGate",
    "experiment": "Self-Improvement V3",
    "base_model": BASE_MODEL,
    "representation": {
        "hidden_dimension": 896,
        "pca_dimension": 20,
        "confidence_dimension": 1,
        "entropy_dimension": 1,
        "total_dimension": 22
    },
    "classifier": {
        "type": "LinearSVC",
        "class_weight": "balanced"
    },
    "revision_policy": {
        "max_revision_attempts": MAX_REVISION_ATTEMPTS,
        "rollback_enabled": True,
        "margin_used_as_quality_score": False
    },
    "summary": {
        "questions": len(all_results),
        "class_improved": improved,
        "representation_improved": representation_improved,
        "class_worsened": worsened,
        "representation_worsened": representation_worsened,
        "no_clear_change": no_change
    },
    "results": all_results
}


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
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
print("SELF-IMPROVEMENT V3 SUMMARY")
print("=" * 70)

print(
    "Total questions:",
    len(all_results)
)

print(
    "Class improved:",
    improved
)

print(
    "Representation improved:",
    representation_improved
)

print(
    "Class worsened:",
    worsened
)

print(
    "Representation worsened:",
    representation_worsened
)

print(
    "No clear change:",
    no_change
)

print("\nPer-question results:")

for result in all_results:

    print(
        f"\n{result['question']}"
    )

    print(
        f"  {result['initial_label']} "
        f"-> "
        f"{result['final_label']}"
    )

    print(
        f"  Status: "
        f"{result['overall_status']}"
    )

    print(
        f"  Confidence Δ: "
        f"{result['confidence_change']:+.4f}"
    )

    print(
        f"  Entropy Δ: "
        f"{result['entropy_change']:+.4f}"
    )


print("\n")
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(OUTPUT_FILE)