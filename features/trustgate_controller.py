import os
import joblib
import numpy as np


MODEL_PATH = "models/trustgate_500_22d_linear_svm.joblib"

LABELS = [
    "RELIABLE",
    "UNCERTAIN",
    "UNRELIABLE"
]


class TrustGateController:

    def __init__(self, model_path=MODEL_PATH):

        self.model_path = model_path

        print("[TrustGate] Loading 22-D model:")
        print(f"            {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"TrustGate model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

        required_keys = [
            "hidden_scaler",
            "pca",
            "final_scaler",
            "classifier"
        ]

        for key in required_keys:
            if key not in self.model:
                raise ValueError(
                    f"Model bundle is missing required component: {key}"
                )

        self.hidden_scaler = self.model["hidden_scaler"]
        self.pca = self.model["pca"]
        self.final_scaler = self.model["final_scaler"]
        self.classifier = self.model["classifier"]

        self.input_hidden_dimension = int(
            self.model["input_hidden_dimension"]
        )

        self.pca_dimension = int(
            self.model["pca_dimension"]
        )

        self.final_dimension = int(
            self.model["final_dimension"]
        )

        self.signal_features = self.model[
            "signal_features"
        ]

        print("[TrustGate] Model loaded successfully.")
        print(
            f"[TrustGate] Hidden input: "
            f"{self.input_hidden_dimension}D"
        )
        print(
            f"[TrustGate] PCA representation: "
            f"{self.pca_dimension}D"
        )
        print(
            f"[TrustGate] Final representation: "
            f"{self.final_dimension}D"
        )

    # ---------------------------------------------------------
    # TRUST REPRESENTATION
    # ---------------------------------------------------------

    def build_trust_representation(
        self,
        hidden_state,
        confidence,
        entropy
    ):
        """
        Convert:

            896-D hidden state
                    ↓
            StandardScaler
                    ↓
                PCA(20)
                    ↓
             20-D hidden
                    +
              confidence
                    +
                entropy
                    ↓
             22-D representation
        """

        hidden_state = np.asarray(
            hidden_state,
            dtype=np.float32
        ).reshape(1, -1)

        if hidden_state.shape[1] != self.input_hidden_dimension:
            raise ValueError(
                f"Expected hidden state with "
                f"{self.input_hidden_dimension} dimensions, "
                f"got {hidden_state.shape[1]}"
            )

        confidence = float(confidence)
        entropy = float(entropy)

        # 896-D → standardized hidden
        hidden_scaled = self.hidden_scaler.transform(
            hidden_state
        )

        # standardized hidden → PCA 20-D
        hidden_pca = self.pca.transform(
            hidden_scaled
        )

        # 20-D + confidence + entropy = 22-D
        representation = np.hstack([
            hidden_pca,
            np.array(
                [[confidence, entropy]],
                dtype=np.float32
            )
        ])

        if representation.shape[1] != self.final_dimension:
            raise ValueError(
                f"Expected final representation with "
                f"{self.final_dimension} dimensions, "
                f"got {representation.shape[1]}"
            )

        return representation

    # ---------------------------------------------------------
    # CLASSIFICATION
    # ---------------------------------------------------------

    def classify(
        self,
        hidden_state,
        confidence,
        entropy
    ):
        """
        Generate the TrustGate prediction for one answer.
        """

        representation = self.build_trust_representation(
            hidden_state,
            confidence,
            entropy
        )

        # Final scaling before LinearSVC
        representation_scaled = (
            self.final_scaler.transform(
                representation
            )
        )

        prediction = self.classifier.predict(
            representation_scaled
        )[0]

        return str(prediction)

    # ---------------------------------------------------------
    # TRUST PRIORITY
    # ---------------------------------------------------------

    @staticmethod
    def trust_priority(label):
        """
        Higher number = more trusted class.

        RELIABLE   = 2
        UNCERTAIN  = 1
        UNRELIABLE = 0
        """

        priorities = {
            "RELIABLE": 2,
            "UNCERTAIN": 1,
            "UNRELIABLE": 0
        }

        if label not in priorities:
            raise ValueError(
                f"Unknown TrustGate label: {label}"
            )

        return priorities[label]

    # ---------------------------------------------------------
    # CLASS COMPARISON
    # ---------------------------------------------------------

    def is_better_class(
        self,
        old_label,
        new_label
    ):
        """
        True when the candidate moved to a
        more trusted TrustGate class.
        """

        return (
            self.trust_priority(new_label)
            >
            self.trust_priority(old_label)
        )

    # ---------------------------------------------------------
    # SAME CLASS SIGNAL CHECK
    # ---------------------------------------------------------

    @staticmethod
    def same_class_better_signals(
        old_confidence,
        old_entropy,
        new_confidence,
        new_entropy
    ):
        """
        Accept same-class candidates only when:

        confidence increases
        AND
        entropy decreases
        """

        confidence_improved = (
            new_confidence > old_confidence
        )

        entropy_improved = (
            new_entropy < old_entropy
        )

        return (
            confidence_improved
            and entropy_improved
        )

    # ---------------------------------------------------------
    # CANDIDATE DECISION
    # ---------------------------------------------------------

    def evaluate_candidate(
        self,
        old_label,
        new_label,
        old_confidence,
        old_entropy,
        new_confidence,
        new_entropy,
        candidate_contradicted=False,
        evidence_supports_candidate=False
    ):
        """
        Decide whether a refined answer should be accepted.

        Decision hierarchy:

        1. Contradicted candidate
              → ROLLBACK

        2. Better TrustGate class
              → ACCEPT

        3. Same class with:
              confidence ↑
              entropy ↓
              → ACCEPT

        4. Evidence supports candidate and
           candidate is not worse
              → ACCEPT

        5. Otherwise
              → ROLLBACK
        """

        # Rule 1: external evidence contradiction
        if candidate_contradicted:

            return {
                "decision": "ROLLBACK",
                "reason": "EXTERNAL_CONTRADICTION"
            }

        # Rule 2: TrustGate class improved
        if self.is_better_class(
            old_label,
            new_label
        ):

            return {
                "decision": "ACCEPT",
                "reason": "TRUST_CLASS_IMPROVED"
            }

        # Rule 3: same class but signals improved
        if (
            old_label == new_label
            and
            self.same_class_better_signals(
                old_confidence,
                old_entropy,
                new_confidence,
                new_entropy
            )
        ):

            return {
                "decision": "ACCEPT",
                "reason": "SAME_CLASS_SIGNALS_IMPROVED"
            }

        # Rule 4: evidence supports candidate
        if evidence_supports_candidate:

            candidate_not_worse = (
                self.trust_priority(new_label)
                >=
                self.trust_priority(old_label)
            )

            if candidate_not_worse:

                return {
                    "decision": "ACCEPT",
                    "reason": "EVIDENCE_SUPPORTS_CANDIDATE"
                }

        # Rule 5: rollback
        return {
            "decision": "ROLLBACK",
            "reason": "NO_TRUST_IMPROVEMENT"
        }


# =============================================================
# SELF TEST
# =============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("TRUSTGATE CONTROLLER TEST")
    print("=" * 60)

    controller = TrustGateController()

    # ---------------------------------------------------------
    # Create deterministic dummy hidden representation
    # ---------------------------------------------------------

    test_hidden = np.zeros(
        896,
        dtype=np.float32
    )

    test_confidence = 0.50
    test_entropy = 1.50

    print()
    print("[TEST 1] Building 22-D representation")

    representation = (
        controller.build_trust_representation(
            test_hidden,
            test_confidence,
            test_entropy
        )
    )

    print(
        "Representation shape:",
        representation.shape
    )

    assert representation.shape == (1, 22)

    print("[PASS] 22-D representation created.")

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    print()
    print("[TEST 2] TrustGate classification")

    prediction = controller.classify(
        test_hidden,
        test_confidence,
        test_entropy
    )

    print(
        "Prediction:",
        prediction
    )

    assert prediction in LABELS

    print("[PASS] Valid TrustGate class.")

    # ---------------------------------------------------------
    # Priority tests
    # ---------------------------------------------------------

    print()
    print("[TEST 3] Trust priorities")

    assert controller.trust_priority(
        "RELIABLE"
    ) == 2

    assert controller.trust_priority(
        "UNCERTAIN"
    ) == 1

    assert controller.trust_priority(
        "UNRELIABLE"
    ) == 0

    print("[PASS] Trust priorities correct.")

    # ---------------------------------------------------------
    # Class improvement
    # ---------------------------------------------------------

    print()
    print("[TEST 4] Class comparison")

    assert controller.is_better_class(
        "UNRELIABLE",
        "UNCERTAIN"
    )

    assert controller.is_better_class(
        "UNCERTAIN",
        "RELIABLE"
    )

    assert not controller.is_better_class(
        "RELIABLE",
        "UNCERTAIN"
    )

    print("[PASS] Class comparison works.")

    # ---------------------------------------------------------
    # Candidate decision
    # ---------------------------------------------------------

    print()
    print("[TEST 5] Candidate evaluation")

    result = controller.evaluate_candidate(
        old_label="UNRELIABLE",
        new_label="UNCERTAIN",
        old_confidence=0.40,
        old_entropy=2.00,
        new_confidence=0.45,
        new_entropy=1.80
    )

    print(
        "Improved candidate:",
        result
    )

    assert result["decision"] == "ACCEPT"

    result = controller.evaluate_candidate(
        old_label="UNCERTAIN",
        new_label="UNRELIABLE",
        old_confidence=0.50,
        old_entropy=1.50,
        new_confidence=0.60,
        new_entropy=1.20
    )

    print(
        "Worse-class candidate:",
        result
    )

    assert result["decision"] == "ROLLBACK"

    result = controller.evaluate_candidate(
        old_label="UNCERTAIN",
        new_label="RELIABLE",
        old_confidence=0.50,
        old_entropy=1.50,
        new_confidence=0.40,
        new_entropy=1.80,
        candidate_contradicted=True
    )

    print(
        "Contradicted candidate:",
        result
    )

    assert result["decision"] == "ROLLBACK"

    print()
    print("=" * 60)
    print("ALL TRUSTGATE CONTROLLER TESTS PASSED")
    print("=" * 60)