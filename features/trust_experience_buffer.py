import json
import os
from datetime import datetime, timezone


DEFAULT_BUFFER_PATH = (
    "outputs/trust_experience_buffer.json"
)


class TrustExperienceBuffer:
    """
    Stores high-quality TrustGate self-improvement
    experiences.

    The buffer is intentionally separate from model
    training. V6 first collects experiences; a later
    module can use approved experiences for TrustGate
    retraining.

    An experience should contain:

        question
        initial_answer
        initial_label
        initial_confidence
        initial_entropy

        refined_answer
        final_label
        final_confidence
        final_entropy

        verification_status
        evidence
        decision

    Only experiences passing the quality gate are
    added to the approved buffer.
    """

    APPROVED_STATUSES = {
        "VERIFIED",
        "SUPPORTED",
        "EVIDENCE_CONFIRMED"
    }

    APPROVED_DECISIONS = {
        "ACCEPT",
        "TRUST_CLASS_IMPROVED",
        "EVIDENCE_SUPPORTS_CANDIDATE"
    }

    TRUST_PRIORITY = {
        "RELIABLE": 2,
        "UNCERTAIN": 1,
        "UNRELIABLE": 0
    }

    def __init__(
        self,
        buffer_path=DEFAULT_BUFFER_PATH
    ):
        self.buffer_path = buffer_path

        self.experiences = []

        self.load()

    # =========================================================
    # LOAD
    # =========================================================

    def load(self):
        """
        Load existing approved experiences.

        If the file does not exist, start with an
        empty buffer.
        """

        if not os.path.exists(
            self.buffer_path
        ):
            self.experiences = []
            return

        try:

            with open(
                self.buffer_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if not isinstance(
                data,
                list
            ):
                raise ValueError(
                    "Buffer file must contain a JSON list."
                )

            self.experiences = data

        except Exception as exc:

            raise RuntimeError(
                "Could not load Trust Experience Buffer: "
                f"{exc}"
            ) from exc

    # =========================================================
    # SAVE
    # =========================================================

    def save(self):
        """
        Atomically save the approved experience buffer.
        """

        directory = os.path.dirname(
            self.buffer_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        temp_path = (
            self.buffer_path
            + ".tmp"
        )

        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.experiences,
                f,
                indent=2,
                ensure_ascii=False
            )

        os.replace(
            temp_path,
            self.buffer_path
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _required_fields():
        return [
            "question",
            "initial_answer",
            "initial_label",
            "initial_confidence",
            "initial_entropy",
            "refined_answer",
            "final_label",
            "final_confidence",
            "final_entropy",
            "verification_status",
            "evidence",
            "decision"
        ]

    def validate_experience(
        self,
        experience
    ):
        """
        Validate the structure of an experience.
        """

        if not isinstance(
            experience,
            dict
        ):
            return False, "Experience must be a dictionary."

        missing = [
            field
            for field in self._required_fields()
            if field not in experience
        ]

        if missing:
            return (
                False,
                "Missing fields: "
                + ", ".join(missing)
            )

        labels = {
            experience["initial_label"],
            experience["final_label"]
        }

        for label in labels:

            if label not in self.TRUST_PRIORITY:
                return (
                    False,
                    f"Invalid TrustGate label: {label}"
                )

        if not isinstance(
            experience["evidence"],
            list
        ):
            return (
                False,
                "Evidence must be a list."
            )

        try:

            float(
                experience["initial_confidence"]
            )

            float(
                experience["initial_entropy"]
            )

            float(
                experience["final_confidence"]
            )

            float(
                experience["final_entropy"]
            )

        except (
            TypeError,
            ValueError
        ):

            return (
                False,
                "Confidence and entropy must be numeric."
            )

        return True, "VALID"

    # =========================================================
    # QUALITY GATE
    # =========================================================

    def quality_gate(
        self,
        experience
    ):
        """
        Decide whether an experience is strong enough
        to enter the approved self-improvement buffer.

        Requirements:

            1. Valid structure
            2. Verification status is approved
            3. Decision represents acceptance
            4. Evidence exists
            5. Final answer is not worse than initial
               according to TrustGate class

        This is deliberately conservative.
        """

        valid, reason = (
            self.validate_experience(
                experience
            )
        )

        if not valid:

            return {
                "approved": False,
                "reason": reason
            }

        verification_status = (
            str(
                experience[
                    "verification_status"
                ]
            ).upper()
        )

        if (
            verification_status
            not in self.APPROVED_STATUSES
        ):

            return {
                "approved": False,
                "reason": (
                    "VERIFICATION_NOT_STRONG_ENOUGH"
                )
            }

        decision = (
            str(
                experience["decision"]
            ).upper()
        )

        if (
            decision
            not in self.APPROVED_DECISIONS
        ):

            return {
                "approved": False,
                "reason": (
                    "DECISION_NOT_ACCEPTED"
                )
            }

        evidence = experience["evidence"]

        if len(evidence) == 0:

            return {
                "approved": False,
                "reason": "NO_EVIDENCE"
            }

        initial_label = (
            experience["initial_label"]
        )

        final_label = (
            experience["final_label"]
        )

        initial_priority = (
            self.TRUST_PRIORITY[
                initial_label
            ]
        )

        final_priority = (
            self.TRUST_PRIORITY[
                final_label
            ]
        )

        if final_priority < initial_priority:

            return {
                "approved": False,
                "reason": (
                    "FINAL_TRUST_CLASS_WORSE"
                )
            }

        return {
            "approved": True,
            "reason": "QUALITY_GATE_PASSED"
        }

    # =========================================================
    # ADD EXPERIENCE
    # =========================================================

    def add_experience(
        self,
        experience
    ):
        """
        Add an experience only if it passes the
        quality gate.
        """

        gate_result = self.quality_gate(
            experience
        )

        if not gate_result["approved"]:

            return {
                "added": False,
                "reason": gate_result["reason"],
                "size": len(
                    self.experiences
                )
            }

        experience = dict(
            experience
        )

        experience[
            "experience_id"
        ] = (
            len(self.experiences)
        )

        experience[
            "created_at"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        experience[
            "quality_status"
        ] = "APPROVED"

        self.experiences.append(
            experience
        )

        self.save()

        return {
            "added": True,
            "reason": "QUALITY_GATE_PASSED",
            "experience_id":
                experience["experience_id"],
            "size":
                len(self.experiences)
        }

    # =========================================================
    # BUFFER SIZE
    # =========================================================

    def size(self):
        return len(
            self.experiences
        )

    # =========================================================
    # GET EXPERIENCES
    # =========================================================

    def get_all(self):
        """
        Return all approved experiences.
        """

        return list(
            self.experiences
        )

    # =========================================================
    # EXPORT TRAINING DATA
    # =========================================================

    def get_training_candidates(self):
        """
        Return only approved experiences that can later
        be considered for TrustGate training.

        This method does NOT perform training.
        """

        candidates = []

        for experience in self.experiences:

            if (
                experience.get(
                    "quality_status"
                )
                != "APPROVED"
            ):
                continue

            candidates.append(
                experience
            )

        return candidates


# =============================================================
# SELF TEST
# =============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("TRUST EXPERIENCE BUFFER TEST")
    print("=" * 60)

    test_buffer_path = (
        "outputs/test_trust_experience_buffer.json"
    )

    # ---------------------------------------------------------
    # Create fresh test buffer
    # ---------------------------------------------------------

    if os.path.exists(
        test_buffer_path
    ):
        os.remove(
            test_buffer_path
        )

    buffer = TrustExperienceBuffer(
        buffer_path=test_buffer_path
    )

    print()
    print("[TEST 1] Empty buffer")

    print(
        "Initial size:",
        buffer.size()
    )

    assert buffer.size() == 0

    print(
        "[PASS] Empty buffer created."
    )

    # ---------------------------------------------------------
    # Good experience
    # ---------------------------------------------------------

    print()
    print("[TEST 2] High-quality experience")

    good_experience = {

        "question":
            "What is the capital of France?",

        "initial_answer":
            "The capital is Lyon.",

        "initial_label":
            "UNRELIABLE",

        "initial_confidence":
            0.42,

        "initial_entropy":
            1.80,

        "refined_answer":
            "The capital of France is Paris.",

        "final_label":
            "RELIABLE",

        "final_confidence":
            0.76,

        "final_entropy":
            0.90,

        "verification_status":
            "SUPPORTED",

        "evidence": [
            {
                "source":
                    "Wikipedia",
                "title":
                    "Paris"
            }
        ],

        "decision":
            "ACCEPT"
    }

    result = buffer.add_experience(
        good_experience
    )

    print(
        "Result:",
        result
    )

    assert result["added"] is True
    assert buffer.size() == 1

    print(
        "[PASS] High-quality experience accepted."
    )

    # ---------------------------------------------------------
    # Bad verification
    # ---------------------------------------------------------

    print()
    print("[TEST 3] Weak verification")

    weak_experience = {

        "question":
            "Example question",

        "initial_answer":
            "Example answer",

        "initial_label":
            "UNRELIABLE",

        "initial_confidence":
            0.40,

        "initial_entropy":
            2.00,

        "refined_answer":
            "Another answer",

        "final_label":
            "RELIABLE",

        "final_confidence":
            0.60,

        "final_entropy":
            1.00,

        "verification_status":
            "EVIDENCE_FOUND",

        "evidence": [
            {
                "source":
                    "Wikipedia"
            }
        ],

        "decision":
            "ACCEPT"
    }

    result = buffer.add_experience(
        weak_experience
    )

    print(
        "Result:",
        result
    )

    assert result["added"] is False

    print(
        "[PASS] Weak verification rejected."
    )

    # ---------------------------------------------------------
    # Worse final class
    # ---------------------------------------------------------

    print()
    print("[TEST 4] Worse final TrustGate class")

    worse_experience = {

        "question":
            "Example question 2",

        "initial_answer":
            "Initial answer",

        "initial_label":
            "RELIABLE",

        "initial_confidence":
            0.80,

        "initial_entropy":
            0.80,

        "refined_answer":
            "Worse answer",

        "final_label":
            "UNRELIABLE",

        "final_confidence":
            0.30,

        "final_entropy":
            2.00,

        "verification_status":
            "SUPPORTED",

        "evidence": [
            {
                "source":
                    "Wikipedia"
            }
        ],

        "decision":
            "ACCEPT"
    }

    result = buffer.add_experience(
        worse_experience
    )

    print(
        "Result:",
        result
    )

    assert result["added"] is False

    print(
        "[PASS] Worse candidate rejected."
    )

    # ---------------------------------------------------------
    # Training candidates
    # ---------------------------------------------------------

    print()
    print("[TEST 5] Training candidate retrieval")

    candidates = (
        buffer.get_training_candidates()
    )

    print(
        "Training candidates:",
        len(candidates)
    )

    assert len(candidates) == 1

    print(
        "[PASS] Only approved experience returned."
    )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    if os.path.exists(
        test_buffer_path
    ):
        os.remove(
            test_buffer_path
        )

    print()
    print("=" * 60)
    print("ALL TRUST EXPERIENCE BUFFER TESTS PASSED")
    print("=" * 60)