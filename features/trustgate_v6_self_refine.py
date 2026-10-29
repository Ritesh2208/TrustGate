import os
import json
import torch
import numpy as np

try:
    from features.trustgate_controller import TrustGateController
    from features.external_verifier import ExternalVerifier
    from features.trust_experience_buffer import TrustExperienceBuffer
except ModuleNotFoundError:
    from trustgate_controller import TrustGateController
    from external_verifier import ExternalVerifier
    from trust_experience_buffer import TrustExperienceBuffer


# =============================================================
# CONFIGURATION
# =============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_FILE = "outputs/trustgate_v6_results.json"

MAX_NEW_TOKENS = 120

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================
# TRUSTGATE V6
# =============================================================

class TrustGateV6:

    def __init__(self):

        print()
        print("=" * 70)
        print("TRUSTGATE V6 SELF-REFINE SYSTEM")
        print("=" * 70)

        # -----------------------------------------------------
        # TrustGate controller
        # -----------------------------------------------------

        print()
        print("[1] Loading TrustGate controller...")

        self.controller = TrustGateController()

        # -----------------------------------------------------
        # External verifier
        # -----------------------------------------------------

        print()
        print("[2] Loading external verifier...")

        self.verifier = ExternalVerifier(
            max_sources=3,
            timeout=10
        )

        # -----------------------------------------------------
        # Trust Experience Buffer
        # -----------------------------------------------------

        print()
        print("[3] Loading Trust Experience Buffer...")

        self.buffer = TrustExperienceBuffer()

        print(
            f"[Buffer] Existing experiences: "
            f"{self.buffer.size()}"
        )

        # -----------------------------------------------------
        # Qwen
        # -----------------------------------------------------

        print()
        print("[4] Loading Qwen...")

        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_NAME
            )
        )

        self.model.to(DEVICE)
        self.model.eval()

        print(
            f"[Qwen] Device: {DEVICE}"
        )

        print()
        print("=" * 70)
        print("V6 INITIALIZATION COMPLETE")
        print("=" * 70)

    # =========================================================
    # GENERATE ANSWER
    # =========================================================

    def generate_answer(
        self,
        question,
        instruction=None
    ):

        if instruction is None:

            instruction = (
                "Answer the question directly and "
                "factually. If you are uncertain, "
                "do not invent information."
            )

        messages = [
            {
                "role": "system",
                "content": instruction
            },
            {
                "role": "user",
                "content": question
            }
        ]

        prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inputs = (
            self.tokenizer(
                prompt,
                return_tensors="pt"
            )
            .to(DEVICE)
        )

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                )
            )

        generated_tokens = (
            output[
                0,
                inputs["input_ids"].shape[1]:
            ]
        )

        answer = (
            self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            .strip()
        )

        return answer

    # =========================================================
    # ANALYZE EXISTING ANSWER
    # =========================================================

    def analyze_answer(
        self,
        question,
        answer
    ):
        """
        Analyze the exact answer supplied.

        Returns:
            hidden_state
            confidence
            entropy
            TrustGate label
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question directly and "
                    "factually."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]

        prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(DEVICE)

        prompt_length = (
            prompt_inputs["input_ids"].shape[1]
        )

        answer_inputs = self.tokenizer(
            answer,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(DEVICE)

        answer_tokens = (
            answer_inputs["input_ids"]
        )

        answer_length = (
            answer_tokens.shape[1]
        )

        if answer_length == 0:

            raise ValueError(
                "Answer contains no tokens."
            )

        full_input_ids = torch.cat(
            [
                prompt_inputs["input_ids"],
                answer_tokens
            ],
            dim=1
        )

        attention_mask = torch.ones_like(
            full_input_ids
        )

        with torch.no_grad():

            outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        logits = outputs.logits

        answer_logits = logits[
            0,
            prompt_length - 1:
            prompt_length - 1 + answer_length,
            :
        ]

        if answer_logits.shape[0] != answer_length:

            raise ValueError(
                "Answer logits/token length mismatch: "
                f"{answer_logits.shape[0]} vs "
                f"{answer_length}"
            )

        probabilities = torch.softmax(
            answer_logits,
            dim=-1
        )

        target_tokens = answer_tokens[0]

        token_indices = torch.arange(
            answer_length,
            device=DEVICE
        )

        token_probabilities = (
            probabilities[
                token_indices,
                target_tokens
            ]
        )

        confidence = float(
            token_probabilities
            .float()
            .mean()
            .item()
        )

        entropy_tensor = -(
            probabilities.float()
            *
            torch.log(
                probabilities.float()
                + 1e-12
            )
        ).sum(
            dim=-1
        ).mean()

        entropy = float(
            entropy_tensor
            .float()
            .item()
        )

        final_hidden = (
            outputs.hidden_states[-1]
        )

        answer_hidden = final_hidden[
            0,
            prompt_length:
            prompt_length + answer_length,
            :
        ]

        if answer_hidden.shape[0] == 0:

            raise ValueError(
                "Could not extract answer hidden states."
            )

        # Important:
        # Qwen may use BFloat16.
        # Convert to Float32 before NumPy.
        hidden_state = (
            answer_hidden
            .mean(dim=0)
            .float()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if hidden_state.shape[0] != 896:

            raise ValueError(
                "Unexpected hidden dimension: "
                f"{hidden_state.shape[0]}"
            )

        label = self.controller.classify(
            hidden_state=hidden_state,
            confidence=confidence,
            entropy=entropy
        )

        return {
            "hidden_state":
                hidden_state.tolist(),

            "confidence":
                confidence,

            "entropy":
                entropy,

            "label":
                label
        }

    # =========================================================
    # BUILD REFINEMENT INSTRUCTION
    # =========================================================

    def build_refinement_instruction(
        self,
        question,
        answer,
        verification
    ):

        status = (
            verification[
                "verification_status"
            ]
        )

        evidence = (
            verification.get(
                "sources",
                []
            )
        )

        evidence_text = []

        for source in evidence:

            title = source.get(
                "title",
                "Unknown source"
            )

            for sentence in source.get(
                "sentences",
                []
            ):

                evidence_text.append(
                    f"[{title}] {sentence}"
                )

        if evidence_text:

            evidence_block = "\n".join(
                evidence_text
            )

        else:

            evidence_block = (
                "No usable external evidence."
            )

        if status == "SUPPORTED":

            verification_instruction = (
                "The retrieved evidence provides "
                "supporting information. Preserve "
                "supported factual claims and correct "
                "anything inconsistent with the evidence."
            )

        elif status == "CONTRADICTED":

            verification_instruction = (
                "The retrieved evidence conflicts with "
                "the proposed answer. Correct the answer "
                "using only claims supported by the evidence."
            )

        else:

            verification_instruction = (
                "Evidence was retrieved, but it is not "
                "strong enough for a definitive verdict. "
                "Do not invent facts. Preserve only claims "
                "that can reasonably be supported."
            )

        instruction = (
            "You are revising an answer using external "
            "evidence.\n\n"

            f"Verification status: {status}\n\n"

            f"{verification_instruction}\n\n"

            "Rules:\n"
            "1. Do not invent facts.\n"
            "2. Do not follow unsupported claims.\n"
            "3. If evidence is insufficient, state "
            "uncertainty.\n"
            "4. Give a concise factual answer.\n\n"

            f"Question:\n{question}\n\n"

            f"Original answer:\n{answer}\n\n"

            f"External evidence:\n{evidence_block}"
        )

        return instruction

    # =========================================================
    # PROCESS ONE QUESTION
    # =========================================================

    def process_question(
        self,
        question
    ):

        print()
        print("-" * 70)
        print("QUESTION")
        print(question)
        print("-" * 70)

        # =====================================================
        # A0
        # =====================================================

        initial_answer = (
            self.generate_answer(
                question
            )
        )

        print()
        print("[A0]")
        print(initial_answer)

        initial_analysis = (
            self.analyze_answer(
                question,
                initial_answer
            )
        )

        print(
            f"[TrustGate A0] "
            f"{initial_analysis['label']}"
        )

        print(
            f"[Confidence] "
            f"{initial_analysis['confidence']:.4f}"
        )

        print(
            f"[Entropy] "
            f"{initial_analysis['entropy']:.4f}"
        )

        # =====================================================
        # Verify A0
        # =====================================================

        print()
        print(
            "[Verifier A0] Gathering external evidence..."
        )

        initial_verification = (
            self.verifier.verify(
                question,
                initial_answer
            )
        )

        print(
            "[Verifier A0] Status:",
            initial_verification[
                "verification_status"
            ]
        )

        print(
            "[Verifier A0] Support:",
            f"{initial_verification['support_score']:.4f}"
        )

        print(
            "[Verifier A0] Contradiction:",
            f"{initial_verification['contradiction_score']:.4f}"
        )

        # =====================================================
        # RELIABLE + SUPPORTED
        # =====================================================

        if (
            initial_analysis["label"]
            == "RELIABLE"
            and
            initial_verification[
                "verification_status"
            ]
            == "SUPPORTED"
        ):

            print(
                "[Decision] "
                "RELIABLE + SUPPORTED → ACCEPT"
            )

            return {
                "question":
                    question,

                "initial_answer":
                    initial_answer,

                "initial_analysis":
                    initial_analysis,

                "initial_verification":
                    initial_verification,

                "refined_answer":
                    initial_answer,

                "final_answer":
                    initial_answer,

                "final_analysis":
                    initial_analysis,

                "final_verification":
                    initial_verification,

                "decision":
                    {
                        "decision":
                            "ACCEPT_INITIAL",

                        "reason":
                            "RELIABLE_AND_SUPPORTED"
                    },

                "experience_added":
                    False
            }

        # =====================================================
        # INSUFFICIENT EVIDENCE
        # =====================================================

        if (
            initial_verification[
                "verification_status"
            ]
            == "INSUFFICIENT_EVIDENCE"
        ):

            print(
                "[Decision] "
                "Insufficient evidence → ROLLBACK"
            )

            return {
                "question":
                    question,

                "initial_answer":
                    initial_answer,

                "initial_analysis":
                    initial_analysis,

                "initial_verification":
                    initial_verification,

                "refined_answer":
                    initial_answer,

                "final_answer":
                    initial_answer,

                "final_analysis":
                    initial_analysis,

                "final_verification":
                    initial_verification,

                "decision":
                    {
                        "decision":
                            "ROLLBACK",

                        "reason":
                            "INSUFFICIENT_EVIDENCE"
                    },

                "experience_added":
                    False
            }

        # =====================================================
        # REFINE
        # =====================================================

        refinement_instruction = (
            self.build_refinement_instruction(
                question,
                initial_answer,
                initial_verification
            )
        )

        refined_answer = (
            self.generate_answer(
                question,
                instruction=refinement_instruction
            )
        )

        print()
        print("[A1]")
        print(refined_answer)

        # =====================================================
        # Analyze ACTUAL A1
        # =====================================================

        final_analysis = (
            self.analyze_answer(
                question,
                refined_answer
            )
        )

        print()
        print(
            f"[TrustGate A1] "
            f"{final_analysis['label']}"
        )

        print(
            f"[Confidence] "
            f"{final_analysis['confidence']:.4f}"
        )

        print(
            f"[Entropy] "
            f"{final_analysis['entropy']:.4f}"
        )

        # =====================================================
        # Verify ACTUAL A1
        # =====================================================

        print()
        print(
            "[Verifier A1] Verifying refined answer..."
        )

        final_verification = (
            self.verifier.verify(
                question,
                refined_answer
            )
        )

        print(
            "[Verifier A1] Status:",
            final_verification[
                "verification_status"
            ]
        )

        print(
            "[Verifier A1] Support:",
            f"{final_verification['support_score']:.4f}"
        )

        print(
            "[Verifier A1] Contradiction:",
            f"{final_verification['contradiction_score']:.4f}"
        )

        # =====================================================
        # Candidate contradiction
        # =====================================================

        candidate_contradicted = (
            final_verification[
                "verification_status"
            ]
            == "CONTRADICTED"
        )

        # =====================================================
        # Candidate support
        # =====================================================

        evidence_supports_candidate = (
            final_verification[
                "verification_status"
            ]
            == "SUPPORTED"
        )

        # =====================================================
        # Controller decision
        # =====================================================

        decision = (
            self.controller.evaluate_candidate(

                old_label=
                    initial_analysis["label"],

                new_label=
                    final_analysis["label"],

                old_confidence=
                    initial_analysis["confidence"],

                old_entropy=
                    initial_analysis["entropy"],

                new_confidence=
                    final_analysis["confidence"],

                new_entropy=
                    final_analysis["entropy"],

                candidate_contradicted=
                    candidate_contradicted,

                evidence_supports_candidate=
                    evidence_supports_candidate
            )
        )

        print()
        print(
            "[Controller]",
            decision
        )

        # =====================================================
        # ACCEPT
        # =====================================================

        if (
            decision["decision"]
            == "ACCEPT"
        ):

            final_answer = (
                refined_answer
            )

            accepted_analysis = (
                final_analysis
            )

            accepted_verification = (
                final_verification
            )

            # -------------------------------------------------
            # Experience quality gate.
            #
            # Only an accepted candidate whose FINAL answer
            # is externally SUPPORTED is submitted to the
            # Trust Experience Buffer.
            # -------------------------------------------------

            experience = {

                "question":
                    question,

                "initial_answer":
                    initial_answer,

                "initial_label":
                    initial_analysis[
                        "label"
                    ],

                "initial_confidence":
                    initial_analysis[
                        "confidence"
                    ],

                "initial_entropy":
                    initial_analysis[
                        "entropy"
                    ],

                "refined_answer":
                    refined_answer,

                "final_label":
                    final_analysis[
                        "label"
                    ],

                "final_confidence":
                    final_analysis[
                        "confidence"
                    ],

                "final_entropy":
                    final_analysis[
                        "entropy"
                    ],

                "verification_status":
                    final_verification[
                        "verification_status"
                    ],

                "evidence":
                    final_verification[
                        "sources"
                    ],

                "decision":
                    decision[
                        "reason"
                    ]
            }

            buffer_result = (
                self.buffer.add_experience(
                    experience
                )
            )

            print()
            print(
                "[Experience Buffer]",
                buffer_result
            )

            experience_added = (
                buffer_result[
                    "added"
                ]
            )

        # =====================================================
        # ROLLBACK
        # =====================================================

        else:

            final_answer = (
                initial_answer
            )

            accepted_analysis = (
                initial_analysis
            )

            accepted_verification = (
                initial_verification
            )

            experience_added = False

            print(
                "[Decision] "
                "Candidate rejected → "
                "original answer retained."
            )

        # =====================================================
        # RETURN RESULT
        # =====================================================

        return {

            "question":
                question,

            "initial_answer":
                initial_answer,

            "initial_analysis":
                initial_analysis,

            "initial_verification":
                initial_verification,

            "refined_answer":
                refined_answer,

            "candidate_analysis":
                final_analysis,

            "candidate_verification":
                final_verification,

            "final_answer":
                final_answer,

            "final_analysis":
                accepted_analysis,

            "final_verification":
                accepted_verification,

            "decision":
                decision,

            "experience_added":
                experience_added
        }

    # =========================================================
    # RUN
    # =========================================================

    def run(
        self,
        questions
    ):

        results = []

        for index, question in enumerate(
            questions
        ):

            print()
            print(
                f"PROCESSING {index + 1}/"
                f"{len(questions)}"
            )

            try:

                result = (
                    self.process_question(
                        question
                    )
                )

                result["index"] = index

                results.append(
                    result
                )

            except Exception as exc:

                print()
                print(
                    "[ERROR]",
                    str(exc)
                )

                results.append({
                    "index":
                        index,

                    "question":
                        question,

                    "error":
                        str(exc)
                })

        self.save_results(
            results
        )

        return results

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    @staticmethod
    def save_results(
        results,
        output_file=OUTPUT_FILE
    ):

        directory = os.path.dirname(
            output_file
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        temp_file = (
            output_file
            + ".tmp"
        )

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False
            )

        os.replace(
            temp_file,
            output_file
        )

        print()
        print(
            "[Results] Saved:",
            output_file
        )


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    questions = [

        "What is the capital of France?",

        "Can humans live without sleep?",

        "Is the Earth flat?",

        "Does drinking bleach cure infections?",

        "Do vaccines cause autism?"
    ]

    system = TrustGateV6()

    results = system.run(
        questions
    )

    print()
    print("=" * 70)
    print("V6 RUN COMPLETE")
    print("=" * 70)

    successful = sum(
        1
        for result in results
        if "error" not in result
    )

    failed = (
        len(results)
        - successful
    )

    accepted = sum(
        1
        for result in results
        if (
            result.get(
                "decision",
                {}
            ).get(
                "decision"
            )
            in [
                "ACCEPT",
                "ACCEPT_INITIAL"
            ]
        )
    )

    rolled_back = sum(
        1
        for result in results
        if (
            result.get(
                "decision",
                {}
            ).get(
                "decision"
            )
            == "ROLLBACK"
        )
    )

    experiences_added = sum(
        1
        for result in results
        if result.get(
            "experience_added",
            False
        )
    )

    print(
        f"Successful:          {successful}"
    )

    print(
        f"Failed:              {failed}"
    )

    print(
        f"Accepted candidates: {accepted}"
    )

    print(
        f"Rollbacks:            {rolled_back}"
    )

    print(
        f"Experiences stored:   {experiences_added}"
    )

    print("=" * 70)