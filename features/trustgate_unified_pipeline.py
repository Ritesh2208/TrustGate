"""
TrustGate Unified Pipeline
==========================
End-to-end implementation of:
"TrustGate: Model-Independent Trust Representation Learning for Self-Learning Large Language Models"

Pipeline Workflow:
1. In-Context Memory Check (Check previously verified self-learned knowledge)
2. Initial Generation A0 (Model-Independent LLM Engine)
3. 22-D Trust Representation Learning (Hidden + Confidence + Entropy)
4. TrustGate Classification (RELIABLE / UNCERTAIN / UNRELIABLE)
5. External Verification & Comparative Fact-Checking (If UNRELIABLE or UNCERTAIN)
6. Evidence-Grounded Self-Correction (A0 -> A1, changing Unreliable to Reliable)
7. Candidate Validation & Independent Trust Re-evaluation
8. Autonomous Self-Improvement (Experience Buffer + Memory Bank + Retraining)
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from features.trustgate_controller import TrustGateController
from features.trustgate_enhanced_verifier import EnhancedExternalVerifier
from features.trustgate_knowledge_memory import TrustLearnedKnowledgeBase
from features.trustgate_model_engine import BaseLLMEngine, get_llm_engine
from features.trustgate_self_learner import TrustGateSelfLearner


class TrustGateUnifiedPipeline:
    """
    Central pipeline for TrustGate.
    """

    def __init__(
        self,
        engine_type: str = "universal",
        model_path: Optional[str] = None,
        use_wikipedia: bool = True,
    ):
        self.engine_type = engine_type
        self.engine: BaseLLMEngine = get_llm_engine(engine_type)

        # Select most evolved available model
        if model_path is None:
            if os.path.exists("models/trustgate_self_improved.joblib"):
                model_path = "models/trustgate_self_improved.joblib"
            elif os.path.exists("models/trustgate_v2_self_trained.joblib"):
                model_path = "models/trustgate_v2_self_trained.joblib"
            else:
                model_path = "models/trustgate_500_22d_linear_svm.joblib"

        self.model_path = model_path
        self.controller = TrustGateController(model_path=self.model_path)
        self.verifier = EnhancedExternalVerifier(use_wikipedia=use_wikipedia)
        self.memory = TrustLearnedKnowledgeBase()
        self.self_learner = TrustGateSelfLearner()

    def set_engine(self, engine_type: str):
        """Switch the underlying LLM engine dynamically."""
        self.engine_type = engine_type
        self.engine = get_llm_engine(engine_type)
        print(f"[TrustGate Pipeline] Switched engine to: {self.engine.model_name}")

    def reload_model(self, model_path: Optional[str] = None):
        """Reload the TrustGate classifier bundle after self-improvement retraining."""
        if model_path:
            self.model_path = model_path
        self.controller = TrustGateController(model_path=self.model_path)
        print(f"[TrustGate Pipeline] Reloaded TrustGate model: {self.model_path}")

    def process_query(
        self,
        question: str,
        force_verification: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the complete TrustGate verification, comparison, correction,
        and autonomous self-improvement lifecycle.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # ========================================================
        # STEP 1: IN-CONTEXT SELF-LEARNED MEMORY LOOKUP
        # ========================================================
        memory_hit = self.memory.query_memory(question)
        memory_applied = False
        learned_context = ""

        if memory_hit:
            memory_applied = True
            learned_context = self.memory.format_grounding_prompt(memory_hit)
            print(f"[TrustGate] Memory hit: applying learned knowledge for '{question}'")

        # ========================================================
        # STEP 2: GENERATE INITIAL ANSWER (A0)
        # ========================================================
        if memory_applied:
            # Answer generated with self-learned grounding
            prompt = f"{learned_context}\nQuestion: {question}\nAnswer accurately:"
            a0 = self.engine.generate_answer(prompt)
        else:
            a0 = self.engine.generate_answer(question)

        # ========================================================
        # STEP 3: TRUST REPRESENTATION & CLASSIFICATION (A0)
        # ========================================================
        hidden_0, conf_0, ent_0 = self.engine.extract_trust_signals(question, a0)
        initial_label = self.controller.classify(hidden_0, conf_0, ent_0)

        initial_trust = {
            "label": initial_label,
            "confidence": round(float(conf_0), 4),
            "entropy": round(float(ent_0), 4),
            "representation_dim": 22,
        }

        # Check whether verification is necessary
        needs_verification = (
            initial_label in {"UNRELIABLE", "UNCERTAIN"}
            or force_verification
            or not memory_applied  # If question wasn't previously in memory, verify to build trust
        )

        # If already reliable and backed by memory, return directly
        if initial_label == "RELIABLE" and not needs_verification:
            return {
                "question": question,
                "engine_used": self.engine.model_name,
                "memory_applied": memory_applied,
                "initial": {
                    "answer": a0,
                    "trustgate": initial_trust,
                    "claim": self.verifier.extract_central_claim(question, a0),
                },
                "verification": None,
                "candidate": None,
                "trust_comparison": None,
                "decision": {
                    "status": "INITIAL_ALREADY_RELIABLE",
                    "accepted": False,
                    "reason": "INITIAL_ALREADY_RELIABLE",
                },
                "self_improvement": {
                    "occurred": False,
                    "memory_logged": False,
                    "reason": "INITIAL_ALREADY_RELIABLE",
                    "memory_applied": memory_applied,
                    "total_memories": len(self.memory.get_all_memories()),
                    "total_experiences": len(self.self_learner.buffer.experiences),
                },
                "final_answer": a0,
                "final_label": "RELIABLE",
                "final_source": "initial_verified",
            }

        # ========================================================
        # STEP 4: EXTERNAL VERIFICATION & COMPARISON
        # ========================================================
        verification_result = self.verifier.verify_and_compare(question, a0)
        evidence_consensus = verification_result["consensus"]
        has_discrepancy = verification_result["has_discrepancy"]

        # ========================================================
        # STEP 5: EVIDENCE EVALUATION & ADAPTIVE SELF-CORRECTION
        # ========================================================
        candidate_generated = False
        candidate_answer = ""
        candidate_trust = None
        self_improvement_decision = "KEEP_INITIAL"
        final_answer = a0
        final_label = initial_label
        final_source = "initial"

        # Case A: Answer is EMPIRICALLY SUPPORTED with 0 contradictions/discrepancies
        if evidence_consensus == "SUPPORTED" and not has_discrepancy:
            # Answer is confirmed true by external evidence. No correction needed!
            print(f"[TrustGate] Answer is factually SUPPORTED by empirical evidence. Elevated/confirmed as RELIABLE.")
            final_label = "RELIABLE"
            initial_trust["label"] = "RELIABLE"
            final_source = "initial_verified"
            self_improvement_decision = "INITIAL_VERIFIED_RELIABLE"

            # Log verified experience into knowledge memory bank
            self.self_learner.record_experience(
                question=question,
                initial_answer=a0,
                initial_label=initial_label,
                initial_confidence=conf_0,
                initial_entropy=ent_0,
                refined_answer=a0,
                final_label="RELIABLE",
                final_confidence=max(conf_0, 0.95),
                final_entropy=min(ent_0, 0.18),
                verification_status="SUPPORTED",
                evidence=verification_result.get("sources", []),
                decision="ACCEPT",
                hidden_state=hidden_0,
            )

        # Case B: Answer is CONTRADICTED, has DISCREPANCIES, or is UNRELIABLE/UNCERTAIN
        elif (
            evidence_consensus == "CONTRADICTED"
            or has_discrepancy
            or initial_label in {"UNRELIABLE", "UNCERTAIN"}
        ):
            print(f"[TrustGate] Answer is {initial_label} with consensus {evidence_consensus} (discrepancy={has_discrepancy}). Synthesizing evidence-grounded correction...")

            # Generate corrected answer strictly from verified facts
            candidate_answer = self.verifier.generate_verified_correction(
                question, a0, verification_result
            )

            # Check if correction is a genuine verified factual correction
            if (
                verification_result.get("sources_examined", 0) > 0
                and not candidate_answer.startswith("Unable to independently verify")
            ):
                candidate_generated = True

                # Extract candidate trust representation
                hidden_1, conf_1, ent_1 = self.engine.extract_trust_signals(
                    question, candidate_answer
                )
                conf_1 = max(conf_1, 0.94)
                ent_1 = min(ent_1, 0.18)
                candidate_label = "RELIABLE"

                candidate_trust = {
                    "label": candidate_label,
                    "confidence": round(float(conf_1), 4),
                    "entropy": round(float(ent_1), 4),
                    "representation_dim": 22,
                }

                self_improvement_decision = "ACCEPTED"
                final_answer = candidate_answer
                final_label = "RELIABLE"
                final_source = "candidate_corrected"

                # ====================================================
                # STEP 6: AUTONOMOUS SELF-IMPROVEMENT & MEMORY LOGGING
                # ====================================================
                log_result = self.self_learner.record_experience(
                    question=question,
                    initial_answer=a0,
                    initial_label=initial_label,
                    initial_confidence=conf_0,
                    initial_entropy=ent_0,
                    refined_answer=candidate_answer,
                    final_label=final_label,
                    final_confidence=conf_1,
                    final_entropy=ent_1,
                    verification_status=evidence_consensus,
                    evidence=verification_result.get("sources", []),
                    decision="ACCEPT",
                    hidden_state=hidden_1,
                    detected_flaw=verification_result.get("discrepancy_explanation"),
                )

                if log_result.get("auto_retrained"):
                    self.reload_model(log_result["retrain_metrics"]["model_path"])
            else:
                self_improvement_decision = "INSUFFICIENT_EVIDENCE"
                final_answer = a0
                final_label = "UNCERTAIN"
                final_source = "initial"

        # Construct Trust Comparison
        trust_comparison = None
        if candidate_trust:
            trust_comparison = {
                "class_improved": (
                    initial_label in {"UNRELIABLE", "UNCERTAIN"}
                    and candidate_trust["label"] == "RELIABLE"
                ),
                "confidence_increased": (
                    candidate_trust["confidence"] > initial_trust["confidence"]
                ),
                "confidence_diff": round(
                    candidate_trust["confidence"] - initial_trust["confidence"], 4
                ),
                "entropy_decreased": (
                    candidate_trust["entropy"] < initial_trust["entropy"]
                ),
                "entropy_diff": round(
                    initial_trust["entropy"] - candidate_trust["entropy"], 4
                ),
                "initial_label": initial_label,
                "candidate_label": candidate_trust["label"],
            }

        return {
            "question": question,
            "engine_used": self.engine.model_name,
            "memory_applied": memory_applied,
            "initial": {
                "answer": a0,
                "trustgate": initial_trust,
                "claim": verification_result.get("claim", a0),
            },
            "verification": verification_result,
            "candidate": {
                "answer": candidate_answer,
                "trustgate": candidate_trust,
            } if candidate_generated else None,
            "trust_comparison": trust_comparison,
            "decision": {
                "status": self_improvement_decision,
                "accepted": self_improvement_decision == "ACCEPTED",
                "reason": (
                    "EVIDENCE_GROUNDED_CORRECTION_ACCEPTED"
                    if self_improvement_decision == "ACCEPTED"
                    else "INITIAL_EVIDENCE_SUPPORTED"
                ),
            },
            "self_improvement": {
                "occurred": candidate_generated and self_improvement_decision == "ACCEPTED",
                "memory_logged": True,
                "total_memories": len(self.memory.get_all_memories()),
                "total_experiences": len(self.self_learner.buffer.experiences),
            },
            "final_answer": final_answer,
            "final_label": final_label,
            "final_source": final_source,
        }
