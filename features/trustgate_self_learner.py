"""
TrustGate Autonomous Self-Learner
==================================
Coordinates autonomous experience accumulation, continuous memory integration,
and online retraining of the TrustGate 22-D trust representation model.

Closes the self-learning feedback loop:
Unreliable Answer -> External Fact-Checking -> Evidence-Grounded Correction ->
Experience Buffer -> Continuous Knowledge Base Update -> Model Retraining.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from features.trust_experience_buffer import TrustExperienceBuffer
from features.trustgate_knowledge_memory import TrustLearnedKnowledgeBase


BASE_MODEL_PATH = "models/trustgate_500_22d_linear_svm.joblib"
SELF_IMPROVED_MODEL_PATH = "models/trustgate_self_improved.joblib"
HUMAN_DATASET_PATH = "outputs/trustgate_standardized_500_labeled.json"


class TrustGateSelfLearner:
    """
    Autonomous self-improvement coordinator for TrustGate.
    """

    def __init__(
        self,
        buffer_path: str = "outputs/trust_experience_buffer.json",
        auto_retrain_threshold: int = 5,
    ):
        self.buffer = TrustExperienceBuffer(buffer_path=buffer_path)
        self.knowledge_base = TrustLearnedKnowledgeBase()
        self.auto_retrain_threshold = auto_retrain_threshold
        self._new_experiences_since_retrain = 0

    def record_experience(
        self,
        question: str,
        initial_answer: str,
        initial_label: str,
        initial_confidence: float,
        initial_entropy: float,
        refined_answer: str,
        final_label: str,
        final_confidence: float,
        final_entropy: float,
        verification_status: str,
        evidence: List[Dict[str, Any]],
        decision: str,
        hidden_state: Optional[np.ndarray] = None,
        detected_flaw: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a verified self-improvement cycle into both:
        1. Experience Buffer (for representation learning & classifier retraining)
        2. Knowledge Memory Bank (for instant recall & prompt grounding on future queries)
        """
        experience = {
            "question": question,
            "initial_answer": initial_answer,
            "initial_label": initial_label,
            "initial_confidence": float(initial_confidence),
            "initial_entropy": float(initial_entropy),
            "refined_answer": refined_answer,
            "final_label": final_label,
            "final_confidence": float(final_confidence),
            "final_entropy": float(final_entropy),
            "verification_status": verification_status,
            "evidence": evidence,
            "decision": decision,
            "experience_id": len(self.buffer.experiences),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "quality_status": "APPROVED" if decision in {"ACCEPT", "EVIDENCE_SUPPORTS_CANDIDATE"} else "PENDING",
            "hidden_state": hidden_state.tolist() if hidden_state is not None else None,
        }

        # Save to experience buffer
        self.buffer.experiences.append(experience)
        self.buffer.save()

        # Save to knowledge memory bank if verified correction or supported
        evidence_quotes = []
        source_names = []
        for src in evidence:
            source_names.append(src.get("title", src.get("source", "Evidence Source")))
            if isinstance(src.get("sentences"), list):
                evidence_quotes.extend(src.get("sentences")[:2])

        self.knowledge_base.store_verified_learning(
            question=question,
            initial_answer=initial_answer,
            verified_answer=refined_answer,
            evidence_quotes=evidence_quotes,
            detected_flaw=detected_flaw,
            sources=source_names,
        )

        self._new_experiences_since_retrain += 1
        print(f"[Self-Learner] Successfully logged self-improvement experience #{experience['experience_id']}.")

        # Check if auto-retraining threshold reached
        auto_retrained = False
        retrain_metrics = None
        if self._new_experiences_since_retrain >= self.auto_retrain_threshold:
            print("[Self-Learner] Auto-retraining threshold reached! Initiating continuous model evolution...")
            try:
                retrain_metrics = self.trigger_model_retraining()
                auto_retrained = True
            except Exception as exc:
                print(f"[Self-Learner] Retraining warning: {exc}")

        return {
            "logged": True,
            "experience_id": experience["experience_id"],
            "total_experiences": len(self.buffer.experiences),
            "auto_retrained": auto_retrained,
            "retrain_metrics": retrain_metrics,
        }

    def trigger_model_retraining(
        self,
        output_model_path: str = SELF_IMPROVED_MODEL_PATH,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Retrain the 22-D TrustGate classifier combining:
        1. Human-labeled baseline examples (500 standardized items)
        2. Verified self-improvement experiences accumulated over time.
        """
        print("=" * 60)
        print("TRIGGERING TRUSTGATE ONLINE MODEL RETRAINING")
        print("=" * 60)

        if not os.path.exists(HUMAN_DATASET_PATH):
            raise FileNotFoundError(f"Human baseline dataset not found at: {HUMAN_DATASET_PATH}")

        with open(HUMAN_DATASET_PATH, "r", encoding="utf-8") as f:
            human_data = json.load(f)

        # Baseline human features
        X_hidden_base = [item["hidden_state"] for item in human_data]
        X_signals_base = [[float(item["confidence"]), float(item["entropy"])] for item in human_data]
        y_base = [item["label"] for item in human_data]

        # Experience features
        valid_experiences = [
            exp for exp in self.buffer.experiences
            if exp.get("hidden_state") and exp.get("final_label")
        ]

        X_hidden_exp = [exp["hidden_state"] for exp in valid_experiences]
        X_signals_exp = [[float(exp["final_confidence"]), float(exp["final_entropy"])] for exp in valid_experiences]
        y_exp = [exp["final_label"] for exp in valid_experiences]

        # Combine datasets
        all_hidden = np.asarray(X_hidden_base + X_hidden_exp, dtype=np.float32)
        all_signals = np.asarray(X_signals_base + X_signals_exp, dtype=np.float32)
        all_y = np.asarray(y_base + y_exp)

        # 1. Standardize hidden vectors
        hidden_scaler = StandardScaler()
        hidden_scaled = hidden_scaler.fit_transform(all_hidden)

        # 2. PCA projection to 20-D
        pca = PCA(n_components=20, random_state=random_state)
        hidden_pca = pca.fit_transform(hidden_scaled)

        # 3. Fuse with confidence and entropy -> 22-D
        X_22d = np.hstack([hidden_pca, all_signals])

        # 4. Final scaling
        final_scaler = StandardScaler()
        X_22d_scaled = final_scaler.fit_transform(X_22d)

        # 5. Train LinearSVC with balanced class weights
        classifier = LinearSVC(
            class_weight="balanced",
            random_state=random_state,
            max_iter=10000,
        )
        classifier.fit(X_22d_scaled, all_y)

        preds = classifier.predict(X_22d_scaled)
        acc = float(accuracy_score(all_y, preds))
        balanced_acc = float(balanced_accuracy_score(all_y, preds))
        macro_f1 = float(f1_score(all_y, preds, average="macro"))

        # Bundle model
        bundle = {
            "model_type": "TrustGate 22-D Self-Improved LinearSVC",
            "input_hidden_dimension": 896,
            "pca_dimension": 20,
            "final_dimension": 22,
            "signal_features": ["confidence", "entropy"],
            "labels": list(np.unique(all_y)),
            "hidden_scaler": hidden_scaler,
            "pca": pca,
            "final_scaler": final_scaler,
            "classifier": classifier,
            "training_samples": len(all_y),
            "human_samples": len(human_data),
            "learned_experiences": len(valid_experiences),
            "training_accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "retrained_at": datetime.now(timezone.utc).isoformat(),
        }

        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        joblib.dump(bundle, output_model_path)
        self._new_experiences_since_retrain = 0

        print(f"[Self-Learner] Self-improved model successfully saved to: {output_model_path}")
        print(f"[Self-Learner] Accuracy: {acc:.4f} | Balanced Acc: {balanced_acc:.4f} | Macro F1: {macro_f1:.4f}")

        return {
            "model_path": output_model_path,
            "training_samples": len(all_y),
            "human_samples": len(human_data),
            "learned_experiences": len(valid_experiences),
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return metrics summarizing self-improvement progress."""
        total_exp = len(self.buffer.experiences)
        approved_exp = sum(1 for e in self.buffer.experiences if e.get("quality_status") == "APPROVED")
        knowledge_facts = len(self.knowledge_base.get_all_memories())

        # Count how many unreliable/uncertain answers were successfully turned into reliable
        conversions = sum(
            1 for e in self.buffer.experiences
            if e.get("initial_label") in {"UNRELIABLE", "UNCERTAIN"} and e.get("final_label") == "RELIABLE"
        )

        return {
            "total_experiences": total_exp,
            "approved_experiences": approved_exp,
            "verified_knowledge_memories": knowledge_facts,
            "successful_conversions": conversions,
            "active_model": (
                SELF_IMPROVED_MODEL_PATH
                if os.path.exists(SELF_IMPROVED_MODEL_PATH)
                else BASE_MODEL_PATH
            ),
        }
